"""
Minimal, retrieval-first CLI chatbot (Option 2).

Behavior:
- Answers ONLY with facts found in retrieved CONTEXT (no hallucinations).
- If info isn’t in CONTEXT, says “I don't know.”
- No metadata filters (we rely on dates embedded in page_content).
- Month-aware query augmentation: for “June 2025”, append normalized tokens and raise k.
- Link sanitizer: ONLY keep URLs that are present in the retrieved CONTEXT.
- "Upcoming" = from tomorrow (Europe/Paris) onward (applied post-retrieval).

Run:
    python -m src.chat_demo --index data/index/faiss --k 12
"""

import os
import re
import argparse
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from langdetect import detect
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---- Strict system prompt: no fabricated links ----
BASE_SYSTEM = """You are an events assistant.
Use ONLY the provided CONTEXT to answer. If the answer is not present in the CONTEXT, say: "I don't know." Do not invent details.
Do NOT fabricate URLs. Only include URLs that are explicitly present in the CONTEXT (on the 'Link:' lines). If no URL is present, omit the link."""
USER_PROMPT = """Question ({lang}): {question}

CONTEXT:
{context}

Respond concisely with facts from the CONTEXT. Include titles, dates, venues, and links only if present in the CONTEXT."""

SUPPORTED_LANGS = {"en", "fr", "de", "es", "ar", "it", "nl", "pt"}

# Month dictionaries (EN + FR, common variants)
MONTHS_EN = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
MONTHS_FR = {
    "janvier": 1, "janv": 1, "février": 2, "fevrier": 2, "févr": 2, "fevr": 2,
    "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "août": 8, "aout": 8, "septembre": 9, "sept": 9, "octobre": 10, "oct": 10,
    "novembre": 11, "nov": 11, "décembre": 12, "decembre": 12, "déc": 12, "dec": 12,
}
MONTH_LOOKUP = {**MONTHS_EN, **MONTHS_FR}


def safe_detect_lang(text: str, default_lang: str = "en") -> str:
    """Formatting-only language detection with a safe fallback."""
    if len(text.strip()) < 3 or len(text.split()) < 3:
        return default_lang
    try:
        code = detect(text)
        return code if code in SUPPORTED_LANGS else default_lang
    except Exception:
        return default_lang


def _extract_years(text: str):
    """Extract 4-digit years (20xx) from the query text."""
    try:
        return {int(y) for y in re.findall(r"\b(20\d{2})\b", text)}
    except Exception:
        return set()


def _extract_target_year_months(q: str, default_year: int):
    """Detect month mentions and pair them with explicit year(s) or the current year."""
    lower = q.lower()
    months = {MONTH_LOOKUP[m] for m in MONTH_LOOKUP if m in lower}
    if not months:
        return set()
    years = _extract_years(q) or {default_year}
    return {(y, m) for y in years for m in months}


def _augment_query_with_norm_dates(q: str, ym_pairs):
    """Append normalized tokens embedded in page_content to boost dense retrieval."""
    parts = [q]
    for (yy, mm) in sorted(ym_pairs):
        parts.append(f"YearStart: {yy} MonthStart: {mm:02d}")
        parts.append(f"YearEnd: {yy} MonthEnd: {mm:02d}")
        parts.append(f"{yy}-{mm:02d}")
    return " | ".join(parts)


def sanitize_links(answer: str, allowed_urls: set[str]) -> str:
    """
    Strip any URLs not present in allowed_urls (built from retrieved docs).
    Keeps link text; removes external/fabricated URLs.
    """
    if not answer:
        return answer

    # 1) Markdown links: [text](url)
    def _md_sub(m):
        text, url = m.group(1), m.group(2)
        return m.group(0) if url in allowed_urls else text

    answer = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", _md_sub, answer)

    # 2) Bare URLs
    def _bare_sub(m):
        url = m.group(0)
        return url if url in allowed_urls else ""

    answer = re.sub(r"https?://\S+", _bare_sub, answer)
    return answer.strip()


# ---- Upcoming-only helpers (Paris time) ----
def wants_upcoming(q: str) -> bool:
    """
    Heuristic: detect if the user is asking for future/upcoming items.
    Covers English & French phrasing.
    """
    ql = q.lower()
    triggers = [
        "upcoming", "coming up", "in the future", "future events", "next events",
        "à venir", "a venir", "prochains", "bientôt", "bientot", "futurs"
    ]
    return any(t in ql for t in triggers)


def paris_tomorrow_start_utc() -> datetime:
    """Start of 'tomorrow' in Europe/Paris, converted to UTC."""
    paris = ZoneInfo("Europe/Paris")
    today_paris = datetime.now(paris).date()
    tomorrow_paris_midnight = datetime.combine(today_paris + timedelta(days=1), time(0, 0), tzinfo=paris)
    return tomorrow_paris_midnight.astimezone(ZoneInfo("UTC"))


def is_upcoming_doc(doc, cutoff_utc: datetime) -> bool:
    """Keep docs whose start or end is >= cutoff_utc (aware, UTC)."""
    s = pd.to_datetime(doc.metadata.get("start_utc"), utc=True, errors="coerce")
    e = pd.to_datetime(doc.metadata.get("end_utc"), utc=True, errors="coerce")
    keep = False
    if (s is not pd.NaT) and not pd.isna(s) and s >= cutoff_utc:
        keep = True
    if (e is not pd.NaT) and not pd.isna(e) and e >= cutoff_utc:
        keep = True
    return keep


def main():
    """Load FAISS, then run an interactive chat loop."""
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY missing. Put it in .env at project root.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="data/index/faiss", help="Folder containing the FAISS index")
    parser.add_argument("--k", type=int, default=12, help="Top-k chunks to retrieve (raise for big cities)")
    parser.add_argument("--answer-lang", type=str, default="auto", help="auto|en|fr|... (formatting only)")
    parser.add_argument("--fallback-lang", type=str, default="en", help="Fallback answer language")
    args = parser.parse_args()

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    vectordb = FAISS.load_local(args.index, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": args.k})
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1, api_key=api_key)

    print("Type your questions (Ctrl+C to quit)\n")
    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue

            # Decide answer language (cosmetic)
            out_lang = args.answer_lang.strip() if args.answer_lang.lower() != "auto" else safe_detect_lang(q, args.fallback_lang)

            # Build prompt (strictly from context)
            system_prompt = f"{BASE_SYSTEM} Always answer in {out_lang}."
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", USER_PROMPT)])

            # Month-aware augmentation (no metadata filters)
            default_year = datetime.now(ZoneInfo("Europe/Paris")).year
            ym_pairs = _extract_target_year_months(q, default_year)

            if ym_pairs:
                k_eff = max(args.k, 30)  # raise k for date-specific queries
                q_aug = _augment_query_with_norm_dates(q, ym_pairs)
                docs = vectordb.similarity_search(q_aug, k=k_eff)
            else:
                docs = retriever.invoke(q)

            # If the user asked for upcoming items, keep only events from 'tomorrow' onward (Paris time).
            if wants_upcoming(q):
                cutoff = paris_tomorrow_start_utc()
                filtered = [d for d in docs if is_upcoming_doc(d, cutoff)]
                if filtered:
                    docs = filtered
                else:
                    print("\nBot: No upcoming events from tomorrow. Showing relevant results from the last 12 months instead.\n")
                    # fall back to unfiltered docs

            if not docs:
                print(
                    "\nBot: I couldn’t find matching results in the current OpenAgenda index. "
                    "We index events from the last 12 months and any future events already listed. "
                    "If you expect a specific date/month/year, it may not be present in this dataset or in the top results.\n"
                )
                continue

            # Build CONTEXT string and collect allowed URLs from metadata
            allowed_urls: set[str] = set()
            context_parts = []
            for d in docs:
                link = d.metadata.get("permalink") or d.metadata.get("website")
                if isinstance(link, str) and link.startswith("http"):
                    allowed_urls.add(link)

                context_parts.append(
                    f"Title: {d.metadata.get('title')}\n"
                    f"City: {d.metadata.get('city')}\n"
                    f"Start: {d.metadata.get('start_utc')} | End: {d.metadata.get('end_utc')}\n"
                    f"Link: {link}\n"
                    f"{d.page_content[:800]}"
                )
            context = "\n\n---\n\n".join(context_parts)

            msg = prompt.format_messages(question=q, context=context, lang=out_lang)
            resp = llm.invoke(msg)
            clean = sanitize_links(resp.content, allowed_urls)
            print(f"\nBot: {clean}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
