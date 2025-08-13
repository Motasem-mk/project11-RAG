"""
Build a FAISS index (Option 2: embed normalized date tokens in the text).

Why this helps:
- Dense retrieval is fuzzy with numbers/dates.
- We inject normalized tokens (YearStart/MonthStart/YearEnd/MonthEnd) into the page_content,
  so queries like "June 2025 in Paris" match semantically *without* metadata filters.

Run:
    python -m src.index.build_index --city Paris --out data/index/faiss --max-records 9000
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.data.fetch_openagenda import fetch_events_df


def _norm_parts(iso_str: str | None):
    """Return (iso, year_str, month_2digit_str) or empty strings on parse failure."""
    if not iso_str:
        return None, "", ""
    try:
        ts = pd.to_datetime(iso_str, utc=True, errors="raise")
        return iso_str, str(ts.year), f"{ts.month:02d}"
    except Exception:
        return iso_str, "", ""


def df_to_documents(df: pd.DataFrame):
    """Create LangChain Documents with date tokens embedded in page_content."""
    docs = []
    for _, row in df.iterrows():
        start_iso, y_start, m_start = _norm_parts(row.get("start_utc"))
        end_iso, y_end, m_end = _norm_parts(row.get("end_utc"))

        page_text = (
            f"Title: {row.get('title')}\n"
            f"City: {row.get('city')}\n"
            f"Venue: {row.get('venue')}\n"
            f"Start: {start_iso}\n"
            f"End: {end_iso}\n"
            f"YearStart: {y_start}\n"
            f"MonthStart: {m_start}\n"
            f"YearEnd: {y_end}\n"
            f"MonthEnd: {m_end}\n"
            f"Tags: {row.get('tags')}\n\n"
            f"{row.get('text', '')}"
        )

        metadata = {
            "uid": row.get("uid"),
            "title": row.get("title"),
            "city": row.get("city"),
            "venue": row.get("venue"),
            "website": row.get("website"),
            "start_utc": start_iso,
            "end_utc": end_iso,
            "tags": row.get("tags"),
            "permalink": row.get("permalink"),
        }
        docs.append(Document(page_content=page_text, metadata=metadata))
    return docs


def main():
    """Fetch -> split -> embed -> FAISS -> save."""
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set. Create a .env at project root with:\n"
            "MISTRAL_API_KEY=your_actual_mistral_api_key_here"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default=None, help="City filter, e.g. 'Paris'")
    parser.add_argument("--out", type=str, default="data/index/faiss", help="Folder to save FAISS index")
    parser.add_argument("--max-records", type=int, default=5000, help="Upper bound on fetched rows")
    parser.add_argument("--chunk-size", type=int, default=800, help="Text chunk size for embedding")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap between chunks")
    args = parser.parse_args()

    df = fetch_events_df(city=args.city, max_records=args.max_records)
    if df.empty:
        raise SystemExit("No events fetched. Try another city or adjust --max-records.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    docs = df_to_documents(df)
    chunks = splitter.split_documents(docs)

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    os.makedirs(args.out, exist_ok=True)
    vectordb.save_local(args.out)
    print(f"Saved FAISS index to: {args.out} (chunks={len(chunks)}, docs={len(docs)}, rows={len(df)})")


if __name__ == "__main__":
    main()
