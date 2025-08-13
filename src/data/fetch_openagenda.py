"""
City-only fetcher for the OpenDataSoft dataset: 'evenements-publics-openagenda'.

What this module does:
- Filters by *city* only (project scope) using the public Opendatasoft Search API.
- Respects the API's 10,000 results "window" (offset + limit <= 10_000) to avoid HTTP 400.
- Keeps only events that are still relevant: END (preferred) or START is within the last 365 days.
- Strips HTML from long descriptions for better embedding quality.
- Normalizes URLs from the source so the chat layer can safely whitelist them.
- Returns a *stable schema* the indexer expects.

Outputs DataFrame columns:
    uid, title, text, city, venue, start_utc, end_utc, tags, website, permalink
"""

from datetime import datetime, timedelta, timezone
import unicodedata
from urllib.parse import urlparse

import requests
import pandas as pd

# ---- API constants ----
OPENAGENDA_DATASET = "evenements-publics-openagenda"
BASE_URL = "https://public.opendatasoft.com/api/records/1.0/search/"

# Opendatasoft Search API hard window: offset + rows <= 10,000
MAX_WINDOW = 10_000
# Page size: 500 avoids throttling but is still efficient
DEFAULT_ROWS_PER_PAGE = 500


def _norm(s: str | None) -> str:
    """Accent/case-insensitive normalization (e.g., 'Île' -> 'ile')."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold().strip()


def _clean_html(txt: str | None) -> str:
    """Strip HTML tags and collapse whitespace (requires beautifulsoup4)."""
    if not isinstance(txt, str):
        return ""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(txt, "html.parser").get_text(" ").strip()
    except Exception:
        return txt.strip()


def _normalize_url(u: str | None) -> str | None:
    """Return a clean http(s) URL or None."""
    if not isinstance(u, str):
        return None
    u = u.strip()
    try:
        p = urlparse(u)
        if p.scheme in ("http", "https") and p.netloc:
            return u
    except Exception:
        pass
    return None


def _within_last_year(start_iso: str | None, end_iso: str | None) -> bool:
    """
    Keep events whose END (if present) or START >= now-365d (UTC).
    Using END favors multi-day events that are still recent at their end date.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365)
    s = pd.to_datetime(start_iso, utc=True, errors="coerce") if start_iso else None
    e = pd.to_datetime(end_iso,   utc=True, errors="coerce") if end_iso   else None
    candidate = e if e is not None else s
    return bool(candidate is not None and candidate >= cutoff)


def _map_record(rec: dict) -> dict:
    """
    Map a raw Opendatasoft record to our normalized schema.
    Handles common field variants across agendas.
    """
    f = rec.get("fields", {}) or {}

    # Titles (best available)
    title = f.get("title_fr") or f.get("title_en") or f.get("title") or f.get("longtitle") or ""

    # Descriptions (HTML stripped)
    desc_raw = f.get("longdescription_fr") or f.get("description") or f.get("free_text") or f.get("body") or ""
    desc = _clean_html(desc_raw)

    # Location
    city  = f.get("location_city") or f.get("city") or ""
    venue = f.get("placename") or f.get("location_name") or ""

    # Dates (common variants)
    start = f.get("date_start") or f.get("firstdate_begin") or f.get("firstdate")
    end   = f.get("date_end")   or f.get("lastdate_end")   or f.get("lastdate")

    # Normalize to ISO strings (UTC)
    start_iso = str(pd.to_datetime(start, utc=True, errors="coerce")) if start else None
    end_iso   = str(pd.to_datetime(end,   utc=True, errors="coerce")) if end   else None

    # Tags / Link (normalize URL)
    tags_list = f.get("keywords_fr") or f.get("keywords_en") or []
    tags = ", ".join([t for t in tags_list if isinstance(t, str)])
    link_raw = f.get("permalink") or f.get("link") or f.get("website")
    link = _normalize_url(link_raw)

    # Text used downstream for embeddings (the indexer augments with date tokens)
    text = (title + "\n\n" + desc).strip()

    return {
        "uid": rec.get("recordid"),
        "title": title,
        "text": text,
        "city": city,
        "venue": venue,
        "start_utc": start_iso,
        "end_utc": end_iso,
        "tags": tags,
        "website": link,
        "permalink": link,
    }


def _initial_nhits(city: str | None) -> int:
    """Ask for nhits (total items) with a 0-row query."""
    params = {"dataset": OPENAGENDA_DATASET, "rows": 0}
    if city:
        params["refine.location_city"] = city
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return int(r.json().get("nhits", 0))


def fetch_events_df(
    city: str | None = None,
    max_records: int = 5000,
    rows_per_page: int = DEFAULT_ROWS_PER_PAGE
) -> pd.DataFrame:
    """
    Fetch and normalize city-filtered events from the last 12 months.

    Args:
        city: e.g. "Paris" (accent/case-insensitive check is done client-side too)
        max_records: upper bound on returned rows (keeps runtime reasonable)
        rows_per_page: page size for API calls (default 500)

    Returns:
        DataFrame with columns: uid, title, text, city, venue, start_utc, end_utc, tags, website, permalink
    """
    nhits = _initial_nhits(city)
    if nhits == 0:
        return pd.DataFrame(columns=[
            "uid","title","text","city","venue","start_utc","end_utc","tags","website","permalink"
        ])

    # Respect both the API window and caller limit
    target = min(nhits, MAX_WINDOW, max_records if max_records > 0 else MAX_WINDOW)

    recs: list[dict] = []
    start = 0

    while start < target:
        # Keep offset+limit within the 10k window
        rows = min(rows_per_page, target - start, MAX_WINDOW - start)
        if rows <= 0:
            break

        params = {"dataset": OPENAGENDA_DATASET, "rows": rows, "start": start}
        if city:
            params["refine.location_city"] = city

        r = requests.get(BASE_URL, params=params, timeout=30)
        # If we accidentally cross the 10k boundary, stop cleanly
        if r.status_code == 400 and "10000" in r.text:
            break
        r.raise_for_status()

        records = r.json().get("records", []) or []
        if not records:
            break

        for rec in records:
            mapped = _map_record(rec)
            if _within_last_year(mapped["start_utc"], mapped["end_utc"]):
                recs.append(mapped)
                if len(recs) >= max_records:
                    break

        if len(recs) >= max_records:
            break
        if len(records) < rows:
            break

        start += rows

    df = pd.DataFrame(recs)
    if df.empty:
        return df

    # Extra client-side guard for city normalization (accent/case-insensitive)
    if city:
        ncity = _norm(city)
        df = df[df["city"].map(lambda x: _norm(str(x)) == ncity)]

    # Deduplicate
    df = df.drop_duplicates(subset=["uid","title","start_utc"], keep="first").reset_index(drop=True)
    return df
