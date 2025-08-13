"""
Verify that data in the FAISS index:
  (1) is within the last 12 months, and
  (2) belongs to a single selected city.

Works with both legacy and newer FAISS pickle formats.
"""

import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import Optional

# Needed so pickle can import the same classes used when the index was saved
import langchain_community  # noqa: F401

import pandas as pd


INDEX_DIR = os.environ.get("INDEX_DIR", "data/index/faiss")
PICKLE_PATH = os.path.join(INDEX_DIR, "index.pkl")


def _norm(s: Optional[str]) -> str:
    """Lowercase + strip accents for robust city equality."""
    import unicodedata
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold().strip()


def _load_docstore_from_pickle(pickle_path: str):
    """
    Return an object with a _dict {id: Document}, handling both legacy dict
    and newer tuple index.pkl formats produced by LangChain/FAISS.
    """
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)

    # Legacy dict format
    if isinstance(payload, dict):
        docstore = payload.get("docstore") or payload.get("_docstore")
        if docstore is not None:
            return docstore
        # Sometimes raw dict is stored directly
        raw = payload.get("_dict") or payload.get("dict")
        if isinstance(raw, dict) and raw:
            class _Wrapper:
                _dict = raw
            return _Wrapper()

    # Newer tuple format (e.g., (docstore, index_to_docstore_id, ...))
    if isinstance(payload, tuple):
        for item in payload:
            if hasattr(item, "_dict"):
                return item
        for item in payload:
            if isinstance(item, dict) and item:
                class _Wrapper:
                    _dict = item
                return _Wrapper()

    raise AssertionError("Unrecognized FAISS pickle structure; cannot locate docstore.")


def test_index_data_constraints_last_year_and_city_consistency():
    # Ensure the index exists
    assert os.path.exists(PICKLE_PATH), (
        f"FAISS pickle not found at {PICKLE_PATH}. "
        "Build the index first with build_index.py."
    )

    # Load docstore in a version-agnostic way
    docstore = _load_docstore_from_pickle(PICKLE_PATH)
    raw = getattr(docstore, "_dict", None)
    assert isinstance(raw, dict) and len(raw) > 0, "Docstore is empty."

    docs = list(raw.values())

    # 1) All events are within the last 12 months (by end_utc if present, else start_utc)
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=365)

    def in_last_year(doc) -> bool:
        s = pd.to_datetime(doc.metadata.get("start_utc"), utc=True, errors="coerce")
        e = pd.to_datetime(doc.metadata.get("end_utc"),   utc=True, errors="coerce")
        cand = e if pd.notna(e) else s
        return bool(pd.notna(cand) and cand >= cutoff)

    assert all(in_last_year(d) for d in docs), (
        "Found events older than 12 months in the FAISS index."
    )

    # 2) All documents belong to the same city (normalized)
    cities = [d.metadata.get("city") for d in docs if d.metadata.get("city")]
    assert len(cities) > 0, "No city metadata found in documents."

    normalized = {_norm(c) for c in cities}
    assert len(normalized) == 1, (
        f"Multiple cities found in index: {sorted(normalized)}. "
        "Ensure you built the index with a single --city."
    )
