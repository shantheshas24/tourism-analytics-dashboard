"""
sentiment.py
============
Sentiment Analysis on Tourism Reviews.

This script:
  1. Loads the dataset (using chunked reading for memory efficiency).
  2. Cleans the review text (lowercasing, punctuation removal, stopword stripping).
  3. Runs TextBlob sentiment analysis on the cleaned text.
  4. Assigns Positive / Negative / Neutral labels.
  5. Computes per-city and per-place sentiment summaries.
  6. Saves sentiment results to outputs/.

NOTE: TextBlob is a lexicon-based analyser — no training data needed.
      We use the Raw_Review column when Review is null, and fall back to
      an empty string otherwise, to maximise coverage.

Save to: BDT_project/scripts/sentiment.py
Run with: python scripts/sentiment.py
"""

import os
import sys
import re
import logging
import warnings
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Download NLTK corpora (one-time) ─────────────────────────────────────────
for resource in ["stopwords", "punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data",    "Review_db.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CHUNKSIZE  = 200_000
# Polarity thresholds for label assignment
POS_THRESHOLD =  0.05
NEG_THRESHOLD = -0.05


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load only necessary columns for sentiment analysis."""
    log.info(f"Loading dataset from: {filepath}")
    usecols   = ["City", "Place", "Review", "Rating", "Raw_Review"]
    dtype_map = {
        "City":       "category",
        "Place":      "category",
        "Rating":     "float32",
        "Review":     "object",
        "Raw_Review": "object",
    }
    chunks = []
    total  = 0
    for chunk in pd.read_csv(
        filepath,
        usecols=usecols,
        dtype=dtype_map,
        chunksize=CHUNKSIZE,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=True,
    ):
        chunk = chunk.dropna(subset=["Rating"])
        chunk["Rating"] = chunk["Rating"].clip(1.0, 5.0)
        chunks.append(chunk)
        total += len(chunk)
        log.info(f"  {total:,} rows loaded …")

    df = pd.concat(chunks, ignore_index=True)
    for col in ["City", "Place"]:
        df[col] = df[col].astype("category")
    log.info(f"Loaded {len(df):,} rows.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════
_PUNCT_RE   = re.compile(r"[^a-z\s]")
_SPACE_RE   = re.compile(r"\s+")

def clean_text(text) -> str:
    """
    Lightweight text cleaner:
      - Coerce to string (handles NaN gracefully)
      - Lowercase
      - Remove non-alphabetic characters
      - Collapse whitespace
      - Remove English stopwords
    Stopword removal is skipped intentionally for TextBlob because
    polarity uses the full phrase context; we keep it as an option below.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def build_review_text(row) -> str:
    """
    Choose the best available review column.
    Preference order: Review > Raw_Review > empty string.
    """
    rv = row.get("Review", "")
    if isinstance(rv, str) and rv.strip():
        return clean_text(rv)
    rr = row.get("Raw_Review", "")
    if isinstance(rr, str) and rr.strip():
        return clean_text(rr)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# 3. SENTIMENT ANALYSIS  (vectorised batch approach)
# ══════════════════════════════════════════════════════════════════════════════
def _get_polarity(text: str) -> float:
    """Return TextBlob polarity score in [-1, 1]."""
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity


def _get_subjectivity(text: str) -> float:
    """Return TextBlob subjectivity score in [0, 1]."""
    if not text:
        return 0.0
    return TextBlob(text).sentiment.subjectivity


def polarity_to_label(polarity: float) -> str:
    """Map a polarity score to a human-readable sentiment label."""
    if polarity > POS_THRESHOLD:
        return "Positive"
    elif polarity < NEG_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main sentiment analysis function.

    Strategy for 1.48 M rows:
      - Build cleaned text column first.
      - Use pandas apply on the text series (single-threaded TextBlob is the
        most reliable; multiprocessing adds overhead due to pickle cost).
      - Process in memory-efficient fashion.
    """
    log.info("Building cleaned review text …")
    df["clean_text"] = df.apply(build_review_text, axis=1)

    # How many reviews have usable text?
    non_empty = (df["clean_text"] != "").sum()
    log.info(f"  Reviews with usable text: {non_empty:,} / {len(df):,}")

    log.info("Running TextBlob sentiment analysis (this may take a few minutes) …")

    # Process in batches to show progress
    batch_size = 50_000
    polarity_list     = []
    subjectivity_list = []

    for start in range(0, len(df), batch_size):
        batch = df["clean_text"].iloc[start : start + batch_size]
        polarity_list.extend(batch.apply(_get_polarity).tolist())
        subjectivity_list.extend(batch.apply(_get_subjectivity).tolist())
        done = min(start + batch_size, len(df))
        log.info(f"  Processed {done:,} / {len(df):,} rows …")

    df["polarity"]      = polarity_list
    df["subjectivity"]  = subjectivity_list
    df["sentiment"]     = df["polarity"].apply(polarity_to_label)

    log.info("Sentiment analysis complete.")
    log.info(df["sentiment"].value_counts().to_string())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. AGGREGATED SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════
def sentiment_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Per-city average polarity, subjectivity, and label distribution."""
    grp = df.groupby("City", observed=True)
    agg = grp.agg(
        avg_polarity=("polarity",     "mean"),
        avg_subjectivity=("subjectivity", "mean"),
        review_count=("polarity",     "count"),
    ).reset_index()
    agg["avg_polarity"]      = agg["avg_polarity"].round(4)
    agg["avg_subjectivity"]  = agg["avg_subjectivity"].round(4)
    agg["dominant_sentiment"] = agg["avg_polarity"].apply(polarity_to_label)
    agg = agg.sort_values("avg_polarity", ascending=False)
    return agg


def sentiment_by_place(df: pd.DataFrame) -> pd.DataFrame:
    """Per-place average polarity, subjectivity, and dominant sentiment."""
    grp = df.groupby("Place", observed=True)
    agg = grp.agg(
        avg_polarity=("polarity",     "mean"),
        avg_subjectivity=("subjectivity", "mean"),
        review_count=("polarity",     "count"),
    ).reset_index()
    agg["avg_polarity"]      = agg["avg_polarity"].round(4)
    agg["avg_subjectivity"]  = agg["avg_subjectivity"].round(4)
    agg["dominant_sentiment"] = agg["avg_polarity"].apply(polarity_to_label)
    agg = agg.sort_values("avg_polarity", ascending=False)
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(df: pd.DataFrame, city_sent: pd.DataFrame, place_sent: pd.DataFrame):
    """Save sentiment results to CSV files."""
    # Full result with polarity + label (drop heavy text columns to keep size small)
    slim = df[["City", "Place", "Rating", "polarity", "subjectivity", "sentiment"]].copy()
    slim.to_csv(os.path.join(OUTPUT_DIR, "sentiment_results.csv"), index=False)
    log.info(f"  Saved → {os.path.join(OUTPUT_DIR, 'sentiment_results.csv')}")

    city_sent.to_csv(os.path.join(OUTPUT_DIR, "sentiment_by_city.csv"), index=False)
    log.info(f"  Saved → {os.path.join(OUTPUT_DIR, 'sentiment_by_city.csv')}")

    place_sent.to_csv(os.path.join(OUTPUT_DIR, "sentiment_by_place.csv"), index=False)
    log.info(f"  Saved → {os.path.join(OUTPUT_DIR, 'sentiment_by_place.csv')}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("▶  Starting Sentiment Analysis Pipeline …")
    df              = load_dataset()
    df              = run_sentiment(df)
    city_sent       = sentiment_by_city(df)
    place_sent      = sentiment_by_place(df)
    save_outputs(df, city_sent, place_sent)
    log.info("\n✔  Sentiment Pipeline Complete.")
    log.info(f"   All outputs saved to: {OUTPUT_DIR}")
    return df


if __name__ == "__main__":
    main()
