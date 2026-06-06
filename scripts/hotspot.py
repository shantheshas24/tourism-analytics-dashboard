"""
hotspot.py
==========
Tourism Hotspot Detection and Ranking.

Hotspot Score Formula:
  HotspotScore = 0.5 × RatingScore
               + 0.3 × ReviewVolume
               + 0.2 × SentimentScore

Where:
  RatingScore    = MinMaxNorm(avg_rating)            — normalised to [0,1]
  ReviewVolume   = MinMaxNorm(log1p(review_count))   — log-scaled then normalised
  SentimentScore = MinMaxNorm(avg_polarity + 1) / 2  — polarity in [-1,1] → [0,1]

This script:
  1. Loads pre-computed analytics (city_ratings.csv) and
     sentiment (sentiment_by_city.csv, sentiment_by_place.csv) outputs.
  2. Merges them at both city level and place level.
  3. Computes the weighted hotspot score.
  4. Ranks destinations.
  5. Saves top hotspots to outputs/hotspots_city.csv and outputs/hotspots_place.csv.

Save to: BDT_project/scripts/hotspot.py
Run AFTER: analytics.py  AND  sentiment.py
Run with: python scripts/hotspot.py
"""

import os
import sys
import logging
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Weights (must sum to 1.0) ────────────────────────────────────────────────
W_RATING    = 0.5
W_VOLUME    = 0.3
W_SENTIMENT = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Min-Max Normalisation
# ══════════════════════════════════════════════════════════════════════════════
def minmax_norm(series: pd.Series) -> pd.Series:
    """Normalise a pandas Series to [0, 1]. Handles zero-range safely."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD PRE-COMPUTED DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_city_data() -> pd.DataFrame:
    """
    Merge city_ratings.csv with sentiment_by_city.csv.
    Falls back to zero sentiment if sentiment file is missing.
    """
    ratings_path  = os.path.join(OUTPUT_DIR, "city_ratings.csv")
    sent_path     = os.path.join(OUTPUT_DIR, "sentiment_by_city.csv")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"City ratings not found at {ratings_path}. "
            "Please run analytics.py first."
        )

    city_df = pd.read_csv(ratings_path)
    log.info(f"Loaded city ratings: {len(city_df):,} cities")

    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path)[["City", "avg_polarity"]]
        city_df = city_df.merge(sent_df, on="City", how="left")
        log.info("Merged sentiment data for cities.")
    else:
        log.warning("Sentiment file not found — defaulting avg_polarity to 0.")
        city_df["avg_polarity"] = 0.0

    city_df["avg_polarity"] = city_df["avg_polarity"].fillna(0.0)
    return city_df


def load_place_data() -> pd.DataFrame:
    """
    Merge place_ratings.csv with sentiment_by_place.csv.
    Falls back to zero sentiment if sentiment file is missing.
    """
    ratings_path  = os.path.join(OUTPUT_DIR, "place_ratings.csv")
    sent_path     = os.path.join(OUTPUT_DIR, "sentiment_by_place.csv")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"Place ratings not found at {ratings_path}. "
            "Please run analytics.py first."
        )

    place_df = pd.read_csv(ratings_path)
    log.info(f"Loaded place ratings: {len(place_df):,} places")

    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path)[["Place", "avg_polarity"]]
        place_df = place_df.merge(sent_df, on="Place", how="left")
        log.info("Merged sentiment data for places.")
    else:
        log.warning("Sentiment file not found — defaulting avg_polarity to 0.")
        place_df["avg_polarity"] = 0.0

    place_df["avg_polarity"] = place_df["avg_polarity"].fillna(0.0)
    return place_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE HOTSPOT SCORE
# ══════════════════════════════════════════════════════════════════════════════
def compute_hotspot_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the weighted hotspot score for each row (city or place).

    Steps:
      a) rating_score    = MinMaxNorm(avg_rating)
      b) volume_score    = MinMaxNorm(log1p(review_count))
      c) sentiment_score = MinMaxNorm((avg_polarity + 1) / 2)
                         ← shifts polarity from [-1,1] to [0,1] before normalising
      d) hotspot_score   = 0.5*a + 0.3*b + 0.2*c
    """
    df = df.copy()

    # a) Rating score
    df["rating_score"]    = minmax_norm(df["avg_rating"])

    # b) Volume score (log-scale dampens outliers)
    df["log_volume"]      = np.log1p(df["review_count"])
    df["volume_score"]    = minmax_norm(df["log_volume"])

    # c) Sentiment score: shift [-1,1] → [0,1] then normalise
    df["sent_shifted"]    = (df["avg_polarity"] + 1.0) / 2.0
    df["sentiment_score"] = minmax_norm(df["sent_shifted"])

    # d) Weighted combination
    df["hotspot_score"] = (
        W_RATING    * df["rating_score"]    +
        W_VOLUME    * df["volume_score"]    +
        W_SENTIMENT * df["sentiment_score"]
    ).round(6)

    # Drop intermediate columns
    df.drop(columns=["log_volume", "sent_shifted"], inplace=True, errors="ignore")

    return df.sort_values("hotspot_score", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RANK AND DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
def rank_destinations(df: pd.DataFrame, name_col: str, top_n: int = 30) -> pd.DataFrame:
    """Add rank column and log the top-N destinations."""
    df = df.copy()
    df.insert(0, "rank", range(1, len(df) + 1))
    log.info(f"\n  TOP {top_n} HOTSPOTS ({name_col}):")
    display_cols = [name_col, "hotspot_score", "avg_rating", "review_count", "avg_polarity"]
    display_cols = [c for c in display_cols if c in df.columns]
    log.info(df[display_cols].head(top_n).to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(city_df: pd.DataFrame, place_df: pd.DataFrame) -> None:
    city_path  = os.path.join(OUTPUT_DIR, "hotspots_city.csv")
    place_path = os.path.join(OUTPUT_DIR, "hotspots_place.csv")
    city_df.to_csv(city_path,  index=False)
    place_df.to_csv(place_path, index=False)
    log.info(f"\n  Saved → {city_path}")
    log.info(f"  Saved → {place_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("▶  Starting Hotspot Detection Pipeline …")
    log.info(f"   Weights → Rating={W_RATING}, Volume={W_VOLUME}, Sentiment={W_SENTIMENT}")

    # ── City-level hotspots
    city_df  = load_city_data()
    city_df  = compute_hotspot_score(city_df)
    city_df  = rank_destinations(city_df, "City", top_n=30)

    # ── Place-level hotspots
    place_df = load_place_data()
    place_df = compute_hotspot_score(place_df)
    place_df = rank_destinations(place_df, "Place", top_n=30)

    save_outputs(city_df, place_df)

    log.info("\n✔  Hotspot Detection Pipeline Complete.")
    log.info(f"   All outputs saved to: {OUTPUT_DIR}")
    return city_df, place_df


if __name__ == "__main__":
    main()
