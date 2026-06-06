"""
analytics.py
============
Tourism Trend Analytics - Data Loading, Exploration, and Statistics.

This script:
  1. Loads the Review_db.csv dataset efficiently using chunked pandas reads.
  2. Displays key dataset statistics (shape, dtypes, nulls, rating distribution).
  3. Finds top cities by review count and average rating.
  4. Finds top places by review count and average rating.
  5. Computes per-city and per-place average ratings.
  6. Saves all output tables to CSV in the outputs/ directory.

Save to: BDT_project/scripts/analytics.py
Run with: python scripts/analytics.py
"""

import os
import sys
import logging
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "Review_db.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CHUNKSIZE   = 200_000   # rows per chunk for memory-efficient loading
TOP_N       = 20        # how many top cities / places to export


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV in chunks to handle 1.48 M rows without OOM issues.
    Only the required columns are loaded; the Date column (fully null) is
    intentionally skipped to save memory.
    """
    log.info(f"Loading dataset from: {filepath}")

    usecols = ["City", "Place", "Review", "Rating", "Name", "Raw_Review"]
    dtype_map = {
        "City":       "category",
        "Place":      "category",
        "Name":       "category",
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
        # Drop rows where Rating is NaN (unusable for analytics)
        chunk = chunk.dropna(subset=["Rating"])
        # Clamp ratings to [1, 5] to handle any data-quality issues
        chunk["Rating"] = chunk["Rating"].clip(1.0, 5.0)
        chunks.append(chunk)
        total += len(chunk)
        log.info(f"  Loaded {total:,} rows so far …")

    df = pd.concat(chunks, ignore_index=True)

    # Convert categoricals back (lost during concat from mixed chunks)
    for col in ["City", "Place", "Name"]:
        df[col] = df[col].astype("category")

    log.info(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def display_statistics(df: pd.DataFrame) -> dict:
    """Print and return a summary of key dataset statistics."""
    stats = {
        "total_rows":      len(df),
        "total_columns":   df.shape[1],
        "unique_cities":   df["City"].nunique(),
        "unique_places":   df["Place"].nunique(),
        "unique_reviewers":df["Name"].nunique(),
        "rating_min":      float(df["Rating"].min()),
        "rating_max":      float(df["Rating"].max()),
        "rating_mean":     round(float(df["Rating"].mean()), 4),
        "rating_median":   float(df["Rating"].median()),
        "null_review":     int(df["Review"].isna().sum()),
        "null_raw_review": int(df["Raw_Review"].isna().sum()),
    }

    log.info("=" * 60)
    log.info("  DATASET STATISTICS")
    log.info("=" * 60)
    for k, v in stats.items():
        log.info(f"  {k:<22} : {v:,}" if isinstance(v, int) else f"  {k:<22} : {v}")

    # Rating distribution
    rating_dist = (
        df["Rating"]
        .value_counts()
        .sort_index()
        .rename("count")
        .rename_axis("rating")
        .reset_index()
    )
    rating_dist["pct"] = (rating_dist["count"] / len(df) * 100).round(2)
    log.info("\n  Rating Distribution:")
    log.info(rating_dist.to_string(index=False))

    return stats, rating_dist


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOP CITIES
# ══════════════════════════════════════════════════════════════════════════════
def get_top_cities(df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Aggregate by City:
      - review_count  : total number of reviews
      - avg_rating    : mean rating
      - unique_places : distinct places reviewed in that city
    Return top_n cities sorted by review_count descending.
    """
    city_agg = (
        df.groupby("City", observed=True)
        .agg(
            review_count=("Rating", "count"),
            avg_rating=("Rating", "mean"),
            unique_places=("Place", "nunique"),
        )
        .reset_index()
    )
    city_agg["avg_rating"] = city_agg["avg_rating"].round(4)
    city_agg = city_agg.sort_values("review_count", ascending=False).head(top_n)
    log.info(f"\n  Top {top_n} Cities by Review Count:")
    log.info(city_agg.to_string(index=False))
    return city_agg


# ══════════════════════════════════════════════════════════════════════════════
# 4. TOP PLACES
# ══════════════════════════════════════════════════════════════════════════════
def get_top_places(df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Aggregate by Place:
      - review_count : total reviews
      - avg_rating   : mean rating
      - city         : most common city for that place
    Return top_n places sorted by review_count descending.
    """
    place_agg = (
        df.groupby("Place", observed=True)
        .agg(
            review_count=("Rating", "count"),
            avg_rating=("Rating", "mean"),
            city=("City", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
        )
        .reset_index()
    )
    place_agg["avg_rating"] = place_agg["avg_rating"].round(4)
    place_agg = place_agg.sort_values("review_count", ascending=False).head(top_n)
    log.info(f"\n  Top {top_n} Places by Review Count:")
    log.info(place_agg.to_string(index=False))
    return place_agg


# ══════════════════════════════════════════════════════════════════════════════
# 5. AVERAGE RATINGS
# ══════════════════════════════════════════════════════════════════════════════
def compute_average_ratings(df: pd.DataFrame):
    """
    Compute average ratings for every city and every place.
    Returns two DataFrames: city_ratings, place_ratings.
    """
    city_ratings = (
        df.groupby("City", observed=True)["Rating"]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "avg_rating", "count": "review_count", "std": "rating_std"})
        .reset_index()
    )
    city_ratings["avg_rating"]  = city_ratings["avg_rating"].round(4)
    city_ratings["rating_std"]  = city_ratings["rating_std"].round(4)
    city_ratings = city_ratings.sort_values("avg_rating", ascending=False)

    place_ratings = (
        df.groupby("Place", observed=True)["Rating"]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "avg_rating", "count": "review_count", "std": "rating_std"})
        .reset_index()
    )
    place_ratings["avg_rating"] = place_ratings["avg_rating"].round(4)
    place_ratings["rating_std"] = place_ratings["rating_std"].round(4)
    place_ratings = place_ratings.sort_values("avg_rating", ascending=False)

    log.info(f"\n  City avg ratings computed for {len(city_ratings):,} cities")
    log.info(f"  Place avg ratings computed for {len(place_ratings):,} places")
    return city_ratings, place_ratings


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(
    top_cities:   pd.DataFrame,
    top_places:   pd.DataFrame,
    city_ratings: pd.DataFrame,
    place_ratings: pd.DataFrame,
    rating_dist:  pd.DataFrame,
) -> None:
    """Save all analytics results to CSV files in outputs/."""
    files = {
        "top_cities.csv":    top_cities,
        "top_places.csv":    top_places,
        "city_ratings.csv":  city_ratings,
        "place_ratings.csv": place_ratings,
        "rating_distribution.csv": rating_dist,
    }
    for fname, df_out in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        df_out.to_csv(path, index=False)
        log.info(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("▶  Starting Analytics Pipeline …")

    df                        = load_dataset()
    stats, rating_dist        = display_statistics(df)
    top_cities                = get_top_cities(df)
    top_places                = get_top_places(df)
    city_ratings, place_ratings = compute_average_ratings(df)

    save_outputs(top_cities, top_places, city_ratings, place_ratings, rating_dist)

    log.info("\n✔  Analytics Pipeline Complete.")
    log.info(f"   All outputs saved to: {OUTPUT_DIR}")

    # Return df so other scripts can import and reuse it
    return df


if __name__ == "__main__":
    main()
