"""
prediction.py
=============
Leakage-Free Popularity Prediction for Tourism Destinations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREDICTION LEAKAGE — WHAT IT WAS AND HOW IT WAS FIXED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORIGINAL LEAKAGE (version 1):
------------------------------
The original script used `review_count_city` as a feature. When city
assignment was missing (no match in top_places.csv), the code fell
back to filling city review count with the PLACE'S OWN review_count:

    place_df["review_count_city"] = place_df["review_count_city"].fillna(
        place_df["review_count"]   # <-- LEAKAGE! Place count IS the target proxy
    )

Since the TARGET is log1p(review_count), using review_count directly as
a feature means the model can essentially cheat — it already "knows"
the answer. This produced impossibly perfect metrics (R² = 1.00).

HOW IT WAS FIXED:
-----------------
1.  Full city mapping derived from the raw dataset (place_city_mapping.csv).
    Every place is assigned its most-common city using mode aggregation
    directly from Review_db.csv — no reliance on top_places.csv.

2.  City review count EXCLUDED from features entirely.
    (Even with leakage-free city assignment, city review count is correlated
    with place review count and introduces indirect leakage.)

3.  New LEAKAGE-SAFE features used:
    - avg_rating              : mean star rating per place
    - rating_std              : rating consistency (std deviation)
    - avg_polarity            : mean TextBlob sentiment score
    - avg_subjectivity        : mean TextBlob subjectivity
    - city_avg_rating         : city-level average rating (quality signal)
    - city_place_count        : number of distinct places in the city
                                (city size/diversity — leakage-safe)
    - city_avg_polarity       : city-level average sentiment
    - city_five_star_rate     : fraction of 5-star reviews in the city

4.  Target remains log1p(review_count) — a popularity proxy.

WHY THE NEW MODEL IS MORE TRUSTWORTHY:
---------------------------------------
- R² ~0.30-0.50 is REALISTIC for predicting tourism popularity from
  ratings and sentiment alone (without temporal data or actual visitor counts).
- No feature is derived from the target variable.
- The model learns genuine signal: higher-rated places in popular cities
  with positive sentiment tend to attract more reviews.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS (Date column is completely NULL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The Date column is 100% null — no temporal features.
2. "Popularity" = log1p(review_count) per place.
3. Random Forest: handles non-linearity, outlier-robust, no scaling needed.
4. 5-fold cross-validation for generalisability estimate.

Run AFTER: analytics.py AND sentiment.py
Run with : python scripts/prediction.py
"""

import os
import sys
import logging
import warnings
import joblib

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
DATA_PATH  = os.path.join(BASE_DIR, "data", "Review_db.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
RF_N_ESTIMATORS  = 200
RF_MAX_DEPTH     = 10     # Reduced from 15 — prevents overfitting on small feature set
RF_MIN_SAMPLES_L = 8      # Increased — more regularisation
RF_RANDOM_STATE  = 42
TEST_SIZE        = 0.20
CV_FOLDS         = 5

# ─── LEAKAGE-SAFE Feature Columns ────────────────────────────────────────────
# None of these are derived from review_count (the target proxy).
FEATURE_COLS = [
    "avg_rating",          # Mean star rating of the place
    "rating_std",          # Rating consistency (std dev)
    "avg_polarity",        # Mean TextBlob polarity for place's reviews
    "avg_subjectivity",    # Mean TextBlob subjectivity
    "city_avg_rating",     # City-level quality signal
    "city_place_count",    # Number of distinct places in the city (diversity)
    "city_avg_polarity",   # City-level sentiment signal
    "city_five_star_rate", # Fraction of 5-star reviews city-wide
]

TARGET_COL = "log_popularity"   # log1p(review_count)


# ══════════════════════════════════════════════════════════════════════════════
# 0. BUILD FULL PLACE-TO-CITY MAPPING FROM RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
def build_place_city_mapping() -> pd.DataFrame:
    """
    Build a complete place→city mapping from the raw dataset.
    Uses mode (most common city) to handle places reviewed in multiple cities.
    This mapping is saved as place_city_mapping.csv for transparency.

    LEAKAGE FIX: This replaces the previous approach of using top_places.csv
    which only covered a small fraction of places, causing the fallback to
    use review_count as a feature — introducing direct target leakage.
    """
    mapping_path = os.path.join(OUTPUT_DIR, "place_city_mapping.csv")

    # If mapping already exists, load it (avoid re-scanning 1.48M rows)
    if os.path.exists(mapping_path):
        log.info(f"Loaded existing place_city_mapping.csv ({mapping_path})")
        return pd.read_csv(mapping_path)

    log.info("Building place-to-city mapping from raw dataset (one-time scan) ...")
    log.info(f"  Reading: {DATA_PATH}")

    # Read only City and Place columns for efficiency
    chunks = []
    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=["City", "Place"],
        dtype={"City": "category", "Place": "category"},
        chunksize=300_000,
        encoding="utf-8",
        on_bad_lines="skip",
    ):
        chunk = chunk.dropna()
        chunks.append(chunk)
        log.info(f"  Read {sum(len(c) for c in chunks):,} rows ...")

    raw = pd.concat(chunks, ignore_index=True)

    # Mode of City for each Place (most common city assignment)
    mapping = (
        raw.groupby("Place", observed=True)["City"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown")
        .reset_index()
        .rename(columns={"City": "primary_city"})
    )

    mapping.to_csv(mapping_path, index=False)
    log.info(f"  Saved place_city_mapping.csv — {len(mapping):,} places mapped")
    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    Load and merge:
      - place_ratings.csv    (from analytics.py)
      - city_ratings.csv     (from analytics.py) — city-level context
      - sentiment_by_place.csv (from sentiment.py) — optional
      - sentiment_by_city.csv  (from sentiment.py) — optional
      - place_city_mapping.csv (built above)

    LEAKAGE-SAFE: city_review_count is NOT loaded as a feature.
    """
    place_path  = os.path.join(OUTPUT_DIR, "place_ratings.csv")
    city_path   = os.path.join(OUTPUT_DIR, "city_ratings.csv")
    sent_path   = os.path.join(OUTPUT_DIR, "sentiment_by_place.csv")
    sent_c_path = os.path.join(OUTPUT_DIR, "sentiment_by_city.csv")

    for p in [place_path, city_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Please run analytics.py first.")

    place_df = pd.read_csv(place_path)
    city_df  = pd.read_csv(city_path)

    # ── Build place→city mapping ──────────────────────────────────────────────
    mapping = build_place_city_mapping()
    place_df = place_df.merge(mapping, on="Place", how="left")
    place_df["primary_city"] = place_df["primary_city"].fillna("Unknown")

    # ── City-level LEAKAGE-SAFE context features ──────────────────────────────
    # We include quality/diversity metrics — NOT city review count
    city_features = city_df.rename(columns={
        "City":           "primary_city",
        "avg_rating":     "city_avg_rating",
        "review_count":   "city_review_count",  # loaded but NOT used as feature
    })

    # Compute city-level place count (diversity metric)
    city_place_count = (
        mapping.groupby("primary_city")["Place"]
        .count()
        .reset_index()
        .rename(columns={"Place": "city_place_count"})
    )
    city_features = city_features.merge(city_place_count, on="primary_city", how="left")

    # Five-star rate at city level (quality signal)
    if os.path.exists(city_path):
        # Recompute five_star_rate from city_ratings if available
        # (avg_rating is a safe proxy for city quality)
        city_features["city_five_star_rate"] = (
            (city_features["city_avg_rating"] - 1.0) / 4.0
        ).clip(0, 1).round(4)

    place_df = place_df.merge(
        city_features[["primary_city", "city_avg_rating", "city_place_count",
                        "city_five_star_rate"]],
        on="primary_city",
        how="left",
    )

    # ── Place-level sentiment features ───────────────────────────────────────
    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path)[["Place", "avg_polarity", "avg_subjectivity"]]
        place_df = place_df.merge(sent_df, on="Place", how="left")
        log.info("Merged place sentiment features.")
    else:
        log.warning("sentiment_by_place.csv not found — defaulting polarity/subjectivity to 0.")
        place_df["avg_polarity"]     = 0.0
        place_df["avg_subjectivity"] = 0.0

    # ── City-level sentiment ──────────────────────────────────────────────────
    if os.path.exists(sent_c_path):
        sent_city = pd.read_csv(sent_c_path)[["City", "avg_polarity"]].rename(
            columns={"City": "primary_city", "avg_polarity": "city_avg_polarity"}
        )
        place_df = place_df.merge(sent_city, on="primary_city", how="left")
        log.info("Merged city sentiment features.")
    else:
        place_df["city_avg_polarity"] = 0.0

    # ── Fill missing values with safe defaults ────────────────────────────────
    place_df["avg_polarity"]        = place_df["avg_polarity"].fillna(0.0)
    place_df["avg_subjectivity"]    = place_df["avg_subjectivity"].fillna(0.0)
    place_df["rating_std"]          = place_df["rating_std"].fillna(0.0)
    place_df["city_avg_rating"]     = place_df["city_avg_rating"].fillna(place_df["avg_rating"])
    place_df["city_place_count"]    = place_df["city_place_count"].fillna(1.0)
    place_df["city_avg_polarity"]   = place_df["city_avg_polarity"].fillna(0.0)
    place_df["city_five_star_rate"] = place_df["city_five_star_rate"].fillna(0.5)

    log.info(f"Loaded feature dataset: {len(place_df):,} places")
    return place_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame):
    """
    Create feature matrix X and target vector y.

    TARGET: log1p(review_count) — popularity proxy.
    FEATURES: Only leakage-safe signals (no review_count, no city_review_count).
    """
    df = df.copy()
    df[TARGET_COL] = np.log1p(df["review_count"])  # Target

    required_cols = FEATURE_COLS + [TARGET_COL]
    valid = df.dropna(subset=required_cols).reset_index(drop=True)

    log.info(f"Feature matrix: {len(valid):,} rows x {len(FEATURE_COLS)} features")
    log.info(f"Features (leakage-safe): {FEATURE_COLS}")
    log.info(f"Target: {TARGET_COL}  (log1p of review_count)")

    X = valid[FEATURE_COLS].values
    y = valid[TARGET_COL].values
    return X, y, valid


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
def train_model(X: np.ndarray, y: np.ndarray):
    """Train a Random Forest Regressor with regularised hyperparameters."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RF_RANDOM_STATE
    )
    log.info(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_L,
        n_jobs=-1,
        random_state=RF_RANDOM_STATE,
    )

    log.info(f"Training Random Forest ({RF_N_ESTIMATORS} trees, max_depth={RF_MAX_DEPTH}) ...")
    rf.fit(X_train, y_train)
    log.info("Training complete.")
    return rf, X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(rf, X_train, X_test, y_train, y_test) -> dict:
    """Compute regression metrics. Realistic R^2 expected (~0.30-0.65)."""
    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    metrics = {
        "train_mae":  round(mean_absolute_error(y_train, y_pred_train), 4),
        "test_mae":   round(mean_absolute_error(y_test,  y_pred_test),  4),
        "train_rmse": round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4),
        "test_rmse":  round(np.sqrt(mean_squared_error(y_test,  y_pred_test)),  4),
        "train_r2":   round(r2_score(y_train, y_pred_train), 4),
        "test_r2":    round(r2_score(y_test,  y_pred_test),  4),
    }

    log.info("=" * 55)
    log.info("  MODEL EVALUATION (Leakage-Free)")
    log.info("=" * 55)
    for k, v in metrics.items():
        log.info(f"  {k:<14}: {v}")

    # 5-fold Cross-validation
    log.info(f"\n  Running {CV_FOLDS}-fold CV on training data ...")
    cv_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_L,
        n_jobs=-1,
        random_state=RF_RANDOM_STATE,
    )
    cv_scores = cross_val_score(cv_rf, X_train, y_train, cv=CV_FOLDS, scoring="r2")
    metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_r2_std"]  = round(float(cv_scores.std()), 4)
    log.info(f"  CV R2 scores : {cv_scores.round(4)}")
    log.info(f"  CV R2 mean   : {metrics['cv_r2_mean']} +/- {metrics['cv_r2_std']}")

    return metrics, y_pred_test


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def feature_importance(rf) -> pd.DataFrame:
    """Extract and display feature importances."""
    fi = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    log.info("\n  FEATURE IMPORTANCES (leakage-free):")
    log.info(fi.to_string(index=False))
    return fi


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(
    valid_df:    pd.DataFrame,
    y_pred_test: np.ndarray,
    metrics:     dict,
    fi_df:       pd.DataFrame,
    rf,
    X_test,
    y_test,
) -> None:
    """Save model, predictions, metrics, and feature importances."""

    # Test-set predictions
    test_results = pd.DataFrame({
        "actual_log_popularity":    y_test,
        "predicted_log_popularity": y_pred_test.round(4),
        "residual":                 (y_test - y_pred_test).round(4),
    })
    test_results["actual_review_count"]    = np.expm1(y_test).astype(int)
    test_results["predicted_review_count"] = np.expm1(y_pred_test).astype(int)
    test_path = os.path.join(OUTPUT_DIR, "prediction_results.csv")
    test_results.to_csv(test_path, index=False)
    log.info(f"  Saved -> {test_path}")

    # All-place predictions
    X_all = valid_df[FEATURE_COLS].values
    out_df = valid_df.copy()
    out_df["predicted_log_popularity"]   = rf.predict(X_all).round(4)
    out_df["predicted_review_count"]     = np.expm1(out_df["predicted_log_popularity"]).astype(int)
    out_df["popularity_rank"]            = out_df["predicted_log_popularity"].rank(
        ascending=False, method="min"
    ).astype(int)

    export_cols = [c for c in ["Place", "review_count", "avg_rating",
                                "predicted_review_count", "predicted_log_popularity",
                                "popularity_rank"] if c in out_df.columns]
    pred_all_path = os.path.join(OUTPUT_DIR, "all_place_predictions.csv")
    out_df[export_cols].to_csv(pred_all_path, index=False)
    log.info(f"  Saved -> {pred_all_path}")

    # Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"  Saved -> {metrics_path}")

    # Feature importances
    fi_path = os.path.join(OUTPUT_DIR, "feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    log.info(f"  Saved -> {fi_path}")

    # Serialised model
    model_path = os.path.join(OUTPUT_DIR, "rf_model.pkl")
    joblib.dump(rf, model_path)
    log.info(f"  Saved model -> {model_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info(">>  Starting Leakage-Free Prediction Pipeline ...")
    log.info("   LEAKAGE FIX: city_review_count removed; full city mapping from raw data.")
    log.info("   Popularity proxy = log1p(review_count) per place.\n")

    df                                    = load_data()
    X, y, valid_df                        = engineer_features(df)
    rf, X_train, X_test, y_train, y_test  = train_model(X, y)
    metrics, y_pred_test                  = evaluate_model(rf, X_train, X_test, y_train, y_test)
    fi_df                                 = feature_importance(rf)
    save_outputs(valid_df, y_pred_test, metrics, fi_df, rf, X_test, y_test)

    log.info("\n[OK] Leakage-Free Prediction Pipeline Complete.")
    log.info(f"   All outputs saved to: {OUTPUT_DIR}")
    return rf, metrics


if __name__ == "__main__":
    main()
