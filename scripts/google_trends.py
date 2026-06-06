"""
google_trends.py
================
Real-Time Tourism Trend Module — Transparent Google Trends Integration.

TRANSPARENCY IMPROVEMENTS (v2):
--------------------------------
This version adds explicit status/source/notes tracking to every record:

  status  : 'live'        — data fetched successfully from Google Trends API
            'fallback'    — API unavailable; using historical proxy score
            'unavailable' — API and all fallbacks failed
            'cached'      — data loaded from existing cached file

  source  : 'Google Trends' | 'fallback' | 'cached'

  notes   : Human-readable explanation of how the score was derived

  RawTrendScore : Raw Google Trends score (0–100 scale from API, or 0 for fallback)
  TrendScore    : Normalised score [0, 1]

FAILURE HANDLING POLICY:
--------------------------
- If pytrends is not installed  → log warning, use fallback, continue
- If API rate-limited           → exponential back-off, then fallback
- If API returns empty data     → record status='unavailable', score=0
- If any exception occurs       → record failure honestly, continue
- Execution NEVER aborts due to Google Trends failure

FALLBACK SCORING:
-----------------
When live data is unavailable, TrendScore is set to 0.0 (not fake scores).
This ensures the FinalRealtimeScore = 0.9 × ExistingHotspotScore (unmodified),
which is honest and report-friendly.

FORMULA:
  FinalRealtimeScore = TrendScore

Run AFTER: analytics.py
Run with : python scripts/google_trends.py
"""

import os
import sys
import time
import logging
import warnings
import random
import json
import inspect

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

# ─── Constants ────────────────────────────────────────────────────────────────
TOP_N_PLACES      = 20
BATCH_SIZE        = 5
REQUEST_DELAY_SEC = 2.0
TIMEFRAME         = "today 3-m"
GEO               = ""          # Global; use "IN" for India-only
W_EXISTING        = 0.0
W_TREND           = 1.0

# Status constants
STATUS_LIVE        = "live"
STATUS_FALLBACK    = "fallback"
STATUS_UNAVAILABLE = "unavailable"
STATUS_CACHED      = "cached"

SOURCE_GOOGLE  = "Google Trends"
SOURCE_FALLBACK = "fallback"
SOURCE_CACHED  = "cached"


def patch_urllib3_retry_compat() -> None:
    """
    Patch urllib3 Retry for pytrends/urllib3 compatibility.

    Some pytrends releases still call Retry(method_whitelist=...), while
    urllib3 2.x renamed that argument to allowed_methods. Without this shim,
    live Google Trends fetches fail before the request is sent.
    """
    try:
        from urllib3.util.retry import Retry
    except Exception:
        return

    if "method_whitelist" in inspect.signature(Retry.__init__).parameters:
        return
    if getattr(Retry.__init__, "_bdt_compat_patched", False):
        return

    original_init = Retry.__init__

    def compat_init(self, *args, **kwargs):
        if "method_whitelist" in kwargs:
            methods = kwargs.pop("method_whitelist")
            kwargs.setdefault("allowed_methods", methods)
        return original_init(self, *args, **kwargs)

    compat_init._bdt_compat_patched = True
    Retry.__init__ = compat_init


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════
def minmax_norm(series: pd.Series) -> pd.Series:
    """Normalise to [0, 1]. Handles zero-range safely."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.ones(len(series)) * 0.5, index=series.index)
    return (series - mn) / (mx - mn)


# ══════════════════════════════════════════════════════════════════════════════
def load_top_places(top_n: int = TOP_N_PLACES) -> pd.DataFrame:
    """
    Load top-N places from top_places.csv to query their Google Trends.
    This ensures trends are fetched for overall popular places, entirely separated from hotspot scores.
    """
    fallback_path = os.path.join(OUTPUT_DIR, "top_places.csv")

    if os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
        log.info(f"Loaded top_places.csv — {len(df):,} places total.")
        df = df.rename(columns={"review_count": "BasePopularity"})
        return df.head(top_n)[["Place", "BasePopularity"]].copy()
    else:
        raise FileNotFoundError(
            "top_places.csv not found in outputs/. Please run analytics.py first."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHECK IF CACHED DATA IS RECENT
# ══════════════════════════════════════════════════════════════════════════════
def load_cached_trends() -> pd.DataFrame | None:
    """
    Load existing google_trends.csv if it contains live/live-derived data.
    Returns None if cache is stale, missing, or all-fallback.
    """
    cache_path = os.path.join(OUTPUT_DIR, "google_trends.csv")
    if not os.path.exists(cache_path):
        return None

    try:
        cached = pd.read_csv(cache_path)
        # Only reuse cache if it has the 'status' column and has live data
        if "status" in cached.columns and (cached["status"] == STATUS_LIVE).any():
            log.info(f"Found cached Google Trends data with live records ({len(cached)} rows).")
            cached["status"] = STATUS_CACHED
            cached["source"] = SOURCE_CACHED
            cached["notes"]  = cached.get("notes", "Loaded from existing google_trends.csv")
            return cached
    except Exception as e:
        log.warning(f"Could not parse cached trends file: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. FETCH FROM GOOGLE TRENDS API
# ══════════════════════════════════════════════════════════════════════════════
def fetch_trends_for_keywords(keywords: list) -> dict:
    """
    Fetch Google Trends interest-over-time for keywords.
    Returns dict: keyword → {'raw_score': float, 'status': str, 'notes': str}
    """
    patch_urllib3_retry_compat()

    try:
        from pytrends.request import TrendReq
    except ImportError:
        log.warning("pytrends not installed (pip install pytrends). Using fallback scores.")
        return {
            kw: {"raw_score": 0.0, "status": STATUS_FALLBACK,
                 "notes": "pytrends not installed"}
            for kw in keywords
        }

    # pytrends >= 4.9 removed backoff_factor / method_whitelist (urllib3 2.x compat)
    try:
        pytrends = TrendReq(hl="en-US", tz=330, timeout=(10, 25), retries=2, backoff_factor=0.5)
    except TypeError:
        try:
            pytrends = TrendReq(hl="en-US", tz=330, timeout=(10, 25))
        except Exception as e:
            log.warning(f"Could not instantiate TrendReq: {e}. Using fallback.")
            return {
                kw: {"raw_score": 0.0, "status": STATUS_FALLBACK,
                     "notes": f"TrendReq init failed: {str(e)[:60]}"}
                for kw in keywords
            }
    results = {}

    for i in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[i: i + BATCH_SIZE]
        log.info(f"  Fetching trends for batch {i // BATCH_SIZE + 1}: {batch}")

        max_retries = 3
        batch_success = False

        for attempt in range(max_retries):
            try:
                pytrends.build_payload(
                    batch, cat=0, timeframe=TIMEFRAME, geo=GEO, gprop=""
                )
                iot = pytrends.interest_over_time()

                if iot.empty:
                    log.warning(f"    No data returned for batch: {batch}")
                    for kw in batch:
                        results[kw] = {
                            "raw_score": 0.0,
                            "status":    STATUS_UNAVAILABLE,
                            "notes":     "API returned empty data for this keyword",
                        }
                else:
                    for kw in batch:
                        if kw in iot.columns:
                            score = float(iot[kw].mean())
                            results[kw] = {
                                "raw_score": score,
                                "status":    STATUS_LIVE,
                                "notes":     f"Mean interest over {TIMEFRAME} (0-100 scale)",
                            }
                        else:
                            results[kw] = {
                                "raw_score": 0.0,
                                "status":    STATUS_UNAVAILABLE,
                                "notes":     "Keyword not in API response",
                            }
                batch_success = True
                break  # Success

            except Exception as exc:
                err_str = str(exc).lower()
                if "429" in err_str or "too many requests" in err_str:
                    wait = REQUEST_DELAY_SEC * (2 ** attempt) + random.uniform(1, 3)
                    log.warning(
                        f"    Rate-limited (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait:.1f}s ..."
                    )
                    time.sleep(wait)
                    if attempt == max_retries - 1:
                        log.error(f"    Max retries exceeded for batch {batch}.")
                        for kw in batch:
                            results.setdefault(kw, {
                                "raw_score": 0.0,
                                "status":    STATUS_FALLBACK,
                                "notes":     f"Rate-limited after {max_retries} retries",
                            })
                else:
                    log.error(f"    Error fetching batch {batch}: {exc}")
                    for kw in batch:
                        results.setdefault(kw, {
                            "raw_score": 0.0,
                            "status":    STATUS_FALLBACK,
                            "notes":     f"API error: {str(exc)[:60]}",
                        })
                    break

        if i + BATCH_SIZE < len(keywords):
            time.sleep(REQUEST_DELAY_SEC)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD TREND DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
def fetch_google_trends(places: list) -> pd.DataFrame:
    """
    Fetch trends and build a transparent DataFrame with status/source/notes.
    """
    log.info(f"Fetching Google Trends for {len(places)} places ...")
    scores = fetch_trends_for_keywords(places)

    rows = []
    for place in places:
        info = scores.get(place, {
            "raw_score": 0.0, "status": STATUS_FALLBACK, "notes": "No data fetched"
        })
        rows.append({
            "Place":        place,
            "RawTrendScore": info["raw_score"],
            "status":       info["status"],
            "source":       SOURCE_GOOGLE if info["status"] == STATUS_LIVE else SOURCE_FALLBACK,
            "notes":        info["notes"],
        })

    trend_df = pd.DataFrame(rows)

    # Normalise only among live scores where possible
    live_mask = trend_df["status"] == STATUS_LIVE
    if live_mask.any():
        # Normalise live scores; fallback scores stay at 0
        live_scores = trend_df.loc[live_mask, "RawTrendScore"]
        trend_df.loc[live_mask, "TrendScore"] = minmax_norm(live_scores).round(6)
        trend_df.loc[~live_mask, "TrendScore"] = 0.0
        log.info(f"  Live scores obtained for {live_mask.sum()} / {len(places)} places.")
    else:
        trend_df["TrendScore"] = 0.0
        log.warning("  No live trend data — all TrendScores set to 0.0 (honest fallback).")

    return trend_df


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE FINAL REAL-TIME SCORE
# ══════════════════════════════════════════════════════════════════════════════
def compute_realtime_score(base_df: pd.DataFrame, trend_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge and compute:
        FinalRealtimeScore = TrendScore

    This completely separates real-time trends from the hotspot scoring system.
    """
    merged = base_df.merge(
        trend_df[["Place", "TrendScore", "status", "source", "notes"]],
        on="Place",
        how="left",
    )
    merged["TrendScore"] = merged["TrendScore"].fillna(0.0)
    merged["status"]     = merged["status"].fillna(STATUS_UNAVAILABLE)
    merged["source"]     = merged["source"].fillna(SOURCE_FALLBACK)
    merged["notes"]      = merged["notes"].fillna("No trend data available")

    merged["FinalRealtimeScore"] = merged["TrendScore"].round(6)

    return merged.sort_values("FinalRealtimeScore", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(trend_df: pd.DataFrame, realtime_df: pd.DataFrame) -> None:
    """Save transparent outputs with all metadata columns."""

    # google_trends.csv — full transparency
    gt_path = os.path.join(OUTPUT_DIR, "google_trends.csv")
    cols = ["Place", "RawTrendScore", "TrendScore", "status", "source", "notes"]
    trend_df[[c for c in cols if c in trend_df.columns]].to_csv(gt_path, index=False)
    log.info(f"  Saved -> {gt_path}")

    # realtime_hotspots.csv — with trend metadata
    rt_path = os.path.join(OUTPUT_DIR, "realtime_hotspots.csv")
    rt_cols = ["Place", "BasePopularity", "TrendScore",
               "FinalRealtimeScore", "status", "source", "notes"]
    realtime_df[[c for c in rt_cols if c in realtime_df.columns]].to_csv(rt_path, index=False)
    log.info(f"  Saved -> {rt_path}")

    # Summary log
    status_counts = trend_df["status"].value_counts().to_dict()
    log.info(f"  Status summary: {status_counts}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info(">>  Starting Real-Time Tourism Trend Pipeline (Google Trends) ...")
    log.info(f"   Weights: Existing={W_EXISTING}, Trend={W_TREND}")
    log.info(f"   Timeframe: {TIMEFRAME} | Geo: '{GEO or 'Global'}' | Top-N: {TOP_N_PLACES}")
    log.info("   Note: Google Trends failure will not stop execution.")

    # Step 1: Load top places
    base_df = load_top_places(TOP_N_PLACES)
    places = base_df["Place"].tolist()
    log.info(f"   Top places: {places}")

    # Step 2: Fetch Google Trends (failure-safe)
    try:
        trend_df = fetch_google_trends(places)
    except Exception as e:
        log.error(f"Unexpected error in trend fetch: {e}")
        # Create honest fallback dataframe — never fake scores
        trend_df = pd.DataFrame({
            "Place":         places,
            "RawTrendScore": 0.0,
            "TrendScore":    0.0,
            "status":        STATUS_FALLBACK,
            "source":        SOURCE_FALLBACK,
            "notes":         f"Fetch failed: {str(e)[:80]}",
        })

    # Step 3: Compute final real-time score
    realtime_df = compute_realtime_score(base_df, trend_df)

    # Step 4: Display top results
    log.info("\n  TOP REAL-TIME TRENDS:")
    display_cols = ["Place", "TrendScore", "FinalRealtimeScore", "status"]
    log.info(realtime_df[[c for c in display_cols if c in realtime_df.columns]].head(10).to_string(index=False))

    # Step 5: Save outputs
    save_outputs(trend_df, realtime_df)

    # Step 6: Summary
    live_count = (trend_df["status"] == STATUS_LIVE).sum()
    fallback_count = len(trend_df) - live_count
    log.info(f"\n  Live trend data: {live_count} / {len(places)} places")
    if fallback_count > 0:
        log.info(
            f"  {fallback_count} places using fallback (TrendScore=0.0). "
            f"FinalRealtimeScore = 0.0 for these."
        )

    log.info("\n[OK] Real-Time Trend Pipeline Complete.")
    log.info(f"   All outputs saved to: {OUTPUT_DIR}")
    return trend_df, realtime_df


if __name__ == "__main__":
    main()
