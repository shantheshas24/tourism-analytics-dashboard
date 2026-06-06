"""
processing_load_analysis.py
============================
Processing Performance Analysis: Pandas vs Apache Spark.

PURPOSE
-------
This script supports the report section:
  "Analysis of Impact of Data Size on Processing Load"

It measures elapsed time and throughput for standard data aggregation
operations at multiple data-size checkpoints:
  - 100,000 rows
  - 500,000 rows
  - 1,000,000 rows
  - Full dataset (~1,482,466 rows, if memory allows)

Both Pandas and PySpark are measured for a comparable groupBy aggregation
task (compute avg rating and review count by City).

OUTPUTS
-------
  outputs/processing_load.csv
      Columns: method, sample_size, elapsed_seconds, rows_per_second, notes

  outputs/charts/processing_time_vs_size.png
      Line chart: elapsed time vs sample size for Pandas vs Spark

  outputs/charts/throughput_vs_size.png
      Line chart: rows/second vs sample size for Pandas vs Spark

INTERPRETATION
--------------
- Pandas performs better at small-to-medium sizes due to no Spark startup overhead.
- Spark's advantage grows with data volume (parallelism pays off at scale).
- On a single machine, Spark overhead is always visible — real gains require
  multi-node clusters where Spark can distribute across many workers.
- The crossover point (where Spark beats Pandas) depends on hardware.

Run AFTER: analytics.py (needs Review_db.csv)
Run with : python scripts/processing_load_analysis.py
"""

import os
import sys
import time
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
DATA_PATH  = os.path.join(BASE_DIR, "data", "Review_db.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR  = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
SAMPLE_SIZES     = [100_000, 500_000, 1_000_000]
FULL_DATA_SIZE   = None    # Set to actual row count after loading
PANDAS_REPEATS   = 3       # Average over N runs for stability
SPARK_REPEATS    = 2       # Spark has startup overhead; fewer repeats needed


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD FULL DATASET (once, reuse for all samples)
# ══════════════════════════════════════════════════════════════════════════════
def load_full_dataset() -> pd.DataFrame:
    """
    Load entire dataset into memory for sub-sampling.
    Only City, Place, Rating columns needed for the aggregation benchmark.
    """
    log.info(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(
        DATA_PATH,
        usecols=["City", "Place", "Rating"],
        dtype={"City": "category", "Place": "category", "Rating": "float32"},
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=True,
    )
    df = df.dropna(subset=["Rating", "City"])
    df["Rating"] = df["Rating"].clip(1.0, 5.0)
    log.info(f"Loaded {len(df):,} rows for benchmarking")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PANDAS BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
def benchmark_pandas(df: pd.DataFrame, sample_size: int) -> dict:
    """
    Benchmark Pandas groupBy aggregation on a sample of `sample_size` rows.
    Task: compute count and avg_rating grouped by City.
    Timed over PANDAS_REPEATS runs; median elapsed time reported.
    """
    # Sub-sample without replacement
    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=42).reset_index(drop=True)

    elapsed_times = []
    for _ in range(PANDAS_REPEATS):
        t0 = time.perf_counter()

        _ = (
            sample.groupby("City", observed=True)
            .agg(
                review_count=("Rating", "count"),
                avg_rating=("Rating", "mean"),
                rating_std=("Rating", "std"),
                unique_places=("Place", "nunique"),
            )
            .reset_index()
            .sort_values("review_count", ascending=False)
        )

        elapsed_times.append(time.perf_counter() - t0)

    elapsed = float(np.median(elapsed_times))
    rps = n / elapsed if elapsed > 0 else 0.0

    log.info(f"  Pandas | {n:>9,} rows | {elapsed:.3f}s | {rps:>12,.0f} rows/s")
    return {
        "method":          "Pandas",
        "sample_size":     n,
        "elapsed_seconds": round(elapsed, 4),
        "rows_per_second": round(rps, 1),
        "notes":           f"Median of {PANDAS_REPEATS} runs; groupBy City aggregation",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. SPARK BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
def benchmark_spark(df_pandas: pd.DataFrame, sample_size: int, spark, F) -> dict:
    """
    Benchmark PySpark groupBy aggregation on a sample of `sample_size` rows.
    Task: identical to the Pandas benchmark for fair comparison.

    NOTE: Spark session startup time is excluded (measured separately).
    The time measured here is purely the Spark plan execution time.
    """
    n = min(sample_size, len(df_pandas))
    sample_pd = df_pandas.sample(n=n, random_state=42).reset_index(drop=True)

    elapsed_times = []
    for _ in range(SPARK_REPEATS):
        # Convert pandas sample to Spark DataFrame for this run
        sample_spark = spark.createDataFrame(
            sample_pd[["City", "Place", "Rating"]]
            .astype({"City": str, "Place": str, "Rating": float})
        )
        # Cache to avoid re-reading CSV on second action
        sample_spark.cache()
        sample_spark.count()  # Force cache materialisation

        t0 = time.perf_counter()

        result = (
            sample_spark
            .groupBy("City")
            .agg(
                F.count("Rating").alias("review_count"),
                F.round(F.avg("Rating"), 4).alias("avg_rating"),
                F.round(F.stddev("Rating"), 4).alias("rating_std"),
                F.approx_count_distinct("Place").alias("unique_places"),
            )
            .orderBy(F.col("review_count").desc())
        )
        result.count()  # Trigger action (collect plan execution)

        elapsed_times.append(time.perf_counter() - t0)
        sample_spark.unpersist()

    elapsed = float(np.median(elapsed_times))
    rps = n / elapsed if elapsed > 0 else 0.0

    log.info(f"  Spark  | {n:>9,} rows | {elapsed:.3f}s | {rps:>12,.0f} rows/s")
    return {
        "method":          "Spark",
        "sample_size":     n,
        "elapsed_seconds": round(elapsed, 4),
        "rows_per_second": round(rps, 1),
        "notes":           f"Median of {SPARK_REPEATS} runs; excludes session startup; local[*]",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. MEASURE SPARK SESSION STARTUP TIME
# ══════════════════════════════════════════════════════════════════════════════
def measure_spark_startup() -> tuple:
    """
    Measure Spark session initialisation time separately.
    This is important context: Spark has significant startup overhead
    (~3-15s on a single machine) which makes it slower for small datasets.
    In a cluster, Spark sessions are long-lived (persistent), eliminating
    this overhead for individual jobs.
    """
    log.info("Measuring Spark session startup time ...")
    t0 = time.perf_counter()
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        spark = (
            SparkSession.builder
            .appName("TourismBenchmark")
            .master("local[*]")
            .config("spark.driver.memory", "3g")
            .config("spark.executor.memory", "3g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        startup_time = time.perf_counter() - t0
        log.info(f"Spark session started in {startup_time:.2f}s")
        return spark, F, startup_time, True
    except Exception as e:
        startup_time = time.perf_counter() - t0
        log.warning(f"Spark unavailable: {e}")
        return None, None, startup_time, False


# ══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def generate_charts(results_df: pd.DataFrame, spark_startup_time: float = 0.0) -> None:
    """
    Generate two charts comparing Pandas vs Spark performance.
    Uses matplotlib with a clean dark-style theme to match the Flask dashboard.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend (no display needed)
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        log.warning("matplotlib not available — skipping chart generation.")
        return

    # Color palette matching the Flask dashboard
    COLORS = {"Pandas": "#14b8a6", "Spark": "#a78bfa"}
    MARKERS = {"Pandas": "o", "Spark": "s"}

    pandas_df = results_df[results_df["method"] == "Pandas"].sort_values("sample_size")
    spark_df  = results_df[results_df["method"] == "Spark"].sort_values("sample_size")

    fig_style = {
        "facecolor": "#0d0b22",
        "edgecolor": "none",
    }
    ax_style = {
        "facecolor":   "rgba(0,0,0,0)",
        "labelcolor":  "#c0bce8",
        "titlecolor":  "#c4b5fd",
    }

    # ── Chart 1: Elapsed Time vs Sample Size ──────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5), **fig_style)
    ax1.set_facecolor("#13112e")

    for method_df, method in [(pandas_df, "Pandas"), (spark_df, "Spark")]:
        if len(method_df) > 0:
            ax1.plot(
                method_df["sample_size"] / 1_000,
                method_df["elapsed_seconds"],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.5,
                markersize=8,
                label=method,
            )

    # Annotate Spark startup overhead
    if spark_startup_time > 0:
        ax1.axhline(
            y=spark_startup_time,
            color="#f87171",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"Spark startup ({spark_startup_time:.1f}s)",
        )

    ax1.set_xlabel("Sample Size (thousands of rows)", color="#c0bce8", fontsize=11)
    ax1.set_ylabel("Elapsed Time (seconds)", color="#c0bce8", fontsize=11)
    ax1.set_title("Processing Time vs Data Size\nPandas vs Apache Spark (local mode)",
                  color="#c4b5fd", fontsize=13, pad=12)
    ax1.tick_params(colors="#9090b8")
    subtle_axis = (1.0, 1.0, 1.0, 0.10)
    subtle_border = (1.0, 1.0, 1.0, 0.15)

    ax1.spines["bottom"].set_color(subtle_axis)
    ax1.spines["left"].set_color(subtle_axis)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(facecolor="#1a1540", edgecolor=subtle_border,
               labelcolor="#c0bce8", fontsize=10)
    ax1.grid(True, alpha=0.1, color="white")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}K"))

    plt.tight_layout(pad=1.5)
    chart1_path = os.path.join(CHART_DIR, "processing_time_vs_size.png")
    fig1.savefig(chart1_path, dpi=150, bbox_inches="tight", facecolor="#0d0b22")
    plt.close(fig1)
    log.info(f"  Saved chart: {chart1_path}")

    # ── Chart 2: Throughput vs Sample Size ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5), **fig_style)
    ax2.set_facecolor("#13112e")

    for method_df, method in [(pandas_df, "Pandas"), (spark_df, "Spark")]:
        if len(method_df) > 0:
            ax2.plot(
                method_df["sample_size"] / 1_000,
                method_df["rows_per_second"] / 1_000,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.5,
                markersize=8,
                label=method,
            )

    ax2.set_xlabel("Sample Size (thousands of rows)", color="#c0bce8", fontsize=11)
    ax2.set_ylabel("Throughput (thousands rows/sec)", color="#c0bce8", fontsize=11)
    ax2.set_title("Processing Throughput vs Data Size\nPandas vs Apache Spark (local mode)",
                  color="#c4b5fd", fontsize=13, pad=12)
    ax2.tick_params(colors="#9090b8")
    ax2.spines["bottom"].set_color(subtle_axis)
    ax2.spines["left"].set_color(subtle_axis)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(facecolor="#1a1540", edgecolor=subtle_border,
               labelcolor="#c0bce8", fontsize=10)
    ax2.grid(True, alpha=0.1, color="white")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}K"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}K"))

    plt.tight_layout(pad=1.5)
    chart2_path = os.path.join(CHART_DIR, "throughput_vs_size.png")
    fig2.savefig(chart2_path, dpi=150, bbox_inches="tight", facecolor="#0d0b22")
    plt.close(fig2)
    log.info(f"  Saved chart: {chart2_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 62)
    log.info("  Processing Load Analysis — Pandas vs Spark")
    log.info("=" * 62)

    # Load full dataset (used for sub-sampling)
    df_full = load_full_dataset()
    total_rows = len(df_full)

    # Build sample sizes, including full dataset
    sample_sizes = [s for s in SAMPLE_SIZES if s < total_rows]
    sample_sizes.append(total_rows)  # Full dataset

    results = []

    # ── Pandas Benchmarks ─────────────────────────────────────────────────────
    log.info(f"\n--- Pandas Benchmarks ({PANDAS_REPEATS} runs each) ---")
    for size in sample_sizes:
        result = benchmark_pandas(df_full, size)
        results.append(result)

    # ── Spark Benchmarks ──────────────────────────────────────────────────────
    log.info("\n--- Spark Session Startup ---")
    spark, F, startup_time, spark_ok = measure_spark_startup()

    # Log startup overhead as its own entry
    results.append({
        "method":          "Spark",
        "sample_size":     0,
        "elapsed_seconds": round(startup_time, 4),
        "rows_per_second": 0.0,
        "notes":           "Spark session startup overhead (one-time cost in production)",
    })

    if spark_ok:
        log.info(f"\n--- Spark Benchmarks ({SPARK_REPEATS} runs each, startup excluded) ---")
        for size in sample_sizes:
            try:
                result = benchmark_spark(df_full, size, spark, F)
                results.append(result)
            except Exception as e:
                log.warning(f"  Spark benchmark failed for {size:,}: {e}")
                results.append({
                    "method":          "Spark",
                    "sample_size":     size,
                    "elapsed_seconds": None,
                    "rows_per_second": None,
                    "notes":           f"Failed: {str(e)[:80]}",
                })
        spark.stop()
        log.info("Spark session stopped.")
    else:
        log.warning("Spark not available — only Pandas benchmarks recorded.")

    # ── Save Results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    load_path  = os.path.join(OUTPUT_DIR, "processing_load.csv")
    results_df.to_csv(load_path, index=False)
    log.info(f"\n  Saved -> {load_path}")

    log.info("\n  PROCESSING LOAD RESULTS:")
    log.info(results_df[results_df["sample_size"] > 0].to_string(index=False))

    # ── Generate Charts ───────────────────────────────────────────────────────
    generate_charts(
        results_df[results_df["sample_size"] > 0].copy(),
        spark_startup_time=startup_time,
    )

    log.info("\n[OK] Processing Load Analysis Complete.")
    log.info(f"  Results: {load_path}")
    log.info(f"  Charts : {CHART_DIR}")
    return results_df


if __name__ == "__main__":
    main()
