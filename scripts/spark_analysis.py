"""
spark_analysis.py
=================
Apache Spark Big Data Analysis for Tourism Trend Analytics.

WHY SPARK?
----------
The dataset contains 1,482,466 rows of tourism review data. While pandas can
handle this on a single machine with sufficient RAM, Apache Spark is the
industry-standard framework for distributed Big Data processing. This script
demonstrates how Spark techniques apply to the same dataset, enabling the
project to scale to datasets 10x or 100x larger without code changes.

SPARK TECHNIQUES DEMONSTRATED
------------------------------
1.  SparkSession creation with tuned configuration
2.  Explicit schema definition (StructType / StructField)
3.  DataFrame API (select, filter, withColumn, alias)
4.  cache() / persist() for iterative computation
5.  groupBy() aggregations (count, mean, stddev, min, max)
6.  join() operations (inner, left outer)
7.  Spark SQL (registerTempView + spark.sql())
8.  Window functions (rank, dense_rank, row_number, percent_rank)
9.  Sorting and filtering with Column expressions
10. Distributed aggregations with approxCountDistinct

OUTPUTS
-------
  outputs/spark_dataset_summary.csv    - overall dataset statistics
  outputs/spark_city_metrics.csv       - per-city aggregations
  outputs/spark_place_metrics.csv      - per-place aggregations
  outputs/spark_rating_distribution.csv - rating counts/percentages
  outputs/spark_top_cities.csv         - top-20 cities by review volume
  outputs/spark_top_places.csv         - top-20 places by review volume
  outputs/spark_state_metrics.csv      - state-level aggregated metrics

Run AFTER: analytics.py
Run with : python scripts/spark_analysis.py
"""

import os
import sys
import time
import logging
import warnings

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

# ─── City → State mapping (mirrors app.py for consistency) ───────────────────
CITY_TO_STATE = {
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra", "Aurangabad": "Maharashtra", "Thane": "Maharashtra",
    "Lonavala": "Maharashtra", "Mahabaleshwar": "Maharashtra", "Shirdi": "Maharashtra",
    "Jaipur": "Rajasthan", "Jodhpur": "Rajasthan", "Udaipur": "Rajasthan",
    "Ajmer": "Rajasthan", "Bikaner": "Rajasthan", "Jaisalmer": "Rajasthan",
    "Pushkar": "Rajasthan", "Mount Abu": "Rajasthan",
    "Agra": "Uttar Pradesh", "Lucknow": "Uttar Pradesh", "Varanasi": "Uttar Pradesh",
    "Mathura": "Uttar Pradesh", "Vrindavan": "Uttar Pradesh", "Ayodhya": "Uttar Pradesh",
    "Prayagraj": "Uttar Pradesh", "Gorakhpur": "Uttar Pradesh",
    "New Delhi": "Delhi", "Delhi": "Delhi",
    "Bengaluru": "Karnataka", "Bangalore": "Karnataka", "Mysore": "Karnataka",
    "Mysuru": "Karnataka", "Hampi": "Karnataka", "Coorg": "Karnataka",
    "Gokarna": "Karnataka", "Udupi": "Karnataka", "Chikmagalur": "Karnataka",
    "Chennai": "Tamil Nadu", "Madurai": "Tamil Nadu", "Coimbatore": "Tamil Nadu",
    "Ooty": "Tamil Nadu", "Kanyakumari": "Tamil Nadu", "Mahabalipuram": "Tamil Nadu",
    "Rameswaram": "Tamil Nadu", "Thanjavur": "Tamil Nadu", "Kodaikanal": "Tamil Nadu",
    "Kochi": "Kerala", "Thiruvananthapuram": "Kerala", "Munnar": "Kerala",
    "Alleppey": "Kerala", "Kozhikode": "Kerala", "Thrissur": "Kerala",
    "Wayanad": "Kerala", "Varkala": "Kerala", "Thekkady": "Kerala",
    "Kolkata": "West Bengal", "Darjeeling": "West Bengal", "Siliguri": "West Bengal",
    "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Vadodara": "Gujarat",
    "Rajkot": "Gujarat", "Dwarka": "Gujarat", "Somnath": "Gujarat", "Kutch": "Gujarat",
    "Panaji": "Goa", "Margao": "Goa", "Calangute": "Goa",
    "Anjuna": "Goa", "Baga": "Goa", "Panjim": "Goa",
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh", "Gwalior": "Madhya Pradesh",
    "Jabalpur": "Madhya Pradesh", "Ujjain": "Madhya Pradesh", "Khajuraho": "Madhya Pradesh",
    "Hyderabad": "Telangana", "Secunderabad": "Telangana", "Warangal": "Telangana",
    "Visakhapatnam": "Andhra Pradesh", "Vijayawada": "Andhra Pradesh",
    "Tirupati": "Andhra Pradesh",
    "Patna": "Bihar", "Bodh Gaya": "Bihar", "Gaya": "Bihar", "Nalanda": "Bihar",
    "Amritsar": "Punjab", "Ludhiana": "Punjab", "Jalandhar": "Punjab",
    "Chandigarh": "Chandigarh",
    "Shimla": "Himachal Pradesh", "Manali": "Himachal Pradesh",
    "Dharamsala": "Himachal Pradesh", "Dharamshala": "Himachal Pradesh",
    "Dehradun": "Uttarakhand", "Haridwar": "Uttarakhand", "Rishikesh": "Uttarakhand",
    "Nainital": "Uttarakhand", "Mussoorie": "Uttarakhand",
    "Guwahati": "Assam", "Kaziranga": "Assam",
    "Bhubaneswar": "Odisha", "Puri": "Odisha", "Konark": "Odisha",
    "Ranchi": "Jharkhand", "Jamshedpur": "Jharkhand",
    "Raipur": "Chhattisgarh",
    "Faridabad": "Haryana", "Gurgaon": "Haryana", "Gurugram": "Haryana",
    "Panipat": "Haryana",
    "Srinagar": "Jammu & Kashmir", "Gulmarg": "Jammu & Kashmir",
    "Pahalgam": "Jammu & Kashmir", "Jammu": "Jammu & Kashmir",
    "Gangtok": "Sikkim", "Pelling": "Sikkim",
    "Shillong": "Meghalaya", "Cherrapunji": "Meghalaya",
    "Imphal": "Manipur",
    "Kohima": "Nagaland",
    "Agartala": "Tripura",
    "Aizawl": "Mizoram",
    "Itanagar": "Arunachal Pradesh", "Tawang": "Arunachal Pradesh",
    "Port Blair": "Andaman & Nicobar Island",
    "Puducherry": "Puducherry", "Pondicherry": "Puducherry",
}


# ══════════════════════════════════════════════════════════════════════════════
# SPARK SESSION SETUP
# ══════════════════════════════════════════════════════════════════════════════
def create_spark_session():
    """
    Create and configure a SparkSession for local execution.

    TECHNIQUE: SparkSession is the unified entry point for all Spark
    functionality (replacing SparkContext + SQLContext + HiveContext).
    Configuration is tuned for single-machine processing of ~1.5M rows.

    In a real cluster environment, these configs would be set via
    spark-submit flags or a cluster manager (YARN/Kubernetes).
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        log.info("Initialising Apache Spark session ...")
        spark = (
            SparkSession.builder
            .appName("TourismTrendAnalytics_BDT")
            .master("local[*]")            # Use all CPU cores locally
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")   # Reduce for local mode
            .config("spark.sql.adaptive.enabled", "true")  # Adaptive Query Execution
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        log.info(f"Spark version: {spark.version}")
        log.info(f"Spark app name: {spark.sparkContext.appName}")
        return spark, F
    except ImportError:
        log.error("PySpark not installed. Run: pip install pyspark")
        sys.exit(1)
    except Exception as e:
        log.error(f"Failed to create Spark session: {e}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA WITH EXPLICIT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
def load_data_with_schema(spark, F):
    """
    Load Review_db.csv using an explicit Spark schema.

    TECHNIQUE: Explicit schema definition with StructType/StructField avoids
    Spark's schema inference pass (which requires reading the file twice).
    This is a best practice for production pipelines with known schema.

    The Date column is included but cast to StringType since it is 100% null
    in this dataset — demonstrating graceful null handling in Spark.
    """
    from pyspark.sql.types import (
        StructType, StructField, StringType, FloatType
    )

    # TECHNIQUE: Explicit schema — avoids costly inference scan
    schema = StructType([
        StructField("City",       StringType(), True),
        StructField("Place",      StringType(), True),
        StructField("Review",     StringType(), True),
        StructField("Rating",     FloatType(),  True),
        StructField("Name",       StringType(), True),
        StructField("Date",       StringType(), True),
        StructField("Raw_Review", StringType(), True),
    ])

    log.info(f"Loading dataset with explicit schema: {DATA_PATH}")
    t0 = time.time()

    df = (
        spark.read
        .option("header", "true")
        .option("encoding", "UTF-8")
        .option("mode", "PERMISSIVE")        # Skip malformed rows gracefully
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .schema(schema)
        .csv(DATA_PATH)
    )

    # TECHNIQUE: Drop rows with null Rating (primary analytics field)
    df = df.filter(F.col("Rating").isNotNull())
    # TECHNIQUE: Clamp ratings to [1,5] to handle data quality issues
    df = df.withColumn("Rating", F.greatest(F.lit(1.0), F.least(F.lit(5.0), F.col("Rating"))))
    # TECHNIQUE: Drop rows where City or Place is null
    df = df.filter(F.col("City").isNotNull() & F.col("Place").isNotNull())
    # TECHNIQUE: Trim whitespace from string columns
    df = df.withColumn("City",  F.trim(F.col("City")))
    df = df.withColumn("Place", F.trim(F.col("Place")))

    elapsed = time.time() - t0
    row_count = df.count()
    log.info(f"Loaded {row_count:,} rows in {elapsed:.2f}s")
    return df, row_count


# ══════════════════════════════════════════════════════════════════════════════
# 2. CACHE FOR PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def cache_dataframe(df, name="main_df"):
    """
    TECHNIQUE: cache() / persist()
    Spark DataFrames are lazy — each action re-reads from source.
    Calling cache() stores the DataFrame in memory after the first action,
    making all subsequent operations (groupBy, joins, SQL queries) much faster.

    In our case, the CSV is read once and all 6 aggregations share the cache.
    This is critical for Big Data pipelines where IO is the bottleneck.
    """
    log.info(f"Caching DataFrame '{name}' in memory ...")
    df = df.cache()
    # Force materialisation by counting (triggers cache population)
    count = df.count()
    log.info(f"DataFrame cached: {count:,} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. REGISTER TEMP VIEW FOR SPARK SQL
# ══════════════════════════════════════════════════════════════════════════════
def register_sql_views(df, spark):
    """
    TECHNIQUE: Spark SQL — registerTempView
    Spark SQL allows writing standard SQL queries against DataFrames.
    This bridges familiarity for analysts who know SQL but not the
    DataFrame API. Both approaches produce identical execution plans.
    """
    df.createOrReplaceTempView("reviews")
    log.info("Registered temp view: reviews")

    # TECHNIQUE: Spark SQL query — overall dataset statistics
    summary_sql = """
        SELECT
            COUNT(*)                          AS total_reviews,
            COUNT(DISTINCT City)              AS unique_cities,
            COUNT(DISTINCT Place)             AS unique_places,
            COUNT(DISTINCT Name)              AS unique_reviewers,
            ROUND(AVG(Rating), 4)             AS avg_rating,
            ROUND(STDDEV(Rating), 4)          AS rating_std,
            MIN(Rating)                       AS min_rating,
            MAX(Rating)                       AS max_rating,
            SUM(CASE WHEN Rating = 5 THEN 1 ELSE 0 END)  AS five_star_count,
            SUM(CASE WHEN Rating = 1 THEN 1 ELSE 0 END)  AS one_star_count
        FROM reviews
    """
    summary_df = spark.sql(summary_sql)
    log.info("Dataset summary (via Spark SQL):")
    summary_df.show(truncate=False)
    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. CITY-LEVEL AGGREGATIONS (DataFrame API)
# ══════════════════════════════════════════════════════════════════════════════
def compute_city_metrics(df, F):
    """
    TECHNIQUE: groupBy() + agg() with multiple aggregation functions.
    This is a distributed operation — each Spark partition computes partial
    aggregates, which are then shuffled and merged (similar to MapReduce
    combine + reduce phases). approxCountDistinct uses HyperLogLog for
    memory-efficient cardinality estimation at scale.
    """
    from pyspark.sql.functions import approx_count_distinct, countDistinct

    log.info("Computing city-level metrics via Spark groupBy aggregation ...")
    city_metrics = (
        df.groupBy("City")
        .agg(
            F.count("Rating").alias("review_count"),
            F.round(F.avg("Rating"), 4).alias("avg_rating"),
            F.round(F.stddev("Rating"), 4).alias("rating_std"),
            F.round(F.min("Rating"), 1).alias("min_rating"),
            F.round(F.max("Rating"), 1).alias("max_rating"),
            # TECHNIQUE: approxCountDistinct uses HyperLogLog algorithm
            # for memory-efficient distinct counting at scale
            approx_count_distinct("Place").alias("unique_places"),
            approx_count_distinct("Name").alias("unique_reviewers"),
            F.round(F.avg(F.when(F.col("Rating") == 5, 1).otherwise(0)), 4)
             .alias("five_star_rate"),
        )
        .orderBy(F.col("review_count").desc())
    )
    log.info(f"City metrics computed for {city_metrics.count()} cities")
    return city_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLACE-LEVEL AGGREGATIONS + CITY JOIN
# ══════════════════════════════════════════════════════════════════════════════
def compute_place_metrics(df, F):
    """
    TECHNIQUE: join() operation
    After computing per-place metrics, we join with city-level aggregates
    to enrich each place with its city's average rating and review count.
    This demonstrates Spark's distributed join capability — joins are
    automatically optimised (broadcast join for small tables, sort-merge
    join for large ones).
    """
    log.info("Computing place-level metrics ...")
    place_metrics = (
        df.groupBy("Place", "City")
        .agg(
            F.count("Rating").alias("review_count"),
            F.round(F.avg("Rating"), 4).alias("avg_rating"),
            F.round(F.stddev("Rating"), 4).alias("rating_std"),
        )
        .orderBy(F.col("review_count").desc())
    )

    # City-level summary for the join
    city_summary = (
        df.groupBy("City")
        .agg(
            F.count("Rating").alias("city_review_count"),
            F.round(F.avg("Rating"), 4).alias("city_avg_rating"),
            F.approx_count_distinct("Place").alias("city_place_count"),
        )
    )

    # TECHNIQUE: left outer join — keeps all places even if city lookup fails
    place_enriched = (
        place_metrics
        .join(city_summary, on="City", how="left")
        .orderBy(F.col("review_count").desc())
    )
    log.info(f"Place metrics computed and enriched for {place_enriched.count()} places")
    return place_enriched


# ══════════════════════════════════════════════════════════════════════════════
# 6. WINDOW FUNCTIONS — RANKING WITHIN GROUPS
# ══════════════════════════════════════════════════════════════════════════════
def apply_window_functions(city_metrics, place_metrics, F):
    """
    TECHNIQUE: Window Functions
    Window functions in Spark compute values across a set of rows related to
    the current row (similar to SQL window functions). Key functions used:
      - rank()         : gaps in rank sequence for ties
      - dense_rank()   : no gaps for ties
      - row_number()   : unique sequential numbering
      - percent_rank() : relative position as a fraction [0, 1]

    These are essential for leaderboard, percentile, and ranking analytics.
    In distributed computing, window operations use partitioned shuffles.
    """
    from pyspark.sql.window import Window

    log.info("Applying window functions for ranking ...")

    # ── City window: rank by review_count descending ──────────────────────────
    city_window = Window.orderBy(F.col("review_count").desc())

    city_ranked = city_metrics.withColumns({
        "rank":          F.rank().over(city_window),
        "dense_rank":    F.dense_rank().over(city_window),
        "row_num":       F.row_number().over(city_window),
        "pct_rank":      F.round(F.percent_rank().over(city_window), 4),
        "review_pct":    F.round(
            F.col("review_count") / F.sum("review_count").over(Window.rowsBetween(
                Window.unboundedPreceding, Window.unboundedFollowing
            )) * 100, 2
        ),
    })

    # ── Place window: rank within each city by review_count ───────────────────
    city_partition_window = (
        Window
        .partitionBy("City")                        # Reset rank for each city
        .orderBy(F.col("review_count").desc())      # Rank by reviews descending
    )

    place_ranked = place_metrics.withColumns({
        "city_rank":        F.rank().over(city_partition_window),
        "city_dense_rank":  F.dense_rank().over(city_partition_window),
        "city_row_number":  F.row_number().over(city_partition_window),
    })

    log.info("Window functions applied successfully")
    return city_ranked, place_ranked


# ══════════════════════════════════════════════════════════════════════════════
# 7. RATING DISTRIBUTION (Spark SQL + DataFrame API combined)
# ══════════════════════════════════════════════════════════════════════════════
def compute_rating_distribution(df, spark, F):
    """
    TECHNIQUE: Combined Spark SQL + DataFrame API
    Here we use Spark SQL for the initial groupBy and add computed columns
    via the DataFrame API. Both approaches produce the same execution plan.
    """
    log.info("Computing rating distribution ...")

    # TECHNIQUE: Spark SQL for readable groupBy
    rating_sql = """
        SELECT
            CAST(Rating AS INT) AS rating,
            COUNT(*) AS count
        FROM reviews
        GROUP BY CAST(Rating AS INT)
        ORDER BY rating
    """
    rating_dist = spark.sql(rating_sql)
    total = df.count()

    # TECHNIQUE: withColumn to add percentage column
    rating_dist = rating_dist.withColumn(
        "pct",
        F.round(F.col("count") / F.lit(total) * 100, 2)
    )

    rating_dist.show()
    return rating_dist


# ══════════════════════════════════════════════════════════════════════════════
# 8. STATE-LEVEL AGGREGATION via UDF + JOIN
# ══════════════════════════════════════════════════════════════════════════════
def compute_state_metrics(city_metrics, F, spark):
    """
    TECHNIQUE: User-Defined Functions (UDF) + join
    We apply a Python UDF to map city names to Indian states, then aggregate
    at the state level. UDFs allow custom Python logic within Spark's
    distributed execution engine, though they have serialization overhead
    (native Spark functions are preferred when available).

    For state-level analysis, this is necessary since state mapping is
    application-specific business logic.
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    log.info("Computing state-level metrics via UDF mapping ...")

    # TECHNIQUE: UDF (User-Defined Function) for city → state mapping
    city_state_udf = udf(lambda city: CITY_TO_STATE.get(city, None), StringType())

    city_with_state = city_metrics.withColumn(
        "State", city_state_udf(F.col("City"))
    ).filter(F.col("State").isNotNull())

    # Aggregate to state level
    state_metrics = (
        city_with_state
        .groupBy("State")
        .agg(
            F.sum("review_count").alias("total_reviews"),
            F.round(F.avg("avg_rating"), 4).alias("avg_rating"),
            F.round(F.avg("rating_std"), 4).alias("avg_rating_std"),
            F.count("City").alias("num_cities"),
            F.round(F.avg("five_star_rate"), 4).alias("avg_five_star_rate"),
        )
        .orderBy(F.col("total_reviews").desc())
    )

    log.info(f"State metrics computed for {state_metrics.count()} states")
    return state_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 9. SORTING AND FILTERING
# ══════════════════════════════════════════════════════════════════════════════
def get_top_n(city_ranked, place_ranked, F, top_n=20):
    """
    TECHNIQUE: filter() + orderBy() + limit()
    Spark's lazy evaluation means the filter is pushed down to the data
    source where possible (predicate pushdown), and limit() stops
    processing once enough rows are retrieved from each partition.
    """
    log.info(f"Extracting top-{top_n} cities and places ...")

    # TECHNIQUE: filter with Column expressions
    top_cities = (
        city_ranked
        .filter(F.col("review_count") > 100)   # Minimum review threshold
        .orderBy(F.col("review_count").desc())
        .limit(top_n)
    )

    top_places = (
        place_ranked
        .filter(F.col("review_count") > 10)
        .orderBy(F.col("review_count").desc())
        .limit(top_n)
    )

    log.info(f"Top cities: {top_cities.count()}, Top places: {top_places.count()}")
    return top_cities, top_places


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE OUTPUTS (Spark → pandas → CSV)
# ══════════════════════════════════════════════════════════════════════════════
def save_spark_outputs(
    summary_df,
    city_ranked,
    place_ranked,
    rating_dist,
    top_cities,
    top_places,
    state_metrics,
):
    """
    Convert Spark DataFrames to pandas and save as CSV.
    In production, Spark DataFrames would be written directly to HDFS,
    S3, or other distributed storage using df.write.parquet() or
    df.write.csv(). Here we use pandas for local file compatibility.
    """
    log.info("Saving Spark analysis outputs ...")

    outputs = {
        "spark_dataset_summary.csv":      summary_df,
        "spark_city_metrics.csv":         city_ranked,
        "spark_place_metrics.csv":        place_ranked,
        "spark_rating_distribution.csv":  rating_dist,
        "spark_top_cities.csv":           top_cities,
        "spark_top_places.csv":           top_places,
        "spark_state_metrics.csv":        state_metrics,
    }

    for fname, sdf in outputs.items():
        path = os.path.join(OUTPUT_DIR, fname)
        try:
            pdf = sdf.toPandas()
            pdf.to_csv(path, index=False)
            log.info(f"  Saved {fname} ({len(pdf)} rows)")
        except Exception as e:
            log.warning(f"  Could not save {fname}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 62)
    log.info("  Apache Spark Tourism Analytics — BDT Project")
    log.info("=" * 62)
    log.info("Demonstrating: SparkSession, Schema, DataFrame API, SQL,")
    log.info("  cache/persist, groupBy, joins, window functions, UDFs")
    log.info("=" * 62)

    t_start = time.time()

    # Step 1: Create Spark session
    spark, F = create_spark_session()

    # Step 2: Load data with explicit schema
    df, row_count = load_data_with_schema(spark, F)

    # Step 3: TECHNIQUE — cache() for reuse across multiple downstream ops
    df = cache_dataframe(df, name="reviews_cache")

    # Step 4: Register Spark SQL view + get summary
    summary_df = register_sql_views(df, spark)

    # Step 5: City-level aggregations (DataFrame API)
    city_metrics = compute_city_metrics(df, F)

    # Step 6: Place-level aggregations + city join (demonstrates JOIN)
    place_metrics = compute_place_metrics(df, F)

    # Step 7: Window functions for ranking
    city_ranked, place_ranked = apply_window_functions(city_metrics, place_metrics, F)

    # Step 8: Rating distribution (Spark SQL + DataFrame API)
    rating_dist = compute_rating_distribution(df, spark, F)

    # Step 9: State-level metrics via UDF
    state_metrics = compute_state_metrics(city_metrics, F, spark)

    # Step 10: Top-N filtering and sorting
    top_cities, top_places = get_top_n(city_ranked, place_ranked, F, top_n=20)

    # Step 11: Save all outputs
    save_spark_outputs(
        summary_df, city_ranked, place_ranked,
        rating_dist, top_cities, top_places, state_metrics
    )

    # Step 12: Stop Spark session (release resources)
    spark.stop()
    log.info("Spark session stopped.")

    elapsed = time.time() - t_start
    log.info(f"\n[OK] Spark Analysis complete in {elapsed:.1f}s")
    log.info(f"   Rows processed : {row_count:,}")
    log.info(f"   Output dir     : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
