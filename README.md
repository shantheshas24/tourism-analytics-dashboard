# Tourism Trend Analytics & Hotspot Prediction
## Big Data Technologies (BDT) — College AIML Project

A full-stack Big Data analytics dashboard for Indian tourism review data.
Analyses 1.48 million reviews using **Apache Spark**, Pandas, TextBlob, and
a **leakage-free Random Forest** model — presented through a premium Flask web application.

**Primary Dashboard URL:** http://localhost:5000

---

## Project Overview

| Property | Value |
|---|---|
| Dataset | Review_db.csv — 1,482,466 rows |
| Columns | City, Place, Review, Rating, Name, Raw_Review |
| Unique Cities | 1,794 |
| Unique Places | 14,494 |
| Primary Analytics Engine | Apache Spark (PySpark) |
| Secondary Engine | Pandas (for pre-processing and benchmarking) |
| ML Model | Random Forest Regressor (scikit-learn) |
| Web Framework | Flask (Python) |
| Visualisation | Plotly.js |
| Sentiment Tool | TextBlob |
| Trends API | Google Trends via pytrends (optional) |

---

## Architecture

```
data/
  Review_db.csv (1.48M rows)
        │
        ▼
scripts/ (offline processing pipeline)
  analytics.py          ─► Pandas load, aggregation, top-N
  sentiment.py          ─► TextBlob polarity/subjectivity
  spark_analysis.py     ─► Apache Spark distributed analysis
  hotspot.py            ─► Composite hotspot score formula
  prediction.py         ─► Leakage-free Random Forest
  processing_load_analysis.py ─► Pandas vs Spark benchmark
  google_trends.py      ─► Optional: real-time Google Trends
        │
        ▼
outputs/ (CSV/PKL artefacts, shared by Flask)
        │
        ▼
app.py  (Flask server — caches CSVs at startup)
        │
        ▼
templates/ + static/
  8 interactive Plotly dashboard pages
  Dark/light theme toggle
        │
        ▼
http://localhost:5000
```

---

## Apache Spark Usage

### Why Apache Spark?

The dataset (1.48M rows) represents a medium-scale Big Data workload.
Apache Spark is used as the primary analytics engine because:

1. **Scalability**: Spark's distributed architecture scales linearly.
   Adding 10 worker nodes gives ~10× throughput. Pandas is single-threaded.
2. **Industry standard**: Spark is the de facto Big Data processing framework
   in production environments (used by Netflix, Uber, Airbnb, etc.).
3. **SQL interface**: Spark SQL provides a familiar query interface on top of
   distributed DataFrames, enabling analysts to use standard SQL at scale.
4. **Memory efficiency**: cache()/persist() stores intermediate results in
   distributed memory, eliminating redundant data scans.

### Spark Techniques Used

| Technique | API | Purpose |
|---|---|---|
| **SparkSession** | `SparkSession.builder.appName(...).getOrCreate()` | Unified entry point; replaces SparkContext + SQLContext |
| **Explicit Schema** | `StructType([StructField(...)])` | Avoids 2-pass schema inference; production best practice |
| **DataFrame API** | `.filter()`, `.withColumn()`, `.select()`, `.alias()` | Lazy, optimised distributed data manipulation |
| **cache() / persist()** | `df.cache()` | Store materialised DataFrame in memory; avoid re-reading CSV per aggregation |
| **groupBy() + agg()** | `.groupBy("City").agg(count, avg, stddev)` | Distributed MapReduce aggregation across partitions |
| **join()** | `.join(city_summary, on="City", how="left")` | Enrich place metrics with city context; auto-broadcast-optimised |
| **Spark SQL** | `createOrReplaceTempView() + spark.sql(...)` | SQL interface on DataFrames; identical execution plan to DataFrame API |
| **Window Functions** | `rank(), dense_rank(), row_number(), percent_rank()` | Leaderboard ranking within groups (per-city, global) |
| **UDF** | `udf(lambda city: mapping.get(city), StringType())` | Custom Python city→state mapping within Spark execution |
| **approxCountDistinct** | `F.approx_count_distinct("Place")` | HyperLogLog cardinality estimation — memory-efficient at scale |
| **Sorting + Filter + Limit** | `.filter(...).orderBy(...).limit(20)` | Predicate pushdown; early stopping in distributed fetch |

**Script:** `scripts/spark_analysis.py`  
**Outputs:** `outputs/spark_*.csv` (7 files)

---

## Prediction Leakage Fix

### What the Leakage Was

The original `prediction.py` had a **target leakage** bug:

```python
# BUGGY (v1): city_review_count filled with place's own review_count
place_df["review_count_city"] = place_df["review_count_city"].fillna(
    place_df["review_count"]   # <-- LEAKAGE: target proxy used as feature
)
```

Since the prediction target is `log1p(review_count)`, using `review_count`
directly as a feature lets the model "see the answer" during training.
This produced impossible metrics: **R² = 1.00**.

### How It Was Fixed

1. **Full city mapping from raw data** — `build_place_city_mapping()` scans
   the entire 1.48M row dataset to assign every place to its most common city
   using mode aggregation. Saved as `outputs/place_city_mapping.csv`.
   No reliance on top_places.csv (which only covered 20 places).

2. **city_review_count removed from features entirely** — Even with correct
   city assignment, city review count is correlated with place review count
   and introduces indirect leakage.

3. **Leakage-safe feature set:**

| Feature | Type | Safe? | Reason |
|---|---|---|---|
| `avg_rating` | Place quality | ✅ | Not derived from review_count |
| `rating_std` | Rating consistency | ✅ | Not derived from review_count |
| `avg_polarity` | Sentiment | ✅ | TextBlob score, not count-based |
| `avg_subjectivity` | Sentiment | ✅ | TextBlob score, not count-based |
| `city_avg_rating` | City quality | ✅ | Aggregate rating, not counts |
| `city_place_count` | City diversity | ✅ | Number of places, not reviews |
| `city_avg_polarity` | City sentiment | ✅ | Aggregate score, not counts |
| `city_five_star_rate` | City excellence | ✅ | Fraction, not raw count |

### Expected Metrics (Post-Fix)

Realistic R² of **~0.30–0.65** on test set.  
(Before fix: R² = 1.00 — obviously leakage-inflated.)

---

## Processing Load Analysis

**Script:** `scripts/processing_load_analysis.py`

Benchmarks measured on this machine (Spark local[*] mode, startup excluded from per-query times):

| Sample Size | Pandas elapsed | Pandas rows/s | Spark elapsed | Spark rows/s | Winner |
|---|---|---|---|---|---|
| 100,000 | 0.42s | 238,610 | 2.09s | 47,838 | **Pandas** |
| 500,000 | 2.36s | 211,615 | 0.71s | 699,599 | **Spark** |
| 1,000,000 | 1.99s | 502,149 | 1.16s | 862,484 | **Spark** |
| 1,482,466 (full) | 2.71s | 547,047 | 1.06s | **1,399,112** | **Spark** |
| Spark startup | — | — | **23.5s** | — | one-time cost |

**Key finding:** Spark overtakes Pandas at ~500K rows and is **2.56× faster** at the full dataset.
Spark startup (23.5s) is a one-time JVM cost — in production clusters it is persistent (no per-job overhead).

**Interpretation:**
- Pandas is faster on a single machine for this dataset size.
- Spark's overhead (JVM startup, inter-process serialisation) is fixed.
- At scale (100M+ rows, multi-node clusters), Spark's parallelism dominates.
- In production, Spark sessions are persistent — no per-job startup cost.

**Outputs:**
- `outputs/processing_load.csv`
- `outputs/charts/processing_time_vs_size.png`
- `outputs/charts/throughput_vs_size.png`

---

## Google Trends Handling

**Script:** `scripts/google_trends.py`

Every record now includes transparent metadata:

| Column | Values | Meaning |
|---|---|---|
| `RawTrendScore` | 0–100 | Raw interest score from Google API |
| `TrendScore` | 0–1 | Normalised score (0 for fallback) |
| `status` | `live` / `fallback` / `unavailable` | Data quality indicator |
| `source` | `Google Trends` / `fallback` / `cached` | Data origin |
| `notes` | Text | Human-readable explanation |

**Failure policy:**
- pytrends not installed → warning, fallback, continue
- API rate-limited → exponential back-off, then fallback
- Network error → record honestly, continue
- Any exception → **execution never aborts**

**Formula:** `FinalRealtimeScore = 0.9 × ExistingHotspotScore + 0.1 × TrendScore`  
When TrendScore = 0 (fallback), score = 0.9 × ExistingHotspotScore (honest).

---

## India Heat Map

**Page:** http://localhost:5000/india-map  
**Data:** `outputs/state_metrics.csv` (auto-generated at Flask startup)

**Fields:**
| Column | Description |
|---|---|
| `State` | Indian state name |
| `ReviewCount` | Total reviews aggregated from city data |
| `AverageRating` | Mean star rating across cities in state |
| `SentimentScore` | Mean TextBlob polarity across cities |
| `HotspotScore` | Mean composite hotspot score |
| `NumCities` | Number of distinct cities mapped |

**Visualisation features:**
- GeoJSON India state boundaries (from public CDN)
- Choropleth heat colours based on HotspotScore
- Hover tooltips with all 5 metrics
- Dark/light theme support

---

## Project Structure

```
BDT_project/
│
├── app.py                        # Flask server (PRIMARY — localhost:5000)
├── run_pipeline.bat              # Full pipeline execution script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── data/
│   └── Review_db.csv             # 1.48M row tourism reviews dataset
│
├── scripts/
│   ├── analytics.py              # Pandas: load, aggregate, top-N
│   ├── sentiment.py              # TextBlob: polarity, subjectivity
│   ├── spark_analysis.py         # Apache Spark: 10 techniques demonstrated
│   ├── hotspot.py                # Composite hotspot scoring
│   ├── prediction.py             # Leakage-free Random Forest
│   ├── processing_load_analysis.py # Pandas vs Spark benchmark
│   └── google_trends.py          # Optional: Google Trends API
│
├── outputs/
│   ├── rating_distribution.csv
│   ├── top_cities.csv
│   ├── city_ratings.csv
│   ├── place_ratings.csv
│   ├── top_places.csv
│   ├── sentiment_results.csv
│   ├── sentiment_by_city.csv
│   ├── sentiment_by_place.csv
│   ├── hotspots_city.csv
│   ├── hotspots_place.csv
│   ├── google_trends.csv          # status/source/notes columns
│   ├── realtime_hotspots.csv
│   ├── place_city_mapping.csv     # Full place→city map (leakage fix)
│   ├── prediction_results.csv
│   ├── all_place_predictions.csv
│   ├── model_metrics.csv
│   ├── feature_importances.csv
│   ├── rf_model.pkl
│   ├── state_metrics.csv          # India map state aggregations
│   ├── processing_load.csv        # Pandas vs Spark benchmark
│   ├── spark_dataset_summary.csv
│   ├── spark_city_metrics.csv
│   ├── spark_place_metrics.csv
│   ├── spark_rating_distribution.csv
│   ├── spark_top_cities.csv
│   ├── spark_top_places.csv
│   ├── spark_state_metrics.csv
│   └── charts/
│       ├── processing_time_vs_size.png
│       └── throughput_vs_size.png
│
├── templates/
│   ├── base.html                  # Shared layout, sidebar, theme toggle
│   ├── dashboard.html             # KPI summary, top charts
│   ├── sentiment.html             # Polarity histograms, city/place sentiment
│   ├── hotspots.html              # Ranked hotspot city/place tables
│   ├── realtime.html              # Google Trends real-time blend
│   ├── prediction.html            # RF model metrics, scatter, feature importance
│   ├── india_map.html             # State choropleth map
│   ├── spark.html                 # Spark analysis results + technique table
│   └── processing_load.html       # Pandas vs Spark benchmark charts
│
├── static/
│   ├── css/style.css              # Dark/light theme, glassmorphism design
│   └── js/main.js                 # Shared fetch utilities, Plotly helpers
│
└── report_assets/
    ├── CAPTIONS.md                # Figure captions for report
    └── charts/                    # Copy charts here for report use
```

---

## How to Run

### Prerequisites

```bash
pip install flask pandas numpy scikit-learn textblob joblib matplotlib pyspark pytrends
```

### Option 1 — Full Pipeline (Recommended)

```bat
run_pipeline.bat
```

This runs all 7 scripts in order, then launches Flask at http://localhost:5000.

### Option 2 — Flask Only (if outputs exist)

```bash
python app.py
```

### Option 3 — Individual Scripts

```bash
python scripts/analytics.py
python scripts/sentiment.py
python scripts/spark_analysis.py
python scripts/hotspot.py
python scripts/prediction.py
python scripts/processing_load_analysis.py
python scripts/google_trends.py   # Optional
python app.py
```

---

## Generated Outputs

| Script | Key Outputs |
|---|---|
| analytics.py | top_cities.csv, top_places.csv, city_ratings.csv, place_ratings.csv, rating_distribution.csv |
| sentiment.py | sentiment_results.csv, sentiment_by_city.csv, sentiment_by_place.csv |
| spark_analysis.py | spark_dataset_summary.csv, spark_city_metrics.csv, spark_place_metrics.csv, spark_rating_distribution.csv, spark_top_cities.csv, spark_top_places.csv, spark_state_metrics.csv |
| hotspot.py | hotspots_city.csv, hotspots_place.csv |
| prediction.py | prediction_results.csv, all_place_predictions.csv, model_metrics.csv, feature_importances.csv, rf_model.pkl, place_city_mapping.csv |
| processing_load_analysis.py | processing_load.csv, charts/processing_time_vs_size.png, charts/throughput_vs_size.png |
| google_trends.py | google_trends.csv (with status/source/notes), realtime_hotspots.csv |
| app.py (startup) | state_metrics.csv (auto-generated) |

---

## Expected Runtime

| Script | Expected Time |
|---|---|
| analytics.py | ~30–60s (1.48M rows, chunked load) |
| sentiment.py | ~30–90 min (TextBlob on 1.48M rows) — skip if sentiment_results.csv exists |
| spark_analysis.py | ~3–8 min (Spark local mode including startup) |
| hotspot.py | ~10–30s |
| prediction.py | ~2–5 min (includes full city mapping scan on first run) |
| processing_load_analysis.py | ~5–15 min (multiple Spark + Pandas runs) |
| google_trends.py | ~1–3 min (or instant if fallback) |

> **Note:** `sentiment.py` is the most time-consuming. If `sentiment_results.csv`
> already exists in `outputs/`, it will be reused. Do not delete it unless
> regeneration is necessary.

---

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | http://localhost:5000/ | KPI cards, top cities, rating distribution |
| Sentiment Analysis | http://localhost:5000/sentiment | Polarity histogram, city/place sentiment rankings |
| Hotspot Rankings | http://localhost:5000/hotspots | Composite hotspot score tables and charts |
| Real-Time Trends | http://localhost:5000/realtime | Google Trends blend, status transparency |
| Prediction | http://localhost:5000/prediction | RF metrics, feature importance, scatter plot |
| India Tourism Map | http://localhost:5000/india-map | State choropleth with 5-metric tooltips |
| Spark Analysis | http://localhost:5000/spark | Spark outputs + full technique reference table |
| Processing Load | http://localhost:5000/processing-load | Pandas vs Spark benchmark charts and raw data |

---

## Report Sections Support

| Report Section | Evidence Source |
|---|---|
| Apache Spark Techniques Used | spark_analysis.py comments + /spark dashboard |
| Prediction Leakage Fix | prediction.py docstring + model_metrics.csv |
| Analysis of Processing Load Impact | processing_load_analysis.py + /processing-load dashboard |
| Google Trends Integration | google_trends.csv (status/source/notes columns) |
| Visualization Results | All 8 dashboard pages |
| India State Heat Map | india_map.html + state_metrics.csv |
| Dataset Statistics | spark_dataset_summary.csv + spark_rating_distribution.csv |

---

## Limitations & Assumptions

1. **Date column**: 100% null in Review_db.csv — no temporal analysis possible.
2. **TextBlob**: English-only sentiment. Non-English reviews may score neutral.
3. **Popularity proxy**: `log1p(review_count)` used as popularity (no actual visitor count data).
4. **Spark local mode**: Single-machine Spark — true distributed gains require a cluster.
5. **Google Trends**: May be rate-limited or unavailable. Dashboard works without it.
6. **City-to-state mapping**: Manually curated for ~150 Indian cities. Unmapped cities are excluded from state view.
7. **Sentiment sampling**: Only 60,000 rows loaded into Flask memory for performance (full 1.48M used in scripts).
