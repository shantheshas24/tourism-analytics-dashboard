"""
app.py
======
Flask Backend — Tourism Trend Analytics & Hotspot Prediction.

Routes:
  /                   -> Dashboard
  /dashboard          -> Dashboard
  /sentiment          -> Sentiment Analysis
  /hotspots           -> Hotspot Rankings
  /realtime           -> Real-Time Google Trends
  /prediction         -> Popularity Prediction
  /india-map          -> India Choropleth Map
  /spark              -> Apache Spark Analysis Results
  /processing-load    -> Pandas vs Spark Benchmark
  /simulator          -> Hotspot Weight Simulator  [NEW]
  /compare            -> Destination Compare + Similar Recommender  [NEW]

API Endpoints:
  /api/dashboard-stats
  /api/sentiment-data
  /api/hotspot-data
  /api/realtime-data
  /api/prediction-data
  /api/india-map-data
  /api/spark-data
  /api/processing-load
  /api/simulator-data   [NEW]
  /api/compare-data     [NEW]
  /api/similar-places   [NEW]

Run:
    python app.py
    (visit http://localhost:5000)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

warnings.filterwarnings("ignore")

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ══════════════════════════════════════════════════════════════════════════════
# DATA CACHE  (loaded once at startup to avoid repeated disk reads)
# ══════════════════════════════════════════════════════════════════════════════
_CACHE: dict = {}


def _load(filename: str, **kwargs) -> pd.DataFrame | None:
    """Read a CSV from the outputs directory; return None if missing."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        print(f"[WARN] Could not load {filename}: {exc}")
        return None


def _init_cache() -> None:
    """Pre-load all output CSVs into memory once at startup."""
    global _CACHE
    print("  Loading output datasets ...")

    _CACHE["rating_dist"] = _load("rating_distribution.csv")
    _CACHE["top_cities"]  = _load("top_cities.csv")
    _CACHE["hs_city"]     = _load("hotspots_city.csv")
    _CACHE["hs_place"]    = _load("hotspots_place.csv")
    _CACHE["sent_city"]   = _load("sentiment_by_city.csv")
    _CACHE["sent_place"]  = _load("sentiment_by_place.csv")
    _CACHE["gt_df"]       = _load("google_trends.csv")
    _CACHE["rt_df"]       = _load("realtime_hotspots.csv")
    _CACHE["pred_test"]   = _load("prediction_results.csv")
    _CACHE["metrics"]     = _load("model_metrics.csv")
    _CACHE["fi_df"]       = _load("feature_importances.csv")
    _CACHE["pred_all"]    = _load("all_place_predictions.csv")

    # Apache Spark analysis outputs
    _CACHE["spark_summary"]  = _load("spark_dataset_summary.csv")
    _CACHE["spark_cities"]   = _load("spark_top_cities.csv")
    _CACHE["spark_places"]   = _load("spark_top_places.csv")
    _CACHE["spark_rating"]   = _load("spark_rating_distribution.csv")
    _CACHE["spark_city_m"]   = _load("spark_city_metrics.csv")
    _CACHE["spark_state"]    = _load("spark_state_metrics.csv")

    # Processing load benchmark
    _CACHE["proc_load"]   = _load("processing_load.csv")

    # State metrics (for India map)
    _CACHE["state_metrics"] = _load("state_metrics.csv")
    if _CACHE["state_metrics"] is None:
        _CACHE["state_metrics"] = _build_state_metrics()

    # Large sentiment file — sample 60K rows to keep memory manageable
    _CACHE["sent_sample"] = None

    print("  [OK] Data loaded.")


def D(key: str) -> pd.DataFrame | None:
    """Shorthand accessor for the global data cache."""
    return _CACHE.get(key)


# ══════════════════════════════════════════════════════════════════════════════
# STATE METRICS BUILDER (fallback if state_metrics.csv not yet generated)
# ══════════════════════════════════════════════════════════════════════════════
def _build_state_metrics() -> pd.DataFrame | None:
    """
    Build state_metrics.csv from hotspots_city.csv + CITY_TO_STATE mapping.
    Saved to outputs/state_metrics.csv for reuse and report reference.
    """
    hs_city = _CACHE.get("hs_city")
    sent_city = _CACHE.get("sent_city")
    if hs_city is None:
        return None
    try:
        df = hs_city.copy()
        df["State"] = df["City"].map(CITY_TO_STATE)
        mapped = df.dropna(subset=["State"]).copy()
        if mapped.empty:
            return None

        agg_cols: dict = {
            "ReviewCount":    ("review_count",  "sum"),
            "AverageRating":  ("avg_rating",    "mean"),
            "HotspotScore":   ("hotspot_score", "mean"),
            "NumCities":      ("City",          "count"),
        }
        if "avg_polarity" in mapped.columns:
            agg_cols["SentimentScore"] = ("avg_polarity", "mean")

        state_df = mapped.groupby("State").agg(**agg_cols).reset_index()

        # Merge city-level sentiment if available
        if sent_city is not None and "SentimentScore" not in state_df.columns:
            sent_state = sent_city.copy()
            sent_state["State"] = sent_state["City"].map(CITY_TO_STATE)
            sent_agg = sent_state.dropna(subset=["State"]).groupby("State")["avg_polarity"].mean().reset_index()
            sent_agg = sent_agg.rename(columns={"avg_polarity": "SentimentScore"})
            state_df = state_df.merge(sent_agg, on="State", how="left")

        if "SentimentScore" not in state_df.columns:
            state_df["SentimentScore"] = 0.0

        for col in ["AverageRating", "SentimentScore", "HotspotScore"]:
            if col in state_df.columns:
                state_df[col] = state_df[col].round(4)

        state_df = state_df.sort_values("HotspotScore", ascending=False).reset_index(drop=True)

        # Save for reuse
        save_path = os.path.join(OUTPUT_DIR, "state_metrics.csv")
        state_df.to_csv(save_path, index=False)
        print(f"  [OK] state_metrics.csv generated ({len(state_df)} states).")
        return state_df
    except Exception as e:
        print(f"  [WARN] Could not build state_metrics: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def safe_records(df: pd.DataFrame | None, n: int | None = None) -> list:
    """Convert DataFrame to JSON-serialisable list of records."""
    if df is None or df.empty:
        return []
    if n:
        df = df.head(n)
    df = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(df.to_json(orient="records"))


# ══════════════════════════════════════════════════════════════════════════════
# CITY → STATE MAPPING  (for India choropleth)
# State names must match the GeoJSON property  ST_NM
# ══════════════════════════════════════════════════════════════════════════════
CITY_TO_STATE: dict[str, str] = {
    # Maharashtra
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra", "Aurangabad": "Maharashtra", "Thane": "Maharashtra",
    "Solapur": "Maharashtra", "Kolhapur": "Maharashtra", "Nanded": "Maharashtra",
    "Lonavala": "Maharashtra", "Mahabaleshwar": "Maharashtra", "Alibaug": "Maharashtra",
    "Shirdi": "Maharashtra", "Khopoli": "Maharashtra", "Satara": "Maharashtra",
    "Ratnagiri": "Maharashtra", "Palghar": "Maharashtra", "Amravati": "Maharashtra",
    # Rajasthan
    "Jaipur": "Rajasthan", "Jodhpur": "Rajasthan", "Udaipur": "Rajasthan",
    "Ajmer": "Rajasthan", "Bikaner": "Rajasthan", "Kota": "Rajasthan",
    "Alwar": "Rajasthan", "Pushkar": "Rajasthan", "Jaisalmer": "Rajasthan",
    "Amer": "Rajasthan", "Mount Abu": "Rajasthan", "Chittorgarh": "Rajasthan",
    "Sawai Madhopur": "Rajasthan", "Ranthambore": "Rajasthan", "Bharatpur": "Rajasthan",
    "Bundi": "Rajasthan", "Sikar": "Rajasthan",
    # Uttar Pradesh
    "Agra": "Uttar Pradesh", "Lucknow": "Uttar Pradesh", "Varanasi": "Uttar Pradesh",
    "Mathura": "Uttar Pradesh", "Allahabad": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
    "Vrindavan": "Uttar Pradesh", "Noida": "Uttar Pradesh", "Ghaziabad": "Uttar Pradesh",
    "Meerut": "Uttar Pradesh", "Aligarh": "Uttar Pradesh", "Prayagraj": "Uttar Pradesh",
    "Ayodhya": "Uttar Pradesh", "Gorakhpur": "Uttar Pradesh", "Bareilly": "Uttar Pradesh",
    # Delhi
    "New Delhi": "Delhi", "Delhi": "Delhi",
    # Karnataka
    "Bengaluru": "Karnataka", "Bangalore": "Karnataka", "Mysore": "Karnataka",
    "Mysuru": "Karnataka", "Hampi": "Karnataka", "Coorg": "Karnataka",
    "Mangalore": "Karnataka", "Hubli": "Karnataka", "Hassan": "Karnataka",
    "Chikmagalur": "Karnataka", "Davangere": "Karnataka", "Kodagu": "Karnataka",
    "Gokarna": "Karnataka", "Udupi": "Karnataka", "Badami": "Karnataka",
    "Belur": "Karnataka",
    # Tamil Nadu
    "Chennai": "Tamil Nadu", "Madurai": "Tamil Nadu", "Coimbatore": "Tamil Nadu",
    "Ooty": "Tamil Nadu", "Tiruchirappalli": "Tamil Nadu", "Salem": "Tamil Nadu",
    "Kanyakumari": "Tamil Nadu", "Mahabalipuram": "Tamil Nadu", "Rameswaram": "Tamil Nadu",
    "Thanjavur": "Tamil Nadu", "Vellore": "Tamil Nadu", "Kodaikanal": "Tamil Nadu",
    "Tiruvannamalai": "Tamil Nadu", "Chidambaram": "Tamil Nadu",
    # Kerala
    "Kochi": "Kerala", "Thiruvananthapuram": "Kerala", "Munnar": "Kerala",
    "Alleppey": "Kerala", "Kozhikode": "Kerala", "Thrissur": "Kerala",
    "Alappuzha": "Kerala", "Varkala": "Kerala", "Thekkady": "Kerala",
    "Wayanad": "Kerala", "Kannur": "Kerala", "Palakkad": "Kerala",
    "Kovalam": "Kerala", "Kumarakom": "Kerala",
    # West Bengal
    "Kolkata": "West Bengal", "Darjeeling": "West Bengal", "Siliguri": "West Bengal",
    "Durgapur": "West Bengal", "Howrah": "West Bengal", "Asansol": "West Bengal",
    "Bishnupur": "West Bengal", "Sundarbans": "West Bengal", "Shantiniketan": "West Bengal",
    # Gujarat
    "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Vadodara": "Gujarat",
    "Rajkot": "Gujarat", "Dwarka": "Gujarat", "Somnath": "Gujarat",
    "Gir": "Gujarat", "Gandhinagar": "Gujarat", "Bhavnagar": "Gujarat",
    "Kutch": "Gujarat", "Rann of Kutch": "Gujarat", "Junagadh": "Gujarat",
    "Porbandar": "Gujarat",
    # Goa
    "Panaji": "Goa", "Margao": "Goa", "Vasco da Gama": "Goa",
    "Calangute": "Goa", "Anjuna": "Goa", "Vagator": "Goa",
    "Panjim": "Goa", "Baga": "Goa", "Colva": "Goa", "Mapusa": "Goa",
    # Madhya Pradesh
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh", "Gwalior": "Madhya Pradesh",
    "Jabalpur": "Madhya Pradesh", "Ujjain": "Madhya Pradesh", "Khajuraho": "Madhya Pradesh",
    "Pachmarhi": "Madhya Pradesh", "Orchha": "Madhya Pradesh", "Sanchi": "Madhya Pradesh",
    # Telangana
    "Hyderabad": "Telangana", "Secunderabad": "Telangana", "Warangal": "Telangana",
    "Nizamabad": "Telangana", "Karimnagar": "Telangana",
    # Andhra Pradesh
    "Visakhapatnam": "Andhra Pradesh", "Vijayawada": "Andhra Pradesh",
    "Tirupati": "Andhra Pradesh", "Amaravati": "Andhra Pradesh",
    "Guntur": "Andhra Pradesh", "Nellore": "Andhra Pradesh",
    # Bihar
    "Patna": "Bihar", "Bodh Gaya": "Bihar", "Gaya": "Bihar",
    "Nalanda": "Bihar", "Rajgir": "Bihar", "Vaishali": "Bihar",
    "Muzaffarpur": "Bihar",
    # Punjab
    "Amritsar": "Punjab", "Ludhiana": "Punjab", "Jalandhar": "Punjab",
    "Patiala": "Punjab", "Bathinda": "Punjab",
    # Chandigarh
    "Chandigarh": "Chandigarh",
    # Himachal Pradesh
    "Shimla": "Himachal Pradesh", "Manali": "Himachal Pradesh",
    "Dharamsala": "Himachal Pradesh", "Dharamshala": "Himachal Pradesh",
    "Kullu": "Himachal Pradesh", "Kasauli": "Himachal Pradesh",
    "Dalhousie": "Himachal Pradesh", "Spiti": "Himachal Pradesh",
    "McLeod Ganj": "Himachal Pradesh",
    # Uttarakhand
    "Dehradun": "Uttarakhand", "Haridwar": "Uttarakhand", "Rishikesh": "Uttarakhand",
    "Nainital": "Uttarakhand", "Mussoorie": "Uttarakhand", "Jim Corbett": "Uttarakhand",
    "Kedarnath": "Uttarakhand", "Badrinath": "Uttarakhand", "Auli": "Uttarakhand",
    # Assam
    "Guwahati": "Assam", "Kaziranga": "Assam", "Jorhat": "Assam",
    "Dibrugarh": "Assam", "Silchar": "Assam", "Tezpur": "Assam",
    # Odisha
    "Bhubaneswar": "Odisha", "Puri": "Odisha", "Konark": "Odisha",
    "Cuttack": "Odisha", "Rourkela": "Odisha", "Sambalpur": "Odisha",
    # Jharkhand
    "Ranchi": "Jharkhand", "Jamshedpur": "Jharkhand", "Dhanbad": "Jharkhand",
    "Bokaro": "Jharkhand",
    # Chhattisgarh
    "Raipur": "Chhattisgarh", "Bilaspur": "Chhattisgarh", "Durg": "Chhattisgarh",
    # Haryana
    "Faridabad": "Haryana", "Gurgaon": "Haryana", "Gurugram": "Haryana",
    "Panipat": "Haryana", "Kurukshetra": "Haryana", "Ambala": "Haryana",
    "Rohtak": "Haryana", "Hisar": "Haryana",
    # Jammu & Kashmir
    "Srinagar": "Jammu & Kashmir", "Gulmarg": "Jammu & Kashmir",
    "Pahalgam": "Jammu & Kashmir", "Jammu": "Jammu & Kashmir",
    "Sonamarg": "Jammu & Kashmir", "Leh": "Jammu & Kashmir",
    # Sikkim
    "Gangtok": "Sikkim", "Pelling": "Sikkim", "Lachung": "Sikkim",
    # Meghalaya
    "Shillong": "Meghalaya", "Cherrapunji": "Meghalaya",
    # Manipur
    "Imphal": "Manipur",
    # Nagaland
    "Kohima": "Nagaland", "Dimapur": "Nagaland",
    # Tripura
    "Agartala": "Tripura",
    # Mizoram
    "Aizawl": "Mizoram",
    # Arunachal Pradesh
    "Itanagar": "Arunachal Pradesh", "Tawang": "Arunachal Pradesh",
    # Andaman & Nicobar
    "Port Blair": "Andaman & Nicobar Island",
    # Puducherry
    "Puducherry": "Puducherry", "Pondicherry": "Puducherry",
}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
@app.route("/dashboard")
def page_dashboard():
    return render_template("dashboard.html", active="dashboard")


@app.route("/sentiment")
def page_sentiment():
    return render_template("sentiment.html", active="sentiment")


@app.route("/hotspots")
def page_hotspots():
    return render_template("hotspots.html", active="hotspots")


@app.route("/realtime")
def page_realtime():
    return render_template("realtime.html", active="realtime")


@app.route("/prediction")
def page_prediction():
    return render_template("prediction.html", active="prediction")


@app.route("/india-map")
def page_india_map():
    return render_template("india_map.html", active="india_map")


@app.route("/spark")
def page_spark():
    return render_template("spark.html", active="spark")


@app.route("/processing-load")
def page_processing_load():
    return render_template("processing_load.html", active="processing_load")


@app.route("/simulator")
def page_simulator():
    return render_template("simulator.html", active="simulator")


@app.route("/compare")
def page_compare():
    return render_template("compare.html", active="compare")


# ─── Serve benchmark charts from outputs/charts/ ──────────────────────────────
@app.route("/charts/<path:filename>")
def serve_chart(filename):
    """Serve generated chart PNGs from outputs/charts/ directory."""
    charts_dir = os.path.join(OUTPUT_DIR, "charts")
    return send_from_directory(charts_dir, filename)


# ══════════════════════════════════════════════════════════════════════════════
# API: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/dashboard-stats")
def api_dashboard_stats():
    rating_dist = D("rating_dist")
    avg_rating  = None
    if rating_dist is not None:
        total_w = (rating_dist["rating"] * rating_dist["count"]).sum()
        total_c = rating_dist["count"].sum()
        avg_rating = round(float(total_w / total_c), 3) if total_c else None

    return jsonify({
        "kpis": {
            "total_reviews": 1_482_466,
            "total_cities":  1_794,
            "total_places":  14_494,
            "avg_rating":    avg_rating,
        },
        "rating_distribution": safe_records(rating_dist),
        "top_cities":          safe_records(D("top_cities"), 20),
        "top_hotspot_cities":  safe_records(D("hs_city"),    15),
        "top_hotspot_places":  safe_records(D("hs_place"),   10),
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/sentiment-data")
def api_sentiment_data():
    sample    = D("sent_sample")
    sent_city = D("sent_city")
    sent_place = D("sent_place")
    hs_place  = D("hs_place")

    distribution  = []
    polarity_hist = []

    if sample is not None:
        # Sentiment label distribution
        vc = sample["sentiment"].value_counts().reset_index()
        vc.columns = ["label", "count"]
        distribution = safe_records(vc)

        # Polarity histogram
        counts, bins = np.histogram(sample["polarity"].dropna(), bins=50)
        polarity_hist = [
            {"bin_center": round(float((bins[i] + bins[i + 1]) / 2), 4),
             "count": int(counts[i])}
            for i in range(len(counts))
        ]

    # Top-20 treemap data from hotspot places (visual word-map substitute)
    treemap_data = []
    if hs_place is not None:
        top80 = hs_place.head(80)
        treemap_data = safe_records(top80[["Place", "hotspot_score", "review_count", "avg_polarity"]])

    return jsonify({
        "distribution":        distribution,
        "polarity_histogram":  polarity_hist,
        "top_positive_cities": safe_records(sent_city.nlargest(20, "avg_polarity") if sent_city is not None else None),
        "top_negative_cities": safe_records(sent_city.nsmallest(20, "avg_polarity") if sent_city is not None else None),
        "top_positive_places": safe_records(sent_place.nlargest(20, "avg_polarity") if sent_place is not None else None),
        "treemap_data":        treemap_data,
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: HOTSPOT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/hotspot-data")
def api_hotspot_data():
    top_n = min(int(request.args.get("top_n", 50)), 200)
    return jsonify({
        "top_cities": safe_records(D("hs_city"),  top_n),
        "top_places": safe_records(D("hs_place"), top_n),
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: REAL-TIME TRENDS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/realtime-data")
def api_realtime_data():
    gt = D("gt_df")
    rt = D("rt_df")

    # Status summary from new google_trends.csv (status/source columns)
    live_count     = 0
    fallback_count = 0
    if gt is not None and "status" in gt.columns:
        live_count     = int((gt["status"] == "live").sum())
        fallback_count = int((gt["status"] != "live").sum())

    return jsonify({
        "available":         gt is not None,
        "google_trends":     safe_records(gt),
        "realtime_hotspots": safe_records(rt),
        "live_count":        live_count,
        "fallback_count":    fallback_count,
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/prediction-data")
def api_prediction_data():
    metrics   = D("metrics")
    fi_df     = D("fi_df")
    pred_all  = D("pred_all")
    pred_test = D("pred_test")

    metrics_dict = {}
    if metrics is not None and len(metrics) > 0:
        row = metrics.iloc[0]
        metrics_dict = {
            k: (round(float(v), 4) if pd.notna(v) else None)
            for k, v in row.items()
        }

    top_predicted = []
    if pred_all is not None and "predicted_review_count" in pred_all.columns:
        top_predicted = safe_records(pred_all.nlargest(30, "predicted_review_count"))
    elif pred_test is not None and "predicted_review_count" in pred_test.columns:
        top_predicted = safe_records(pred_test.nlargest(30, "predicted_review_count"))

    # Sample test predictions for scatter chart
    test_scatter = []
    if pred_test is not None:
        samp = pred_test.sample(min(3000, len(pred_test)), random_state=42)
        test_scatter = safe_records(samp)

    return jsonify({
        "metrics":             metrics_dict,
        "feature_importances": safe_records(fi_df),
        "top_predicted":       top_predicted,
        "test_scatter":        test_scatter,
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: INDIA MAP
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/india-map-data")
def api_india_map_data():
    # Prefer pre-built state_metrics.csv for richer data
    state_metrics = D("state_metrics")
    if state_metrics is not None and not state_metrics.empty:
        # Normalise column names to match frontend expectations
        col_map = {
            "ReviewCount":   "total_reviews",
            "AverageRating": "avg_rating",
            "HotspotScore":  "avg_hotspot_score",
            "SentimentScore":"avg_polarity",
            "NumCities":     "num_cities",
        }
        agg = state_metrics.rename(columns=col_map)
        if "avg_polarity" not in agg.columns:
            agg["avg_polarity"] = 0.0
        return jsonify({
            "states":        safe_records(agg),
            "mapped_cities": int(agg["num_cities"].sum()) if "num_cities" in agg.columns else 0,
            "total_cities":  1794,
        })

    # Fallback: compute on-the-fly from hotspots_city.csv
    hs_city = D("hs_city")
    if hs_city is None:
        return jsonify({
            "error": "hotspots_city.csv not found — run hotspot.py first.",
            "states": [],
        })

    df = hs_city.copy()
    df["State"] = df["City"].map(CITY_TO_STATE)
    mapped = df.dropna(subset=["State"]).copy()
    if mapped.empty:
        return jsonify({"error": "No cities mapped to states.", "states": []})

    agg_cols: dict = {
        "total_reviews":     ("review_count",  "sum"),
        "avg_rating":        ("avg_rating",    "mean"),
        "avg_hotspot_score": ("hotspot_score", "mean"),
        "num_cities":        ("City",          "count"),
    }
    if "avg_polarity" in mapped.columns:
        agg_cols["avg_polarity"] = ("avg_polarity", "mean")

    agg = mapped.groupby("State").agg(**agg_cols).reset_index()
    if "avg_polarity" not in agg.columns:
        agg["avg_polarity"] = 0.0
    for col in ("avg_rating", "avg_polarity", "avg_hotspot_score"):
        if col in agg.columns:
            agg[col] = agg[col].round(4)

    return jsonify({
        "states":        safe_records(agg),
        "mapped_cities": int(len(mapped)),
        "total_cities":  int(len(df)),
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: SPARK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/spark-data")
def api_spark_data():
    summary  = D("spark_summary")
    cities   = D("spark_cities")
    places   = D("spark_places")
    rating   = D("spark_rating")
    city_m   = D("spark_city_m")
    state_m  = D("spark_state")

    available = any(x is not None for x in [summary, cities, places])

    # Extract summary row as dict
    summary_dict = {}
    if summary is not None and len(summary) > 0:
        row = summary.iloc[0]
        summary_dict = {
            k: (int(v) if pd.notna(v) and str(v).replace('.','').isdigit() else
                round(float(v), 4) if pd.notna(v) else None)
            for k, v in row.items()
        }

    return jsonify({
        "available":          available,
        "summary":            summary_dict,
        "top_cities":         safe_records(cities),
        "top_places":         safe_records(places),
        "rating_distribution":safe_records(rating),
        "city_metrics":       safe_records(city_m, 50),
        "state_metrics":      safe_records(state_m),
        "techniques": [
            "SparkSession with tuned configuration",
            "Explicit schema definition (StructType/StructField)",
            "DataFrame API (select, filter, withColumn)",
            "cache() / persist() for iterative computation",
            "groupBy() aggregations (count, avg, stddev, approxCountDistinct)",
            "join() operations (inner and left outer)",
            "Spark SQL (createOrReplaceTempView + spark.sql())",
            "Window functions (rank, dense_rank, row_number, percent_rank)",
            "User-Defined Functions (UDF) for city-to-state mapping",
            "Sorting, filtering, and limit()",
        ],
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: PROCESSING LOAD
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/processing-load")
def api_processing_load():
    proc = D("proc_load")
    available = proc is not None

    pandas_data = []
    spark_data  = []
    interpretation = ""

    if proc is not None:
        pandas_rows = proc[proc["method"] == "Pandas"].sort_values("sample_size")
        spark_rows  = proc[proc["method"] == "Spark"].sort_values("sample_size")
        pandas_data = safe_records(pandas_rows[pandas_rows["sample_size"] > 0])
        spark_data  = safe_records(spark_rows[spark_rows["sample_size"] > 0])

        # Auto-generate interpretation
        spark_start = proc[(proc["method"] == "Spark") & (proc["sample_size"] == 0)]
        startup_s = float(spark_start["elapsed_seconds"].iloc[0]) if len(spark_start) > 0 else 0

        interpretation = (
            f"Spark session startup: {startup_s:.1f}s (one-time overhead). "
            "Pandas is faster for small-to-medium datasets on a single machine. "
            "Spark's advantage materialises at scale and in multi-node cluster environments "
            "where parallelism eliminates the startup overhead per job."
        )

    return jsonify({
        "available":      available,
        "pandas":         pandas_data,
        "spark":          spark_data,
        "interpretation": interpretation,
        "chart_paths": {
            "processing_time": "/charts/processing_time_vs_size.png",
            "throughput":      "/charts/throughput_vs_size.png",
        },
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: SIMULATOR  (Feature 1 — Hotspot Weight Simulator)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/simulator-data")
def api_simulator_data():
    """
    Return raw hotspot data for client-side weight simulation.
    The client computes custom_score using slider weights.
    """
    hs_place = D("hs_place")
    hs_city  = D("hs_city")

    if hs_place is None and hs_city is None:
        return jsonify({"error": "Hotspot CSVs not found. Run hotspot.py first.",
                        "places": [], "cities": []})

    # Columns needed: name-key, rating_score, volume_score, sentiment_score,
    #                 hotspot_score, avg_rating, review_count, avg_polarity
    def prep(df, name_col):
        if df is None:
            return []
        cols = [name_col, "rating_score", "volume_score", "sentiment_score",
                "hotspot_score", "avg_rating", "review_count", "avg_polarity"]
        available = [c for c in cols if c in df.columns]
        return safe_records(df[available])

    return jsonify({
        "places": prep(hs_place, "Place"),
        "cities": prep(hs_city, "City"),
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: COMPARE DATA  (Feature 2 — Destination Compare Tool)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/compare-data")
def api_compare_data():
    """
    Return hotspot data enriched with sentiment for the compare tool.
    Query params: mode=places|cities
    """
    mode = request.args.get("mode", "places")  # 'places' or 'cities'

    if mode == "cities":
        df       = D("hs_city")
        name_col = "City"
        sent_df  = D("sent_city")
        sent_key = "City"
    else:
        df       = D("hs_place")
        name_col = "Place"
        sent_df  = D("sent_place")
        sent_key = "Place"

    if df is None:
        return jsonify({"error": f"hotspots_{mode.rstrip('s')}.csv not found.",
                        "data": [], "names": []})

    result = df.copy()

    # Merge extra sentiment cols if available and not already present
    if sent_df is not None and sent_key in sent_df.columns:
        extra_cols = [c for c in sent_df.columns
                      if c not in result.columns and c != sent_key]
        if extra_cols:
            result = result.merge(
                sent_df[[sent_key] + extra_cols],
                on=sent_key, how="left"
            )

    cols_want = [name_col, "avg_rating", "review_count", "avg_polarity",
                 "hotspot_score", "rating_score", "volume_score",
                 "sentiment_score", "rating_std"]
    available = [c for c in cols_want if c in result.columns]

    records  = safe_records(result[available])
    names    = result[name_col].dropna().unique().tolist()

    return jsonify({"data": records, "names": sorted(names)})


# ══════════════════════════════════════════════════════════════════════════════
# API: SIMILAR PLACES  (Feature 3 — Similar Destination Recommender)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/similar-places")
def api_similar_places():
    """
    Given a seed place name, return top-N most similar places.
    Similarity = 1 / (1 + Euclidean distance) on normalised features.
    Query params: seed=<place_name>, top_n=<int>
    """
    seed  = request.args.get("seed", "").strip()
    top_n = min(int(request.args.get("top_n", 10)), 50)

    hs_place = D("hs_place")
    if hs_place is None:
        return jsonify({"error": "hotspots_place.csv not found.", "results": []})

    feature_cols = [c for c in
                    ["avg_rating", "review_count", "avg_polarity",
                     "hotspot_score", "rating_std"]
                    if c in hs_place.columns]

    df = hs_place.copy().dropna(subset=feature_cols).reset_index(drop=True)

    # Normalise features to [0, 1]
    feat = df[feature_cols].copy()
    import math
    if "review_count" in feat.columns:
        feat["review_count"] = feat["review_count"].apply(
            lambda x: math.log1p(max(0, x))
        )
    for col in feat.columns:
        col_min = feat[col].min()
        col_max = feat[col].max()
        rng = col_max - col_min
        feat[col] = (feat[col] - col_min) / rng if rng > 0 else 0.0

    feat_arr = feat.values  # numpy array

    # Find seed row (case-insensitive partial match)
    mask = df["Place"].str.lower() == seed.lower()
    if not mask.any():
        mask = df["Place"].str.lower().str.contains(seed.lower(), na=False)
    if not mask.any():
        return jsonify({"error": f"Place '{seed}' not found.", "results": []})

    seed_idx = int(mask.idxmax())
    seed_vec = feat_arr[seed_idx]

    # Compute Euclidean distances from seed
    diffs = feat_arr - seed_vec          # (N, D)
    dists = np.sqrt((diffs ** 2).sum(axis=1))  # (N,)
    df["_distance"] = dists
    df["similarity"] = 1.0 / (1.0 + dists)

    # Exclude the seed itself and return top_n
    similar = (
        df[df.index != seed_idx]
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )

    cols_out = ["Place", "similarity", "avg_rating", "review_count",
                "avg_polarity", "hotspot_score"]
    available = [c for c in cols_out if c in similar.columns]
    similar["similarity"] = similar["similarity"].round(4)

    return jsonify({
        "seed":    df.loc[seed_idx, "Place"],
        "results": safe_records(similar[available]),
    })


# ══════════════════════════════════════════════════════════════════════════════
# API: STATE DETAIL  (Feature 5 — India Map Drill-Down)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/state-detail")
def api_state_detail():
    """
    Return top cities and top places for a given state.
    Query params: state=<state_name>, top_n=<int>
    Uses CITY_TO_STATE mapping + hotspots_city.csv + hotspots_place.csv
    + place_city_mapping.csv.
    """
    state_name = request.args.get("state", "").strip()
    top_n      = min(int(request.args.get("top_n", 10)), 30)

    if not state_name:
        return jsonify({"error": "state parameter required.", "cities": [], "places": []})

    # --- Cities in this state ---
    hs_city = D("hs_city")
    top_cities = []
    state_cities_set: set[str] = set()

    if hs_city is not None:
        # Build reverse mapping: city → state
        city_df = hs_city.copy()
        city_df["_state"] = city_df["City"].map(CITY_TO_STATE)
        in_state = city_df[city_df["_state"] == state_name].copy()
        state_cities_set = set(in_state["City"].dropna().unique())
        in_state_sorted = in_state.sort_values("hotspot_score", ascending=False).head(top_n)
        cols = [c for c in ["City", "avg_rating", "review_count", "avg_polarity",
                             "hotspot_score", "rating_score", "volume_score"] if c in in_state_sorted.columns]
        top_cities = safe_records(in_state_sorted[cols])

    # --- Places in this state (via place→city→state) ---
    hs_place    = D("hs_place")
    pc_map      = _load("place_city_mapping.csv")  # Place, primary_city
    top_places  = []

    if hs_place is not None and state_cities_set:
        place_df = hs_place.copy()
        # Attach city to place via mapping
        if pc_map is not None and "Place" in pc_map.columns and "primary_city" in pc_map.columns:
            place_df = place_df.merge(pc_map[["Place", "primary_city"]], on="Place", how="left")
            # Keep places whose primary_city is in this state
            place_df = place_df[place_df["primary_city"].isin(state_cities_set)]
        else:
            # Fallback: can't filter — skip places
            place_df = place_df.iloc[0:0]

        in_state_places = place_df.sort_values("hotspot_score", ascending=False).head(top_n)
        pcols = [c for c in ["Place", "avg_rating", "review_count", "avg_polarity",
                              "hotspot_score"] if c in in_state_places.columns]
        top_places = safe_records(in_state_places[pcols])

    # --- Aggregate stats for the state ---
    total_reviews = sum(c.get("review_count", 0) or 0 for c in top_cities)
    avg_rating    = (
        round(float(np.mean([c["avg_rating"] for c in top_cities if c.get("avg_rating")])), 3)
        if top_cities else None
    )
    avg_polarity  = (
        round(float(np.mean([c["avg_polarity"] for c in top_cities if c.get("avg_polarity")])), 4)
        if top_cities else None
    )
    avg_hotspot   = (
        round(float(np.mean([c["hotspot_score"] for c in top_cities if c.get("hotspot_score")])), 4)
        if top_cities else None
    )

    return jsonify({
        "state":        state_name,
        "top_cities":   top_cities,
        "top_places":   top_places,
        "total_reviews": int(total_reviews),
        "avg_rating":    avg_rating,
        "avg_polarity":  avg_polarity,
        "avg_hotspot_score": avg_hotspot,
    })


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW EXPLORER — helpers and constants  (Feature 6)
# ══════════════════════════════════════════════════════════════════════════════

# Aspect keyword sets for mention-count analysis
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "cleanliness": ["clean", "dirty", "hygiene", "hygienic", "filthy", "neat", "tidy",
                    "waste", "garbage", "littered", "spotless"],
    "crowd":       ["crowd", "crowded", "rush", "busy", "packed", "tourists", "queue",
                    "waiting", "jammed", "overrated", "overwhelming"],
    "food":        ["food", "eat", "restaurant", "snack", "meal", "tasty", "delicious",
                    "stall", "dhaba", "cuisine", "drink", "beverage", "biryani",
                    "chai", "coffee"],
    "price":       ["price", "cost", "expensive", "cheap", "affordable", "money",
                    "worth", "value", "ticket", "entry fee", "overpriced", "budget"],
    "safety":      ["safe", "safety", "dangerous", "unsafe", "secure", "security",
                    "theft", "police", "guard", "harassment", "risk"],
    "family":      ["family", "kids", "children", "child", "baby", "parents",
                    "couple", "romantic", "picnic", "outing", "trip"],
    "scenery":     ["beautiful", "scenic", "view", "nature", "landscape", "sunrise",
                    "sunset", "hills", "mountains", "waterfall", "beach", "lake",
                    "river", "greenery", "peaceful", "serene"],
    "transport":   ["transport", "bus", "auto", "cab", "taxi", "parking", "road",
                    "accessible", "connectivity", "drive", "train", "metro",
                    "highway", "route"],
}

_REVIEW_CACHE: dict = {}


def _get_review_sample() -> pd.DataFrame | None:
    """
    Return a 100K-row sample of the review dataset (cached after first load).
    Merges sentiment columns from sent_sample if available.
    Prefers outputs/sentiment_results.csv for polarity; falls back to
    data/Review_db.csv if sentiment file is unavailable.
    """
    if "review_sample" in _REVIEW_CACHE:
        return _REVIEW_CACHE["review_sample"]

    sent_sample = D("sent_sample")   # already loaded: City, Place, Rating, polarity, sentiment

    review_db_path = os.path.join(BASE_DIR, "data", "Review_db.csv")
    if os.path.exists(review_db_path):
        try:
            print("  [ReviewExplorer] Sampling Review_db.csv …")
            chunks = []
            chunk_size = 100_000
            target     = 100_000
            collected  = 0
            for chunk in pd.read_csv(
                review_db_path,
                usecols=["City", "Place", "Raw_Review", "Rating"],
                chunksize=chunk_size,
                dtype={"Rating": "float32"},
                on_bad_lines="skip",
            ):
                chunks.append(chunk)
                collected += len(chunk)
                if collected >= target:
                    break
            df = pd.concat(chunks, ignore_index=True).sample(
                min(target, collected), random_state=42
            )
            df.rename(columns={"Raw_Review": "Review"}, inplace=True)
            df["Review"] = df["Review"].fillna("").astype(str)
            df["City"]   = df["City"].fillna("Unknown").astype(str)
            df["Place"]  = df["Place"].fillna("Unknown").astype(str)
            df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

            # Merge polarity/sentiment from sent_sample where possible
            if sent_sample is not None:
                sent_mini = sent_sample[["City", "Place", "polarity", "sentiment"]].copy()
                sent_agg  = (sent_mini.groupby(["City", "Place"])
                             .agg(polarity=("polarity", "mean"),
                                  sentiment=("sentiment", lambda x: x.mode().iloc[0] if len(x) else "Unknown"))
                             .reset_index())
                df = df.merge(sent_agg, on=["City", "Place"], how="left")
            else:
                df["polarity"]  = float("nan")
                df["sentiment"] = "Unknown"

            print(f"  [ReviewExplorer] Sample ready: {len(df):,} rows.")
            _REVIEW_CACHE["review_sample"] = df

            # Also cache city/place lists
            _REVIEW_CACHE["re_cities"] = sorted(df["City"].dropna().unique().tolist())
            _REVIEW_CACHE["re_places"] = sorted(df["Place"].dropna().unique().tolist())
            return df
        except Exception as e:
            print(f"  [WARN] Could not load Review_db.csv: {e}")

    # Fallback: use the sent_sample already in memory
    if sent_sample is not None:
        _REVIEW_CACHE["review_sample"]  = sent_sample.copy()
        _REVIEW_CACHE["re_cities"] = sorted(sent_sample["City"].dropna().unique().tolist())
        _REVIEW_CACHE["re_places"] = sorted(sent_sample["Place"].dropna().unique().tolist())
        return sent_sample
    return None


# ── Review Explorer page route ────────────────────────────────────────────────
@app.route("/review-explorer")
def page_review_explorer():
    return render_template("review_explorer.html", active="review_explorer")


# ── API: Review meta (city/place lists for dropdowns) ────────────────────────
@app.route("/api/review-meta")
def api_review_meta():
    """Return lists of available cities and places for filter dropdowns."""
    _get_review_sample()   # ensure cache is warm
    return jsonify({
        "cities": _REVIEW_CACHE.get("re_cities", []),
        "places": _REVIEW_CACHE.get("re_places", []),
    })


# ── API: Review search ────────────────────────────────────────────────────────
@app.route("/api/review-search")
def api_review_search():
    """
    Filter the review sample and return matching reviews + aspect analysis.
    Query params:
      city=<str>       filter by city (case-insensitive)
      place=<str>      filter by place (case-insensitive)
      sentiment=<str>  Positive|Negative|Neutral
      rating=<float>   minimum rating
      keyword=<str>    keyword to search in review text
      limit=<int>      max reviews to return (default 50, max 200)
    """
    city      = request.args.get("city",      "").strip()
    place     = request.args.get("place",     "").strip()
    sentiment = request.args.get("sentiment", "").strip()
    rating    = request.args.get("rating",    "")
    keyword   = request.args.get("keyword",   "").strip()
    limit     = min(int(request.args.get("limit", 50)), 200)

    df = _get_review_sample()
    if df is None or df.empty:
        return jsonify({"error": "Review data not available.", "reviews": [],
                        "summary": {}, "aspects": {}})

    mask = pd.Series([True] * len(df), index=df.index)

    if city:
        mask &= df["City"].str.lower() == city.lower()

    if place:
        mask &= df["Place"].str.lower() == place.lower()

    if sentiment and "sentiment" in df.columns:
        mask &= df["sentiment"].str.lower() == sentiment.lower()

    if rating:
        try:
            r_val = float(rating)
            mask &= df["Rating"] >= r_val
        except ValueError:
            pass

    filtered = df[mask].copy()

    if keyword:
        kw_lower = keyword.lower()
        kw_mask  = filtered["Review"].str.lower().str.contains(kw_lower, na=False, regex=False)
        filtered = filtered[kw_mask]

    # ── Summary stats ──────────────────────────────────────────────────────────
    n_total = len(filtered)
    avg_r   = round(float(filtered["Rating"].dropna().mean()), 2) if n_total else None
    avg_p   = None
    if "polarity" in filtered.columns:
        pol = filtered["polarity"].dropna()
        if len(pol):
            avg_p = round(float(pol.mean()), 4)

    # Top aspect
    aspect_counts: dict[str, int] = {}
    aspect_ratings: dict[str, list[float]] = {a: [] for a in ASPECT_KEYWORDS}

    review_texts = filtered["Review"].fillna("").str.lower().tolist()
    ratings_list = filtered["Rating"].tolist()

    for text, rat in zip(review_texts, ratings_list):
        for aspect, kws in ASPECT_KEYWORDS.items():
            for kw in kws:
                if kw in text:
                    aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
                    if not np.isnan(rat) if isinstance(rat, float) else True:
                        try:
                            aspect_ratings[aspect].append(float(rat))
                        except (TypeError, ValueError):
                            pass
                    break  # count each aspect once per review

    aspects_out = {}
    for asp in ASPECT_KEYWORDS:
        cnt = aspect_counts.get(asp, 0)
        rat_list = aspect_ratings.get(asp, [])
        aspects_out[asp] = {
            "mentions": cnt,
            "avg_rating": round(float(np.mean(rat_list)), 2) if rat_list else None,
        }

    # Top aspect by mentions
    top_aspect = max(aspects_out, key=lambda a: aspects_out[a]["mentions"]) if aspects_out else None

    # ── Reviews to return ──────────────────────────────────────────────────────
    sample_out = filtered.head(limit).copy()
    # Truncate review text for frontend
    sample_out["Review"] = sample_out["Review"].str.slice(0, 400)
    sample_out = sample_out.replace([np.inf, -np.inf], np.nan)

    cols_out = [c for c in ["City", "Place", "Review", "Rating", "polarity", "sentiment"]
                if c in sample_out.columns]

    return jsonify({
        "total_matched": n_total,
        "reviews":       json.loads(sample_out[cols_out].to_json(orient="records")),
        "summary": {
            "count":       n_total,
            "avg_rating":  avg_r,
            "avg_polarity": avg_p,
            "top_aspect":  top_aspect,
        },
        "aspects": aspects_out,
    })


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
_init_cache()


if __name__ == "__main__":
    print("=" * 62)
    print("  Tourism Trend Analytics — Flask Application")
    print("=" * 62)
    print("  URL: http://localhost:5000")
    print("=" * 62)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
