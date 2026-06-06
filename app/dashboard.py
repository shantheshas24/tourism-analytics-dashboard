"""
dashboard.py
============
Streamlit Dashboard — Tourism Trend Analytics & Hotspot Prediction.

Features:
  ┌─────────────────────────────────────────────────────┐
  │  1. Dataset Overview                                │
  │  2. Top Cities (bar charts, table)                  │
  │  3. Top Places (bar charts, table)                  │
  │  4. Sentiment Analysis Charts                       │
  │  5. Hotspot Rankings (scored leaderboard)           │
  │  6. Prediction Results (feature importance + chart) │
  │  7. Interactive Folium Map                          │
  └─────────────────────────────────────────────────────┘

Save to : BDT_project/app/dashboard.py
Run with: streamlit run app/dashboard.py
          (from the BDT_project root directory)

IMPORTANT: Run all four pipeline scripts first:
  python scripts/analytics.py
  python scripts/sentiment.py
  python scripts/hotspot.py
  python scripts/prediction.py
"""

import os
import sys
import warnings
import json
import re

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Trend Analytics | BDT Project",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(12px);
    }
    [data-testid="stSidebar"] * { color: #d0d0e8 !important; }

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(100,80,255,0.3);
    }
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #9090b8 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Section headers ── */
    h1 { color: #c4b5fd !important; font-weight: 700 !important; }
    h2 { color: #a78bfa !important; font-weight: 600 !important; }
    h3 { color: #8b5cf6 !important; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9090b8 !important;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6d28d9, #4c1d95) !important;
        color: #fff !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* ── Section divider ── */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #7c3aed, transparent);
        margin: 2rem 0;
        border-radius: 2px;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(109,40,217,0.3), rgba(76,29,149,0.2));
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(12px);
        text-align: center;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        color: #9090b8;
        font-size: 1.05rem;
        letter-spacing: 0.03em;
    }

    /* ── Rank badge ── */
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #4338ca);
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        line-height: 28px;
        text-align: center;
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* ── Score bar ── */
    .score-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
    }
    .score-bar-fill {
        background: linear-gradient(90deg, #7c3aed, #a78bfa);
        height: 100%;
        border-radius: 8px;
    }

    /* ── Info box ── */
    .info-box {
        background: rgba(99,102,241,0.12);
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #c7d2fe;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Load CSVs with caching
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame | None:
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_sample_reviews(n: int = 5000) -> pd.DataFrame | None:
    """Load a small sample of the raw dataset for the overview tab."""
    data_path = os.path.join(BASE_DIR, "data", "Review_db.csv")
    if not os.path.exists(data_path):
        return None
    try:
        df = pd.read_csv(
            data_path,
            usecols=["City", "Place", "Rating"],
            dtype={"City": "category", "Place": "category", "Rating": "float32"},
            nrows=n,
            encoding="utf-8",
            on_bad_lines="skip",
        )
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_TEMPLATE = "plotly_dark"
PALETTE = px.colors.sequential.Purpor
PALETTE2 = px.colors.qualitative.Pastel


def apply_chart_style(fig, title: str = "", height: int = 420) -> go.Figure:
    """Apply consistent dark glass styling to all Plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#c4b5fd"), x=0.02),
        height=height,
        paper_bgcolor="rgba(255,255,255,0.03)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(family="Inter", color="#c0bce8"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        margin=dict(l=40, r=30, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    return fig


def minmax(series: pd.Series) -> pd.Series:
    """Return a numeric series scaled to [0, 1]."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    mn, mx = values.min(), values.max()
    if mx == mn:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mn) / (mx - mn)


def log_volume_score(series: pd.Series) -> pd.Series:
    """Normalise review volume after log scaling."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0)
    return minmax(np.log1p(values))


@st.cache_data(show_spinner=False)
def load_review_text_sample(n: int = 60000) -> pd.DataFrame | None:
    """Load a dashboard-sized sample of raw review text for aspect analysis."""
    data_path = os.path.join(BASE_DIR, "data", "Review_db.csv")
    if not os.path.exists(data_path):
        return None
    try:
        df = pd.read_csv(
            data_path,
            usecols=["City", "Place", "Review", "Raw_Review", "Rating"],
            dtype={"City": "object", "Place": "object", "Review": "object", "Raw_Review": "object", "Rating": "float32"},
            nrows=n,
            encoding="utf-8",
            on_bad_lines="skip",
        )
        df["review_text"] = df["Review"].fillna("").astype(str)
        fallback = df["Raw_Review"].fillna("").astype(str)
        df.loc[df["review_text"].str.strip() == "", "review_text"] = fallback
        df = df.drop(columns=["Review", "Raw_Review"], errors="ignore")
        df = df[df["review_text"].str.strip() != ""].copy()
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").clip(1, 5)
        return df.dropna(subset=["Rating"])
    except Exception:
        return None


ASPECT_KEYWORDS = {
    "Cleanliness": ["clean", "hygiene", "dirty", "maintained", "washroom", "toilet"],
    "Crowd": ["crowd", "queue", "rush", "busy", "packed", "waiting", "line"],
    "Food": ["food", "restaurant", "snack", "cafe", "meal", "taste", "dining"],
    "Price": ["price", "ticket", "cost", "expensive", "cheap", "value", "worth"],
    "Safety": ["safe", "security", "police", "danger", "risky", "guard"],
    "Family": ["family", "kids", "children", "child", "elderly", "parents"],
    "Scenery": ["view", "scenic", "beautiful", "sunset", "nature", "photo", "landscape"],
    "Transport": ["parking", "metro", "bus", "taxi", "traffic", "road", "access"],
}


def build_aspect_summary(reviews: pd.DataFrame) -> pd.DataFrame:
    """Create aspect-level quality signals from sampled review text."""
    if reviews is None or reviews.empty:
        return pd.DataFrame()

    rows = []
    text = reviews["review_text"].str.lower().fillna("")
    for aspect, keywords in ASPECT_KEYWORDS.items():
        pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
        mask = text.str.contains(pattern, regex=True, na=False)
        matched = reviews.loc[mask]
        if matched.empty:
            continue
        rows.append({
            "Aspect": aspect,
            "Mentions": int(len(matched)),
            "Avg Rating": float(matched["Rating"].mean()),
            "Positive Share": float((matched["Rating"] >= 4).mean()),
            "Negative Share": float((matched["Rating"] <= 2).mean()),
            "Signal": float((matched["Rating"].mean() - 3.0) / 2.0),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Signal", "Mentions"], ascending=[False, False])


@st.cache_data(show_spinner=False)
def prepare_place_insights() -> pd.DataFrame | None:
    """Merge place-level outputs and add hidden-gem/risk/recommendation scores."""
    hs_place = load_csv("hotspots_place.csv")
    if hs_place is None:
        return None

    df = hs_place.copy()
    df["avg_rating"] = pd.to_numeric(df.get("avg_rating"), errors="coerce")
    df["review_count"] = pd.to_numeric(df.get("review_count"), errors="coerce").fillna(0)
    df["rating_std"] = pd.to_numeric(df.get("rating_std"), errors="coerce").fillna(0)
    df["avg_polarity"] = pd.to_numeric(df.get("avg_polarity"), errors="coerce").fillna(0)

    top_places = load_csv("top_places.csv")
    if top_places is not None and "city" in top_places.columns:
        df = df.merge(top_places[["Place", "city"]], on="Place", how="left")
    if "City" not in df.columns:
        df["City"] = df.get("city", "Unknown")
    df["City"] = df["City"].fillna("Unknown")

    sent_place = load_csv("sentiment_by_place.csv")
    if sent_place is not None:
        extra_cols = [c for c in ["Place", "avg_subjectivity", "dominant_sentiment"] if c in sent_place.columns]
        if len(extra_cols) > 1:
            df = df.merge(sent_place[extra_cols], on="Place", how="left")

    pred_all = load_csv("all_place_predictions.csv")
    if pred_all is not None:
        pred_cols = [c for c in ["Place", "predicted_review_count", "popularity_rank"] if c in pred_all.columns]
        if len(pred_cols) > 1:
            df = df.merge(pred_all[pred_cols], on="Place", how="left")

    rating_signal = pd.to_numeric(df.get("rating_score", minmax(df["avg_rating"])), errors="coerce").fillna(0)
    volume_signal = pd.to_numeric(df.get("volume_score", log_volume_score(df["review_count"])), errors="coerce").fillna(0)
    sentiment_signal = pd.to_numeric(
        df.get("sentiment_score", minmax((df["avg_polarity"] + 1.0) / 2.0)),
        errors="coerce",
    ).fillna(0)
    consistency_signal = 1.0 - minmax(df["rating_std"])
    low_volume_signal = 1.0 - volume_signal

    df["hidden_gem_score"] = (
        0.36 * rating_signal +
        0.28 * sentiment_signal +
        0.20 * consistency_signal +
        0.16 * low_volume_signal
    ).round(6)
    df["risk_score"] = (
        0.35 * volume_signal +
        0.25 * (1.0 - rating_signal) +
        0.25 * (1.0 - sentiment_signal) +
        0.15 * minmax(df["rating_std"])
    ).round(6)
    df["confidence_score"] = volume_signal.round(6)
    return df


@st.cache_data(show_spinner=False)
def prepare_city_insights() -> pd.DataFrame | None:
    """Merge city-level outputs for comparison, maps, and weighted ranking."""
    hs_city = load_csv("hotspots_city.csv")
    if hs_city is None:
        return None
    df = hs_city.copy()
    for col in ["avg_rating", "review_count", "rating_std", "avg_polarity", "hotspot_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def recompute_hotspot_score(df: pd.DataFrame, rating_w: float, volume_w: float, sentiment_w: float) -> pd.DataFrame:
    """Recompute hotspot score from user-selected weights."""
    out = df.copy()
    total = rating_w + volume_w + sentiment_w
    if total <= 0:
        rating_w, volume_w, sentiment_w, total = 0.5, 0.3, 0.2, 1.0

    out["rating_score_live"] = pd.to_numeric(out.get("rating_score", minmax(out["avg_rating"])), errors="coerce").fillna(0)
    out["volume_score_live"] = pd.to_numeric(out.get("volume_score", log_volume_score(out["review_count"])), errors="coerce").fillna(0)
    out["sentiment_score_live"] = pd.to_numeric(
        out.get("sentiment_score", minmax((out["avg_polarity"] + 1.0) / 2.0)),
        errors="coerce",
    ).fillna(0)
    out["custom_hotspot_score"] = (
        (rating_w / total) * out["rating_score_live"] +
        (volume_w / total) * out["volume_score_live"] +
        (sentiment_w / total) * out["sentiment_score_live"]
    ).round(6)
    return out.sort_values("custom_hotspot_score", ascending=False).reset_index(drop=True)


def destination_catalog(view: str) -> pd.DataFrame | None:
    """Return city or place insights with a unified Destination column."""
    if view == "Cities":
        df = prepare_city_insights()
        name_col = "City"
    else:
        df = prepare_place_insights()
        name_col = "Place"
    if df is None:
        return None
    out = df.copy()
    out["Destination"] = out[name_col].astype(str)
    out["Type"] = view[:-1]
    return out


def recommend_similar_places(places: pd.DataFrame, selected_place: str, top_n: int = 10) -> pd.DataFrame:
    """Find places with similar score, rating, sentiment, volume, and consistency."""
    if places is None or places.empty or selected_place not in places["Place"].astype(str).values:
        return pd.DataFrame()

    df = places.copy()
    df["Place"] = df["Place"].astype(str)
    feature_frame = pd.DataFrame({
        "rating": minmax(df["avg_rating"]),
        "volume": log_volume_score(df["review_count"]),
        "sentiment": minmax((df["avg_polarity"] + 1.0) / 2.0),
        "consistency": 1.0 - minmax(df["rating_std"]),
        "hotspot": minmax(df["hotspot_score"]),
    })
    selected_idx = df.index[df["Place"] == selected_place][0]
    selected_vector = feature_frame.loc[selected_idx].values
    distances = np.linalg.norm(feature_frame.values - selected_vector, axis=1)
    df["similarity"] = (1.0 / (1.0 + distances)).round(4)
    return (
        df[df["Place"] != selected_place]
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌍 Navigation")
        st.markdown("---")
        st.markdown("### 📁 Project")
        st.markdown("**Tourism Trend Analytics**")
        st.markdown("*Big Data Technology*")
        st.markdown("---")
        st.markdown("### ℹ️ Dataset")
        st.markdown("- **Rows:** 1,482,466")
        st.markdown("- **Cities:** 1,794")
        st.markdown("- **Places:** 14,494")
        st.markdown("- **Ratings:** 1–5 ★")
        st.markdown("---")
        st.markdown("### ⚙️ Hotspot Weights")
        st.markdown("- Rating Score: **50%**")
        st.markdown("- Review Volume: **30%**")
        st.markdown("- Sentiment: **20%**")
        st.markdown("---")
        st.markdown("### 📦 Tech Stack")
        badges = ["Python", "PySpark", "Pandas", "Scikit-learn",
                  "TextBlob", "Streamlit", "Plotly", "Folium"]
        for b in badges:
            st.markdown(f"`{b}`", unsafe_allow_html=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def tab_overview():
    st.markdown("## 📊 Dataset Overview")
    st.markdown('<div class="info-box">This section provides a high-level statistical overview of the <strong>Review_db.csv</strong> dataset containing 1.48 million tourism reviews across 1,794 cities and 14,494 places.</div>', unsafe_allow_html=True)

    rating_dist = load_csv("rating_distribution.csv")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews",   "1,482,466")
    c2.metric("Unique Cities",   "1,794")
    c3.metric("Unique Places",   "14,494")
    c4.metric("Rating Range",    "1 – 5 ★")

    if rating_dist is not None:
        avg = round((rating_dist["rating"] * rating_dist["count"]).sum() /
                    rating_dist["count"].sum(), 2)
        c5.metric("Avg Rating", f"⭐ {avg}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        if rating_dist is not None:
            fig = px.bar(
                rating_dist, x="rating", y="count",
                color="count",
                color_continuous_scale=PALETTE,
                labels={"rating": "Star Rating", "count": "Number of Reviews"},
                template=PLOTLY_TEMPLATE,
            )
            fig = apply_chart_style(fig, "📈 Rating Distribution", 380)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            # Pie chart
            fig2 = px.pie(
                rating_dist, names="rating", values="count",
                color_discrete_sequence=px.colors.sequential.Purpor,
                hole=0.55,
                template=PLOTLY_TEMPLATE,
            )
            fig2 = apply_chart_style(fig2, "📊 Rating Share", 350)
            fig2.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("rating_distribution.csv not found. Run analytics.py first.")

    with col2:
        st.markdown("### 📋 Dataset Schema")
        schema = pd.DataFrame({
            "Column":   ["City", "Place", "Review", "Rating", "Name", "Date", "Raw_Review"],
            "Type":     ["string", "string", "text", "float", "string", "datetime*", "text"],
            "Unique":   ["1,794", "14,494", "~1.4M", "1–5", "~800K", "NULL ⚠️", "~1.4M"],
            "Nulls":    ["few", "few", "some", "few", "few", "ALL ❌", "some"],
        })
        st.dataframe(schema, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📝 Data Notes")
        st.markdown("""
        - **Date** column is completely null — no temporal analysis possible.
        - **Review** and **Raw_Review** both contain user text; the pipeline prefers `Review`.
        - **Rating** values clipped to valid range [1, 5].
        - Large dataset handled via **chunked reading** (200K rows/chunk).
        """)

        st.markdown("---")
        st.markdown("### 🔧 Pipeline Execution Order")
        for i, (step, desc) in enumerate([
            ("analytics.py",  "Load & aggregate data"),
            ("sentiment.py",  "TextBlob sentiment analysis"),
            ("hotspot.py",    "Compute hotspot scores"),
            ("prediction.py", "Train Random Forest model"),
            ("dashboard.py",  "Launch this dashboard"),
        ], 1):
            st.markdown(f"**Step {i}** — `{step}` → {desc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TOP CITIES
# ══════════════════════════════════════════════════════════════════════════════
def tab_top_cities():
    st.markdown("## 🏙️ Top Cities")
    st.markdown('<div class="info-box">Cities ranked by total review volume and average star rating.</div>', unsafe_allow_html=True)

    top_cities = load_csv("top_cities.csv")
    if top_cities is None:
        st.error("top_cities.csv not found. Please run analytics.py first.")
        return

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        top_n = st.slider("Show top N cities", 5, min(50, len(top_cities)), 15, key="city_n")
    df_show = top_cities.head(top_n)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_show.sort_values("review_count"),
            x="review_count", y="City",
            orientation="h",
            color="review_count",
            color_continuous_scale=PALETTE,
            labels={"review_count": "Review Count", "City": ""},
            template=PLOTLY_TEMPLATE,
        )
        fig = apply_chart_style(fig, "📊 Review Volume by City", 500)
        fig.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df_show.sort_values("avg_rating"),
            x="avg_rating", y="City",
            orientation="h",
            color="avg_rating",
            color_continuous_scale="Viridis",
            labels={"avg_rating": "Avg Rating ★", "City": ""},
            range_x=[1, 5],
            template=PLOTLY_TEMPLATE,
        )
        fig2 = apply_chart_style(fig2, "⭐ Average Rating by City", 500)
        fig2.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=11)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: Volume vs Rating
    fig3 = px.scatter(
        top_cities.head(100),
        x="review_count", y="avg_rating",
        size="unique_places",
        color="avg_rating",
        hover_name="City",
        color_continuous_scale="Purples",
        labels={"review_count": "Review Count", "avg_rating": "Avg Rating ★",
                "unique_places": "Unique Places"},
        template=PLOTLY_TEMPLATE,
        size_max=40,
    )
    fig3 = apply_chart_style(fig3, "🔵 Review Volume vs Average Rating (Top 100 Cities)", 420)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 📋 City Data Table")
    st.dataframe(
        top_cities.head(top_n).style.format({
            "review_count":   "{:,}",
            "avg_rating":     "{:.3f}",
            "unique_places":  "{:,}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: TOP PLACES
# ══════════════════════════════════════════════════════════════════════════════
def tab_top_places():
    st.markdown("## 📍 Top Places")
    st.markdown('<div class="info-box">Individual tourist places/attractions ranked by review volume and quality.</div>', unsafe_allow_html=True)

    top_places = load_csv("top_places.csv")
    if top_places is None:
        st.error("top_places.csv not found. Please run analytics.py first.")
        return

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        top_n = st.slider("Show top N places", 5, min(50, len(top_places)), 15, key="place_n")
    df_show = top_places.head(top_n)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_show.sort_values("review_count"),
            x="review_count", y="Place",
            orientation="h",
            color="review_count",
            color_continuous_scale=PALETTE,
            labels={"review_count": "Review Count", "Place": ""},
            template=PLOTLY_TEMPLATE,
        )
        fig = apply_chart_style(fig, "📊 Most-Reviewed Places", 520)
        fig.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=9)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df_show.sort_values("avg_rating"),
            x="avg_rating", y="Place",
            orientation="h",
            color="avg_rating",
            color_continuous_scale="Teal",
            labels={"avg_rating": "Avg Rating ★", "Place": ""},
            range_x=[1, 5],
            template=PLOTLY_TEMPLATE,
        )
        fig2 = apply_chart_style(fig2, "⭐ Highest-Rated Places", 520)
        fig2.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=9)
        st.plotly_chart(fig2, use_container_width=True)

    # Treemap by City → Place → review_count
    if "city" in top_places.columns:
        fig3 = px.treemap(
            top_places.head(80),
            path=["city", "Place"],
            values="review_count",
            color="avg_rating",
            color_continuous_scale="Purples",
            template=PLOTLY_TEMPLATE,
        )
        fig3 = apply_chart_style(fig3, "🗺️ City → Place Treemap (Top 80 Places)", 480)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 📋 Place Data Table")
    st.dataframe(
        top_places.head(top_n).style.format({
            "review_count": "{:,}",
            "avg_rating":   "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def tab_sentiment():
    st.markdown("## 💬 Sentiment Analysis")
    st.markdown('<div class="info-box">TextBlob lexicon-based sentiment scores computed on 1.48M reviews. Polarity ∈ [-1, 1]; Subjectivity ∈ [0, 1].</div>', unsafe_allow_html=True)

    sent_results = load_csv("sentiment_results.csv")
    sent_city    = load_csv("sentiment_by_city.csv")
    sent_place   = load_csv("sentiment_by_place.csv")

    if sent_results is None:
        st.error("sentiment_results.csv not found. Please run sentiment.py first.")
        return

    # ── Overall label distribution
    col1, col2 = st.columns([2, 1])
    with col1:
        label_counts = sent_results["sentiment"].value_counts().reset_index()
        label_counts.columns = ["Sentiment", "Count"]
        colors = {"Positive": "#34d399", "Neutral": "#60a5fa", "Negative": "#f87171"}
        fig = px.pie(
            label_counts, names="Sentiment", values="Count",
            color="Sentiment",
            color_discrete_map=colors,
            hole=0.55,
            template=PLOTLY_TEMPLATE,
        )
        fig = apply_chart_style(fig, "🧠 Overall Sentiment Distribution", 380)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Sentiment Summary")
        for _, row in label_counts.iterrows():
            pct = row["Count"] / label_counts["Count"].sum() * 100
            emoji = "😊" if row["Sentiment"] == "Positive" else ("😐" if row["Sentiment"] == "Neutral" else "😞")
            st.metric(f"{emoji} {row['Sentiment']}", f"{row['Count']:,}", f"{pct:.1f}%")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Polarity distribution histogram
    sample = sent_results.sample(min(50000, len(sent_results)), random_state=42)
    fig2 = px.histogram(
        sample, x="polarity", nbins=60,
        color_discrete_sequence=["#8b5cf6"],
        labels={"polarity": "Polarity Score"},
        template=PLOTLY_TEMPLATE,
        marginal="box",
    )
    fig2 = apply_chart_style(fig2, "📊 Polarity Score Distribution (50K sample)", 380)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Sentiment by city (top 20 most positive)
    if sent_city is not None:
        col3, col4 = st.columns(2)

        with col3:
            top_pos = sent_city.nlargest(20, "avg_polarity")
            fig3 = px.bar(
                top_pos.sort_values("avg_polarity"),
                x="avg_polarity", y="City", orientation="h",
                color="avg_polarity", color_continuous_scale="Greens",
                labels={"avg_polarity": "Avg Polarity", "City": ""},
                template=PLOTLY_TEMPLATE,
            )
            fig3 = apply_chart_style(fig3, "😊 Top 20 Most Positive Cities", 480)
            fig3.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            top_neg = sent_city.nsmallest(20, "avg_polarity")
            fig4 = px.bar(
                top_neg.sort_values("avg_polarity", ascending=False),
                x="avg_polarity", y="City", orientation="h",
                color="avg_polarity", color_continuous_scale="Reds_r",
                labels={"avg_polarity": "Avg Polarity", "City": ""},
                template=PLOTLY_TEMPLATE,
            )
            fig4 = apply_chart_style(fig4, "😞 Top 20 Most Negative Cities", 480)
            fig4.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Sentiment vs Rating scatter
    fig5 = px.scatter(
        sample.sample(min(10000, len(sample))),
        x="polarity", y="Rating",
        color="sentiment",
        color_discrete_map={"Positive": "#34d399", "Neutral": "#60a5fa", "Negative": "#f87171"},
        opacity=0.35,
        labels={"polarity": "Sentiment Polarity", "Rating": "Star Rating"},
        template=PLOTLY_TEMPLATE,
    )
    fig5 = apply_chart_style(fig5, "⭐ Sentiment Polarity vs Star Rating (10K sample)", 400)
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: HOTSPOT RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
def tab_hotspots():
    st.markdown("## 🔥 Hotspot Rankings")
    st.markdown("""
    <div class="info-box">
    <strong>Hotspot Score = 0.5 × Rating Score + 0.3 × Review Volume + 0.2 × Sentiment Score</strong><br>
    All sub-scores are Min-Max normalised to [0, 1]. Review Volume uses log-scaling to dampen outliers.
    </div>
    """, unsafe_allow_html=True)

    hs_city  = load_csv("hotspots_city.csv")
    hs_place = load_csv("hotspots_place.csv")

    view = st.radio("View hotspots by", ["Cities", "Places"], horizontal=True)
    df_hs = hs_city if view == "Cities" else hs_place
    name_col = "City" if view == "Cities" else "Place"

    if df_hs is None:
        st.error(f"hotspots_{view.lower()}.csv not found. Please run hotspot.py first.")
        return

    top_n = st.slider("Top N destinations", 10, min(100, len(df_hs)), 25, key="hs_n")
    df_top = df_hs.head(top_n)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.bar(
            df_top.sort_values("hotspot_score"),
            x="hotspot_score", y=name_col,
            orientation="h",
            color="hotspot_score",
            color_continuous_scale="Purpor",
            labels={"hotspot_score": "Hotspot Score", name_col: ""},
            template=PLOTLY_TEMPLATE,
        )
        fig = apply_chart_style(fig, f"🔥 Top {top_n} {view} by Hotspot Score", 600)
        fig.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar/spider for top 5
        top5 = df_top.head(5)
        cats = ["rating_score", "volume_score", "sentiment_score"]
        cats_label = ["Rating Score", "Volume Score", "Sentiment Score"]
        fig2 = go.Figure()
        for _, row in top5.iterrows():
            vals = [row.get(c, 0) for c in cats]
            vals_c = vals + [vals[0]]  # close polygon
            fig2.add_trace(go.Scatterpolar(
                r=vals_c,
                theta=cats_label + [cats_label[0]],
                fill="toself",
                name=str(row[name_col])[:22],
                opacity=0.7,
            ))
        fig2.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.04)",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="rgba(255,255,255,0.1)",
                                tickfont_color="#9090b8"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                                 tickfont_color="#c0bce8"),
            ),
            title=dict(text="🕸️ Top 5 Score Breakdown",
                       font=dict(size=15, color="#c4b5fd"), x=0.02),
            height=420,
            paper_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#c0bce8", family="Inter"),
            showlegend=True,
            legend=dict(bgcolor="rgba(255,255,255,0.05)",
                        bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Donut breakdown for #1
        best = df_top.iloc[0]
        components = [
            best.get("rating_score",    0) * 0.5,
            best.get("volume_score",    0) * 0.3,
            best.get("sentiment_score", 0) * 0.2,
        ]
        fig3 = go.Figure(go.Pie(
            labels=["Rating (50%)", "Volume (30%)", "Sentiment (20%)"],
            values=components,
            hole=0.6,
            marker_colors=["#7c3aed", "#2563eb", "#059669"],
        ))
        fig3.update_layout(
            title=dict(text=f"#1: {str(best[name_col])[:20]}",
                       font=dict(size=14, color="#c4b5fd")),
            height=300,
            paper_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#c0bce8"),
            showlegend=True,
            legend=dict(bgcolor="rgba(255,255,255,0.05)", borderwidth=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Leaderboard table
    st.markdown("### 🏆 Hotspot Leaderboard")
    display_cols = [c for c in [name_col, "hotspot_score", "avg_rating",
                                "review_count", "avg_polarity",
                                "rating_score", "volume_score", "sentiment_score"]
                    if c in df_top.columns]
    st.dataframe(
        df_top[display_cols].style.format({
            "hotspot_score":    "{:.4f}",
            "avg_rating":       "{:.3f}",
            "review_count":     "{:,}",
            "avg_polarity":     "{:.4f}",
            "rating_score":     "{:.4f}",
            "volume_score":     "{:.4f}",
            "sentiment_score":  "{:.4f}",
        }).background_gradient(subset=["hotspot_score"], cmap="Purples"),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: PREDICTION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
# ============================================================================
# TAB: HIDDEN GEMS AND RISK SIGNALS
# ============================================================================
def tab_discover():
    st.markdown("## Destination Discovery")
    st.markdown(
        '<div class="info-box">'
        "<strong>Hidden gems</strong> are high-quality destinations with strong sentiment and lower visibility. "
        "<strong>Risk signals</strong> highlight popular places where rating, sentiment, or consistency may disappoint visitors."
        "</div>",
        unsafe_allow_html=True,
    )

    places = prepare_place_insights()
    if places is None:
        st.error("hotspots_place.csv not found. Please run hotspot.py first.")
        return

    c1, c2, c3 = st.columns(3)
    min_reviews = c1.slider("Minimum reviews for hidden gems", 5, 500, 30, step=5)
    max_reviews = c2.slider("Maximum reviews for hidden gems", 100, 5000, 1200, step=100)
    risk_min_reviews = c3.slider("Minimum reviews for risk list", 100, 10000, 800, step=100)

    hidden = places[
        (places["review_count"] >= min_reviews) &
        (places["review_count"] <= max_reviews)
    ].sort_values("hidden_gem_score", ascending=False).head(25)

    risk = places[
        places["review_count"] >= risk_min_reviews
    ].sort_values("risk_score", ascending=False).head(25)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Hidden Gems")
        if hidden.empty:
            st.warning("No hidden gems match the selected review range.")
        else:
            fig = px.scatter(
                hidden,
                x="review_count",
                y="avg_rating",
                size="hidden_gem_score",
                color="avg_polarity",
                hover_name="Place",
                color_continuous_scale="Tealgrn",
                labels={
                    "review_count": "Review Count",
                    "avg_rating": "Average Rating",
                    "avg_polarity": "Sentiment",
                },
                template=PLOTLY_TEMPLATE,
                size_max=34,
            )
            fig = apply_chart_style(fig, "High Quality, Lower Visibility", 430)
            st.plotly_chart(fig, use_container_width=True)

            display_cols = [c for c in ["Place", "City", "hidden_gem_score", "avg_rating", "review_count", "avg_polarity", "rating_std"] if c in hidden.columns]
            st.dataframe(
                hidden[display_cols].style.format({
                    "hidden_gem_score": "{:.4f}",
                    "avg_rating": "{:.3f}",
                    "review_count": "{:,}",
                    "avg_polarity": "{:.4f}",
                    "rating_std": "{:.4f}",
                }).background_gradient(subset=["hidden_gem_score"], cmap="YlGn"),
                use_container_width=True,
                hide_index=True,
            )

    with col2:
        st.markdown("### Overhyped / Risk Watch")
        if risk.empty:
            st.warning("No risk candidates match the selected review threshold.")
        else:
            fig2 = px.scatter(
                risk,
                x="review_count",
                y="avg_rating",
                size="risk_score",
                color="avg_polarity",
                hover_name="Place",
                color_continuous_scale="RdYlGn",
                labels={
                    "review_count": "Review Count",
                    "avg_rating": "Average Rating",
                    "avg_polarity": "Sentiment",
                },
                template=PLOTLY_TEMPLATE,
                size_max=34,
            )
            fig2 = apply_chart_style(fig2, "High Attention, Lower Experience Signals", 430)
            st.plotly_chart(fig2, use_container_width=True)

            display_cols = [c for c in ["Place", "City", "risk_score", "avg_rating", "review_count", "avg_polarity", "rating_std"] if c in risk.columns]
            st.dataframe(
                risk[display_cols].style.format({
                    "risk_score": "{:.4f}",
                    "avg_rating": "{:.3f}",
                    "review_count": "{:,}",
                    "avg_polarity": "{:.4f}",
                    "rating_std": "{:.4f}",
                }).background_gradient(subset=["risk_score"], cmap="OrRd"),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Quality vs Popularity Landscape")
    sample = places.sample(min(4000, len(places)), random_state=42)
    fig3 = px.scatter(
        sample,
        x="review_count",
        y="avg_rating",
        color="hotspot_score",
        size="confidence_score",
        hover_name="Place",
        log_x=True,
        color_continuous_scale="Turbo",
        labels={
            "review_count": "Review Count (log scale)",
            "avg_rating": "Average Rating",
            "hotspot_score": "Hotspot Score",
            "confidence_score": "Volume Confidence",
        },
        template=PLOTLY_TEMPLATE,
        size_max=18,
    )
    fig3 = apply_chart_style(fig3, "All Places: Rating vs Review Volume", 520)
    st.plotly_chart(fig3, use_container_width=True)


# ============================================================================
# TAB: CUSTOM HOTSPOT WEIGHT SIMULATOR
# ============================================================================
def tab_weight_simulator():
    st.markdown("## Hotspot Weight Simulator")
    st.markdown(
        '<div class="info-box">'
        "Change the importance of rating, review volume, and sentiment to see how destination rankings move in real time."
        "</div>",
        unsafe_allow_html=True,
    )

    view = st.radio("Re-rank", ["Cities", "Places"], horizontal=True, key="sim_view")
    base = prepare_city_insights() if view == "Cities" else prepare_place_insights()
    name_col = "City" if view == "Cities" else "Place"
    if base is None:
        st.error("Required hotspot outputs are missing. Please run hotspot.py first.")
        return

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    rating_w = c1.slider("Rating weight", 0, 100, 50, key="sim_rating")
    volume_w = c2.slider("Volume weight", 0, 100, 30, key="sim_volume")
    sentiment_w = c3.slider("Sentiment weight", 0, 100, 20, key="sim_sentiment")
    top_n = c4.slider("Top N", 10, 50, 20, key="sim_topn")

    ranked = recompute_hotspot_score(base, rating_w, volume_w, sentiment_w)
    ranked["custom_rank"] = np.arange(1, len(ranked) + 1)
    if "rank" in ranked.columns:
        ranked["rank_change"] = pd.to_numeric(ranked["rank"], errors="coerce") - ranked["custom_rank"]
    else:
        ranked["rank_change"] = 0
    top = ranked.head(top_n)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.bar(
            top.sort_values("custom_hotspot_score"),
            x="custom_hotspot_score",
            y=name_col,
            orientation="h",
            color="rank_change",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"custom_hotspot_score": "Custom Hotspot Score", "rank_change": "Rank Move", name_col: ""},
            template=PLOTLY_TEMPLATE,
        )
        fig = apply_chart_style(fig, f"Custom Top {top_n} {view}", 560)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_ternary = top.head(20).copy()
        fig2 = px.scatter_ternary(
            top_ternary,
            a="rating_score_live",
            b="volume_score_live",
            c="sentiment_score_live",
            color="custom_hotspot_score",
            hover_name=name_col,
            color_continuous_scale="Turbo",
            labels={
                "rating_score_live": "Rating",
                "volume_score_live": "Volume",
                "sentiment_score_live": "Sentiment",
            },
            template=PLOTLY_TEMPLATE,
        )
        fig2 = apply_chart_style(fig2, "Score Balance", 560)
        st.plotly_chart(fig2, use_container_width=True)

    display_cols = [name_col, "custom_rank", "rank_change", "custom_hotspot_score", "hotspot_score", "avg_rating", "review_count", "avg_polarity"]
    display_cols = [c for c in display_cols if c in top.columns]
    st.dataframe(
        top[display_cols].style.format({
            "custom_hotspot_score": "{:.4f}",
            "hotspot_score": "{:.4f}",
            "avg_rating": "{:.3f}",
            "review_count": "{:,}",
            "avg_polarity": "{:.4f}",
            "rank_change": "{:+.0f}",
        }).background_gradient(subset=["custom_hotspot_score"], cmap="viridis"),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================================
# TAB: COMPARISON AND RECOMMENDATIONS
# ============================================================================
def tab_compare_recommend():
    st.markdown("## Compare & Recommend")
    st.markdown(
        '<div class="info-box">'
        "Compare destinations side by side, then generate similar-place recommendations from rating, volume, sentiment, consistency, and hotspot score."
        "</div>",
        unsafe_allow_html=True,
    )

    view = st.radio("Compare type", ["Cities", "Places"], horizontal=True, key="cmp_view")
    catalog = destination_catalog(view)
    name_col = "City" if view == "Cities" else "Place"
    if catalog is None:
        st.error("Required output files are missing.")
        return

    names = sorted(catalog["Destination"].dropna().astype(str).unique())
    default = names[:2] if len(names) >= 2 else names
    selected = st.multiselect("Select two or more destinations", names, default=default, max_selections=4)
    if len(selected) < 2:
        st.info("Select at least two destinations to compare.")
        return

    cmp_df = catalog[catalog["Destination"].isin(selected)].copy()
    metric_cols = ["avg_rating", "review_count", "avg_polarity", "hotspot_score", "rating_std"]
    metric_cols = [c for c in metric_cols if c in cmp_df.columns]

    col1, col2 = st.columns([2, 3])
    with col1:
        for _, row in cmp_df.iterrows():
            st.metric(
                row["Destination"],
                f"{row.get('hotspot_score', 0):.3f}",
                delta=f"Rating {row.get('avg_rating', 0):.2f} | Reviews {int(row.get('review_count', 0)):,}",
            )

    with col2:
        radar = go.Figure()
        radar_cols = ["avg_rating", "review_count", "avg_polarity", "hotspot_score"]
        available = [c for c in radar_cols if c in cmp_df.columns]
        normed = cmp_df[available].copy()
        for c in available:
            if c == "review_count":
                normed[c] = log_volume_score(normed[c])
            else:
                normed[c] = minmax(normed[c])
        for idx, row in cmp_df.reset_index(drop=True).iterrows():
            vals = normed.iloc[idx][available].fillna(0).tolist()
            radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=available + [available[0]],
                fill="toself",
                name=row["Destination"][:24],
            ))
        radar.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.04)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            ),
            height=430,
            paper_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#c0bce8", family="Inter"),
            legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            margin=dict(l=30, r=30, t=30, b=30),
        )
        st.plotly_chart(radar, use_container_width=True)

    fig = px.bar(
        cmp_df.melt(id_vars=["Destination"], value_vars=metric_cols, var_name="Metric", value_name="Value"),
        x="Metric",
        y="Value",
        color="Destination",
        barmode="group",
        template=PLOTLY_TEMPLATE,
    )
    fig = apply_chart_style(fig, "Metric Comparison", 430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Similar Destination Recommendations")
    places = prepare_place_insights()
    if places is None:
        st.warning("Place recommendations require hotspots_place.csv.")
        return

    place_names = sorted(places["Place"].dropna().astype(str).unique())
    seed_default = selected[0] if view == "Places" and selected[0] in place_names else place_names[0]
    seed = st.selectbox("Choose a place to recommend from", place_names, index=place_names.index(seed_default))
    recs = recommend_similar_places(places, seed, top_n=12)
    if recs.empty:
        st.warning("No recommendations found for the selected place.")
    else:
        fig2 = px.bar(
            recs.sort_values("similarity"),
            x="similarity",
            y="Place",
            orientation="h",
            color="hotspot_score",
            color_continuous_scale="Teal",
            labels={"similarity": "Similarity", "Place": "", "hotspot_score": "Hotspot"},
            template=PLOTLY_TEMPLATE,
        )
        fig2 = apply_chart_style(fig2, f"Places Similar to {seed}", 480)
        st.plotly_chart(fig2, use_container_width=True)
        display_cols = [c for c in ["Place", "City", "similarity", "hotspot_score", "avg_rating", "review_count", "avg_polarity"] if c in recs.columns]
        st.dataframe(
            recs[display_cols].style.format({
                "similarity": "{:.4f}",
                "hotspot_score": "{:.4f}",
                "avg_rating": "{:.3f}",
                "review_count": "{:,}",
                "avg_polarity": "{:.4f}",
            }).background_gradient(subset=["similarity"], cmap="BuGn"),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================================
# TAB: ASPECT-BASED REVIEW SIGNALS
# ============================================================================
def tab_aspect_signals():
    st.markdown("## Aspect-Based Review Signals")
    st.markdown(
        '<div class="info-box">'
        "This screen scans sampled review text for practical visitor concerns such as cleanliness, crowd, food, price, safety, family suitability, scenery, and transport."
        "</div>",
        unsafe_allow_html=True,
    )

    sample_size = st.slider("Review sample size", 10000, 100000, 60000, step=10000)
    reviews = load_review_text_sample(sample_size)
    if reviews is None or reviews.empty:
        st.error("Could not load review text from data/Review_db.csv.")
        return

    city_options = ["All"] + sorted(reviews["City"].dropna().astype(str).value_counts().head(80).index.tolist())
    city = st.selectbox("Filter by city", city_options)
    filtered = reviews if city == "All" else reviews[reviews["City"].astype(str) == city]

    place_options = ["All"] + sorted(filtered["Place"].dropna().astype(str).value_counts().head(80).index.tolist())
    place = st.selectbox("Filter by place", place_options)
    if place != "All":
        filtered = filtered[filtered["Place"].astype(str) == place]

    summary = build_aspect_summary(filtered)
    if summary.empty:
        st.warning("No aspect keywords found in the selected review sample.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Sampled Reviews", f"{len(filtered):,}")
    c2.metric("Top Aspect", summary.iloc[0]["Aspect"])
    c3.metric("Top Aspect Rating", f"{summary.iloc[0]['Avg Rating']:.2f}")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.bar(
            summary.sort_values("Signal"),
            x="Signal",
            y="Aspect",
            orientation="h",
            color="Mentions",
            color_continuous_scale="Viridis",
            labels={"Signal": "Rating-based Sentiment Signal", "Aspect": ""},
            template=PLOTLY_TEMPLATE,
        )
        fig = apply_chart_style(fig, "Aspect Experience Signal", 460)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(data=go.Heatmap(
            z=summary[["Positive Share", "Negative Share"]].values,
            x=["Positive Share", "Negative Share"],
            y=summary["Aspect"],
            colorscale="RdYlGn",
            reversescale=False,
            zmin=0,
            zmax=1,
        ))
        fig2 = apply_chart_style(fig2, "Positive vs Negative Share", 460)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        summary.style.format({
            "Mentions": "{:,}",
            "Avg Rating": "{:.3f}",
            "Positive Share": "{:.2%}",
            "Negative Share": "{:.2%}",
            "Signal": "{:.4f}",
        }).background_gradient(subset=["Signal"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )


def tab_prediction():
    st.markdown("## 🤖 Popularity Prediction")
    st.markdown('<div class="info-box"><strong>Model:</strong> Random Forest Regressor | <strong>Target:</strong> log1p(review_count) per place | <strong>Note:</strong> Date column is fully null — no temporal features used.</div>', unsafe_allow_html=True)

    metrics_df  = load_csv("model_metrics.csv")
    fi_df       = load_csv("feature_importances.csv")
    pred_all    = load_csv("all_place_predictions.csv")
    pred_test   = load_csv("prediction_results.csv")

    # ── Assumptions callout
    with st.expander("📌 Model Assumptions & Design Decisions", expanded=True):
        st.markdown("""
        | Decision | Rationale |
        |---|---|
        | **Date column excluded** | 100% null — no temporal signal possible |
        | **Target = log1p(review_count)** | Proxy for popularity; log-scale reduces skew |
        | **Random Forest chosen** | Handles non-linearity, outlier-robust, no scaling needed |
        | **City-level features** | Provide spatial context since date unavailable |
        | **Sentiment as feature** | Review quality correlates with organic growth |
        | **Cross-validation (5-fold)** | Ensures generalisability without overfitting |
        """)

    # ── Metrics
    if metrics_df is not None:
        st.markdown("### 📏 Model Performance Metrics")
        row = metrics_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test MAE",    f"{row.get('test_mae', 'N/A'):.4f}")
        c2.metric("Test RMSE",   f"{row.get('test_rmse','N/A'):.4f}")
        c3.metric("Test R²",     f"{row.get('test_r2',  'N/A'):.4f}")
        c4.metric("CV R² (mean)",f"{row.get('cv_r2_mean','N/A'):.4f} ± {row.get('cv_r2_std',0):.4f}")

    # ── Feature Importances
    if fi_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                fi_df.sort_values("importance"),
                x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale="Purpor",
                labels={"importance": "Importance", "feature": "Feature"},
                template=PLOTLY_TEMPLATE,
            )
            fig = apply_chart_style(fig, "📊 Feature Importances (Random Forest)", 380)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.pie(
                fi_df, names="feature", values="importance",
                color_discrete_sequence=px.colors.sequential.Purpor,
                hole=0.5,
                template=PLOTLY_TEMPLATE,
            )
            fig2 = apply_chart_style(fig2, "🍩 Feature Importance Share", 380)
            fig2.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Actual vs Predicted scatter
    if pred_test is not None:
        sample_pred = pred_test.sample(min(3000, len(pred_test)), random_state=42)
        fig3 = px.scatter(
            sample_pred,
            x="actual_log_popularity", y="predicted_log_popularity",
            color="residual",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            opacity=0.5,
            labels={
                "actual_log_popularity":    "Actual log(popularity)",
                "predicted_log_popularity": "Predicted log(popularity)",
                "residual": "Residual",
            },
            template=PLOTLY_TEMPLATE,
        )
        # Perfect prediction line
        mn = sample_pred["actual_log_popularity"].min()
        mx = sample_pred["actual_log_popularity"].max()
        fig3.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                       line=dict(color="#a78bfa", dash="dash", width=2))
        fig3 = apply_chart_style(fig3, "🎯 Actual vs Predicted Popularity (test set sample)", 420)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Top predicted places
    if pred_all is not None:
        st.markdown("### 🏅 Top 20 Predicted Popular Places")
        top20 = pred_all.nlargest(20, "predicted_review_count")
        st.dataframe(
            top20.style.format({
                "review_count":             "{:,}",
                "predicted_review_count":   "{:,}",
                "avg_rating":               "{:.3f}",
                "predicted_log_popularity": "{:.4f}",
            }).background_gradient(subset=["predicted_review_count"], cmap="Purples"),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: INTERACTIVE MAP
# ══════════════════════════════════════════════════════════════════════════════

# ── Curated geocoordinates for top tourism cities (hardcoded since no geo data in dataset)
CITY_COORDS = {
    "New York":        [40.7128, -74.0060],
    "London":          [51.5074,  -0.1278],
    "Paris":           [48.8566,   2.3522],
    "Tokyo":           [35.6762, 139.6503],
    "Dubai":           [25.2048,  55.2708],
    "Rome":            [41.9028,  12.4964],
    "Barcelona":       [41.3851,   2.1734],
    "Amsterdam":       [52.3676,   4.9041],
    "Singapore":       [1.3521,  103.8198],
    "Sydney":          [-33.8688, 151.2093],
    "Bangkok":         [13.7563, 100.5018],
    "Istanbul":        [41.0082,  28.9784],
    "Mumbai":          [19.0760,  72.8777],
    "Delhi":           [28.6139,  77.2090],
    "Los Angeles":     [34.0522, -118.2437],
    "Chicago":         [41.8781,  -87.6298],
    "Toronto":         [43.6532,  -79.3832],
    "Berlin":          [52.5200,  13.4050],
    "Madrid":          [40.4168,  -3.7038],
    "Prague":          [50.0755,  14.4378],
    "Vienna":          [48.2082,  16.3738],
    "Budapest":        [47.4979,  19.0402],
    "Athens":          [37.9838,  23.7275],
    "Lisbon":          [38.7223,  -9.1393],
    "Copenhagen":      [55.6761,  12.5683],
    "Stockholm":       [59.3293,  18.0686],
    "Zurich":          [47.3769,   8.5417],
    "Seoul":           [37.5665, 126.9780],
    "Beijing":         [39.9042, 116.4074],
    "Shanghai":        [31.2304, 121.4737],
    "Hong Kong":       [22.3193, 114.1694],
    "Kuala Lumpur":    [3.1390,  101.6869],
    "Jakarta":         [-6.2088, 106.8456],
    "Cairo":           [30.0444,  31.2357],
    "Cape Town":       [-33.9249,  18.4241],
    "Nairobi":         [-1.2921,  36.8219],
    "Mexico City":     [19.4326,  -99.1332],
    "Buenos Aires":    [-34.6037,  -58.3816],
    "Rio de Janeiro":  [-22.9068,  -43.1729],
    "São Paulo":       [-23.5505,  -46.6333],
    "Bali":            [-8.4095,  115.1889],
    "Phuket":          [7.8804,   98.3923],
    "Maldives":        [3.2028,   73.2207],
    "Las Vegas":       [36.1699, -115.1398],
    "San Francisco":   [37.7749, -122.4194],
    "Miami":           [25.7617,  -80.1918],
    "Orlando":         [28.5383,  -81.3792],
    "Vancouver":       [49.2827, -123.1207],
    "Melbourne":       [-37.8136, 144.9631],
}


def tab_map_legacy():
    st.markdown("## 🗺️ Interactive Tourism Map")
    st.markdown('<div class="info-box">Interactive Folium map showing top tourism hotspots. Marker size and colour reflect the Hotspot Score. City coordinates are geocoded from a curated reference list.</div>', unsafe_allow_html=True)

    hs_city = load_csv("hotspots_city.csv")
    if hs_city is None:
        st.error("hotspots_city.csv not found. Please run hotspot.py first.")
        return

    # ── Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        map_top_n = st.slider("Show top N cities on map", 5, 50, 30, key="map_n")
    with col_ctrl2:
        score_min = st.slider(
            "Minimum Hotspot Score",
            0.0, 1.0, 0.0, 0.01, key="map_score"
        )

    df_map = hs_city[hs_city["hotspot_score"] >= score_min].head(map_top_n).copy()

    # Match city names to coordinates (fuzzy by lowercase strip)
    def get_coords(city_name):
        # Exact match
        if city_name in CITY_COORDS:
            return CITY_COORDS[city_name]
        # Case-insensitive match
        city_lower = city_name.lower().strip()
        for k, v in CITY_COORDS.items():
            if k.lower() == city_lower:
                return v
        # Partial match (city name starts with known key)
        for k, v in CITY_COORDS.items():
            if city_lower.startswith(k.lower()[:5]):
                return v
        return None

    df_map["coords"] = df_map["City"].apply(get_coords)
    df_map_geo = df_map.dropna(subset=["coords"]).copy()
    df_map_geo["lat"] = df_map_geo["coords"].apply(lambda c: c[0])
    df_map_geo["lon"] = df_map_geo["coords"].apply(lambda c: c[1])

    if len(df_map_geo) == 0:
        st.warning("None of the top cities matched the geocoordinate reference. Try running with more cities.")
        st.info("Mapped cities available: " + ", ".join(CITY_COORDS.keys()))
        return

    st.info(f"Displaying {len(df_map_geo)} cities with known coordinates (out of {len(df_map)} filtered).")

    # ── Build Folium map
    m = folium.Map(
        location=[20, 0], zoom_start=2,
        tiles="CartoDB dark_matter",
    )

    # Colour scale: score → hex colour
    max_score = df_map_geo["hotspot_score"].max()
    min_score = df_map_geo["hotspot_score"].min()

    def score_to_color(score):
        """Map score to a purple-orange gradient hex."""
        t = (score - min_score) / (max_score - min_score + 1e-9)
        r = int(80  + t * 175)
        g = int(20  + t * 90)
        b = int(200 - t * 130)
        return f"#{r:02x}{g:02x}{b:02x}"

    for _, row in df_map_geo.iterrows():
        score  = row["hotspot_score"]
        radius = 8 + score * 22        # marker size scales with score
        color  = score_to_color(score)

        popup_html = f"""
        <div style="font-family:Inter,sans-serif; min-width:180px;">
          <h4 style="margin:0 0 6px;color:#7c3aed;">📍 {row['City']}</h4>
          <table style="border-collapse:collapse;width:100%;">
            <tr><td style="color:#888;">Hotspot Score</td>
                <td style="font-weight:700;color:#a78bfa;">{score:.4f}</td></tr>
            <tr><td style="color:#888;">Avg Rating</td>
                <td>⭐ {row.get('avg_rating', 'N/A'):.3f}</td></tr>
            <tr><td style="color:#888;">Reviews</td>
                <td>{int(row.get('review_count', 0)):,}</td></tr>
            <tr><td style="color:#888;">Sentiment</td>
                <td>{row.get('avg_polarity', 0):.4f}</td></tr>
            <tr><td style="color:#888;">Rank</td>
                <td>#{int(row.get('rank', '-'))}</td></tr>
          </table>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"🌍 {row['City']} | Score: {score:.3f}",
        ).add_to(m)

    # Render map
    st_folium(m, width="100%", height=580, returned_objects=[])

    # Legend / stats
    st.markdown(f"**Cities shown:** {len(df_map_geo)} &nbsp;|&nbsp; "
                f"**Score range:** {df_map_geo['hotspot_score'].min():.4f} – "
                f"{df_map_geo['hotspot_score'].max():.4f}")

    # Table below map
    st.markdown("### 📋 Mapped Cities Data")
    disp_cols = [c for c in ["rank", "City", "hotspot_score", "avg_rating",
                              "review_count", "avg_polarity"] if c in df_map_geo.columns]
    st.dataframe(df_map_geo[disp_cols].style.format({
        "hotspot_score": "{:.4f}",
        "avg_rating":    "{:.3f}",
        "review_count":  "{:,}",
        "avg_polarity":  "{:.4f}",
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: REAL-TIME TRENDS  (Google Trends)
# ══════════════════════════════════════════════════════════════════════════════
INDIAN_CITY_COORDS = {
    "Agonda": [15.0447, 73.9854],
    "Agra": [27.1767, 78.0081],
    "Ahmedabad": [23.0225, 72.5714],
    "Aluva": [10.1076, 76.3516],
    "Amer": [26.9855, 75.8513],
    "Amritsar": [31.6340, 74.8723],
    "Baga": [15.5553, 73.7517],
    "Bambolim": [15.4632, 73.8505],
    "Bardez": [15.6000, 73.8200],
    "Belur": [13.1637, 75.8657],
    "Benaulim": [15.2500, 73.9280],
    "Bengaluru": [12.9716, 77.5946],
    "Bhopal": [23.2599, 77.4126],
    "Canacona": [15.0091, 74.0233],
    "Cavelossim": [15.1724, 73.9419],
    "Chandigarh": [30.7333, 76.7794],
    "Chennai": [13.0827, 80.2707],
    "Coimbatore": [11.0168, 76.9558],
    "Colva": [15.2794, 73.9229],
    "Dabolim": [15.3800, 73.8330],
    "Darjeeling": [27.0410, 88.2663],
    "Dehradun": [30.3165, 78.0322],
    "Dharamsala": [32.2190, 76.3234],
    "Gangtok": [27.3314, 88.6138],
    "Geyzing": [27.2890, 88.2570],
    "Gurugram (Gurgaon)": [28.4595, 77.0266],
    "Havelock Island": [12.0085, 93.0060],
    "Hyderabad": [17.3850, 78.4867],
    "Jaipur": [26.9124, 75.7873],
    "Jodhpur": [26.2389, 73.0243],
    "Khopoli": [18.7856, 73.3459],
    "Kochi (Cochin)": [9.9312, 76.2673],
    "Kodaikanal": [10.2381, 77.4892],
    "Kolkata": [22.5726, 88.3639],
    "Kovalam": [8.4004, 76.9787],
    "Leh": [34.1526, 77.5771],
    "Lonavala": [18.7557, 73.4091],
    "Madurai": [9.9252, 78.1198],
    "Magadi": [12.9573, 77.2242],
    "Mahabalipuram": [12.6269, 80.1927],
    "Mahabaleshwar": [17.9307, 73.6477],
    "Manali": [32.2432, 77.1892],
    "Mandrem": [15.6626, 73.7134],
    "Mangalore": [12.9141, 74.8560],
    "Mararikulam": [9.5946, 76.3119],
    "Morjim": [15.6300, 73.7350],
    "Mount Abu": [24.5926, 72.7156],
    "Mumbai": [19.0760, 72.8777],
    "Munnar": [10.0889, 77.0595],
    "Mysuru (Mysore)": [12.2958, 76.6394],
    "Nainital": [29.3919, 79.4542],
    "Nashik": [19.9975, 73.7898],
    "Navi Mumbai": [19.0330, 73.0297],
    "New Delhi": [28.6139, 77.2090],
    "Nileshwar": [12.2600, 75.1350],
    "Noida": [28.5355, 77.3910],
    "Ooty": [11.4102, 76.6950],
    "Pahalgam": [34.0153, 75.3184],
    "Panjim": [15.4909, 73.8278],
    "Pondicherry": [11.9416, 79.8083],
    "Pune": [18.5204, 73.8567],
    "Sadri": [25.1856, 73.4368],
    "Shimla": [31.1048, 77.1734],
    "Sinquerim": [15.5000, 73.7680],
    "Srinagar": [34.0837, 74.7973],
    "Surat": [21.1702, 72.8311],
    "Tapovan": [30.1308, 78.3290],
    "Trivandrum": [8.5241, 76.9366],
    "Udaipur": [24.5854, 73.7125],
    "Utorda": [15.3167, 73.9000],
    "Vadodara": [22.3072, 73.1812],
    "Varca": [15.2324, 73.9431],
    "Varkala Town": [8.7379, 76.7163],
    "Visakhapatnam (Vizag)": [17.6868, 83.2185],
    "Vrindavan": [27.5650, 77.6593],
    "Vypin Island": [10.1170, 76.2100],
}


def get_indian_city_coords(city_name):
    """Resolve a city name to curated Indian coordinates."""
    if not isinstance(city_name, str):
        return None
    clean = re.sub(r"\s+", " ", city_name).strip()
    if clean in INDIAN_CITY_COORDS:
        return INDIAN_CITY_COORDS[clean]

    clean_key = re.sub(r"\s*\([^)]*\)", "", clean).lower()
    for key, coords in INDIAN_CITY_COORDS.items():
        key_clean = re.sub(r"\s*\([^)]*\)", "", key).lower()
        if clean_key == key_clean:
            return coords
    for key, coords in INDIAN_CITY_COORDS.items():
        key_clean = re.sub(r"\s*\([^)]*\)", "", key).lower()
        if clean_key.startswith(key_clean) or key_clean.startswith(clean_key):
            return coords
    return None


def tab_map():
    st.markdown("## India Tourism Hotspot Map")
    st.markdown(
        '<div class="info-box">'
        "India-focused Folium map with clustered city markers and a heat layer. Marker size reflects review volume; marker color follows the selected tourism signal."
        "</div>",
        unsafe_allow_html=True,
    )

    hs_city = prepare_city_insights()
    if hs_city is None:
        st.error("hotspots_city.csv not found. Please run hotspot.py first.")
        return

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 1, 1])
    map_top_n = col_ctrl1.slider("Top cities", 10, 100, 60, key="india_map_n")
    score_min = col_ctrl2.slider("Minimum hotspot score", 0.0, 1.0, 0.0, 0.01, key="india_map_score")
    color_metric = col_ctrl3.selectbox("Color by", ["hotspot_score", "avg_rating", "avg_polarity", "review_count"])
    tile_label = col_ctrl4.selectbox("Map style", ["Light", "Dark", "Street"])

    tiles = {
        "Light": "CartoDB positron",
        "Dark": "CartoDB dark_matter",
        "Street": "OpenStreetMap",
    }[tile_label]

    df_map = hs_city[hs_city["hotspot_score"] >= score_min].head(map_top_n).copy()
    df_map["coords"] = df_map["City"].apply(get_indian_city_coords)
    df_geo = df_map.dropna(subset=["coords"]).copy()
    df_geo["lat"] = df_geo["coords"].apply(lambda c: c[0])
    df_geo["lon"] = df_geo["coords"].apply(lambda c: c[1])
    df_geo["volume_scale"] = log_volume_score(df_geo["review_count"])

    if df_geo.empty:
        st.warning("No selected cities matched the curated India coordinate list.")
        return

    mapped = len(df_geo)
    missing = len(df_map) - mapped
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mapped Cities", f"{mapped:,}")
    c2.metric("Unmapped In Selection", f"{missing:,}")
    c3.metric("Best Hotspot", df_geo.iloc[0]["City"])
    c4.metric("Score Range", f"{df_geo['hotspot_score'].min():.3f}-{df_geo['hotspot_score'].max():.3f}")

    metric_values = pd.to_numeric(df_geo[color_metric], errors="coerce").fillna(0)
    metric_scaled = minmax(metric_values)

    def metric_color(value):
        t = float(value)
        if t < 0.33:
            return "#2a9d8f"
        if t < 0.66:
            return "#3b82f6"
        return "#f59e0b"

    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles=tiles)
    cluster = MarkerCluster(name="City hotspot markers").add_to(m)

    heat_points = []
    for idx, (_, row) in enumerate(df_geo.iterrows()):
        score = float(row.get("hotspot_score", 0))
        review_count = float(row.get("review_count", 0))
        radius = 6 + 22 * float(row.get("volume_scale", 0))
        color = metric_color(metric_scaled.iloc[idx])
        heat_points.append([row["lat"], row["lon"], max(score, 0.05)])

        popup_html = f"""
        <div style="font-family:Inter,Arial,sans-serif; min-width:210px;">
          <h4 style="margin:0 0 8px;color:#1f2937;">{row['City']}</h4>
          <table style="border-collapse:collapse;width:100%;font-size:12px;">
            <tr><td>Rank</td><td><strong>#{int(row.get('rank', 0))}</strong></td></tr>
            <tr><td>Hotspot score</td><td><strong>{score:.4f}</strong></td></tr>
            <tr><td>Average rating</td><td>{row.get('avg_rating', 0):.3f}</td></tr>
            <tr><td>Reviews</td><td>{int(review_count):,}</td></tr>
            <tr><td>Sentiment</td><td>{row.get('avg_polarity', 0):.4f}</td></tr>
          </table>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.72,
            weight=2,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['City']} | {color_metric}: {row.get(color_metric, 0):.3f}",
        ).add_to(cluster)

    HeatMap(
        heat_points,
        name="Hotspot heat layer",
        min_opacity=0.25,
        radius=28,
        blur=18,
        max_zoom=7,
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    st_folium(m, width="100%", height=650, returned_objects=[])

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.scatter(
            df_geo,
            x="review_count",
            y="hotspot_score",
            color="avg_polarity",
            size="avg_rating",
            hover_name="City",
            log_x=True,
            color_continuous_scale="Turbo",
            labels={
                "review_count": "Review Count (log scale)",
                "hotspot_score": "Hotspot Score",
                "avg_polarity": "Sentiment",
                "avg_rating": "Rating",
            },
            template=PLOTLY_TEMPLATE,
            size_max=26,
        )
        fig = apply_chart_style(fig, "Mapped Cities: Volume vs Hotspot Strength", 430)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df_geo.sort_values(color_metric).tail(15),
            x=color_metric,
            y="City",
            orientation="h",
            color=color_metric,
            color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE,
        )
        fig2 = apply_chart_style(fig2, f"Top Mapped Cities by {color_metric}", 430)
        st.plotly_chart(fig2, use_container_width=True)

    disp_cols = [c for c in ["rank", "City", "hotspot_score", "avg_rating", "review_count", "avg_polarity", "rating_std"] if c in df_geo.columns]
    st.dataframe(
        df_geo[disp_cols].style.format({
            "hotspot_score": "{:.4f}",
            "avg_rating": "{:.3f}",
            "review_count": "{:,}",
            "avg_polarity": "{:.4f}",
            "rating_std": "{:.4f}",
        }).background_gradient(subset=["hotspot_score"], cmap="viridis"),
        use_container_width=True,
        hide_index=True,
    )


def tab_realtime_trends():
    st.markdown("## 📡 Real-Time Tourism Trends")
    st.markdown(
        '<div class="info-box">'
        "Google Trends interest scores are blended with your existing hotspot scores to produce "
        "a <strong>FinalRealtimeScore = 0.9 × ExistingHotspotScore + 0.1 × TrendScore</strong>. "
        "Run <code>python scripts/google_trends.py</code> to refresh live data."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Load data ────────────────────────────────────────────────────────────
    gt_df  = load_csv("google_trends.csv")         # Place, TrendScore
    rt_df  = load_csv("realtime_hotspots.csv")     # Place, ExistingHotspotScore, TrendScore, FinalRealtimeScore

    if gt_df is None and rt_df is None:
        st.warning(
            "⚠️ No real-time trend data found.  "
            "Please run: `python scripts/google_trends.py`"
        )
        st.markdown(
            '<div class="info-box">'
            "<strong>How to generate real-time data:</strong><br>"
            "1. Install pytrends: <code>pip install pytrends</code><br>"
            "2. Run: <code>python scripts/google_trends.py</code><br>"
            "3. Refresh this dashboard page."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Section 1: Top Trending Destinations ─────────────────────────────────
    if gt_df is not None:
        st.markdown("### 🔥 Top Trending Tourist Destinations")
        gt_sorted = gt_df.sort_values("TrendScore", ascending=False).reset_index(drop=True)

        # KPI row — top 3 trending
        kpi_cols = st.columns(min(3, len(gt_sorted)))
        for idx, col in enumerate(kpi_cols):
            row = gt_sorted.iloc[idx]
            col.metric(
                label=f"#{idx+1} Trending",
                value=row["Place"],
                delta=f"Trend Score: {row['TrendScore']:.4f}",
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Trend Score Table
        st.markdown("### 📋 Google Trends Score Table")
        st.dataframe(
            gt_sorted.style.format({"TrendScore": "{:.4f}"}).background_gradient(
                subset=["TrendScore"], cmap="Purples"
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Bar chart of trend scores
        st.markdown("### 📊 Trend Score Bar Chart")
        fig_gt = px.bar(
            gt_sorted.sort_values("TrendScore"),
            x="TrendScore",
            y="Place",
            orientation="h",
            color="TrendScore",
            color_continuous_scale="Purpor",
            labels={"TrendScore": "Google Trend Score (0-1)", "Place": ""},
            template=PLOTLY_TEMPLATE,
        )
        fig_gt = apply_chart_style(fig_gt, "📈 Google Trends Interest Score by Place", 520)
        fig_gt.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=10)
        st.plotly_chart(fig_gt, use_container_width=True)

    # ── Section 2: Top 10 Real-Time Hotspots ─────────────────────────────────
    if rt_df is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🏆 Top 10 Real-Time Hotspots")
        st.markdown(
            '<div class="info-box">'
            "<strong>FinalRealtimeScore = 0.9 × ExistingHotspotScore + 0.1 × TrendScore</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

        rt_sorted = rt_df.sort_values("FinalRealtimeScore", ascending=False).reset_index(drop=True)
        top10 = rt_sorted.head(10)

        col1, col2 = st.columns([3, 2])

        with col1:
            fig_rt = px.bar(
                top10.sort_values("FinalRealtimeScore"),
                x="FinalRealtimeScore",
                y="Place",
                orientation="h",
                color="FinalRealtimeScore",
                color_continuous_scale="Teal",
                labels={"FinalRealtimeScore": "Final Real-Time Score", "Place": ""},
                template=PLOTLY_TEMPLATE,
            )
            fig_rt = apply_chart_style(fig_rt, "🏆 Top 10 Real-Time Hotspots", 460)
            fig_rt.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=10)
            st.plotly_chart(fig_rt, use_container_width=True)

        with col2:
            # Grouped bar: Existing vs Trend score comparison
            top5 = top10.head(5).copy()
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name="Existing Score (×0.9)",
                x=top5["Place"].str[:20],
                y=(top5["ExistingHotspotScore"] * 0.9).round(4),
                marker_color="#7c3aed",
            ))
            fig_cmp.add_trace(go.Bar(
                name="Trend Score (×0.1)",
                x=top5["Place"].str[:20],
                y=(top5["TrendScore"] * 0.1).round(4),
                marker_color="#06b6d4",
            ))
            fig_cmp.update_layout(
                barmode="stack",
                title=dict(text="📊 Score Composition (Top 5)",
                           font=dict(size=14, color="#c4b5fd"), x=0.02),
                height=460,
                paper_bgcolor="rgba(255,255,255,0.03)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(family="Inter", color="#c0bce8"),
                legend=dict(bgcolor="rgba(255,255,255,0.05)",
                            bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
                            orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=40, r=20, t=60, b=80),
                xaxis=dict(tickangle=-25, gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Full leaderboard table
        st.markdown("#### 📋 Full Real-Time Leaderboard")
        st.dataframe(
            rt_sorted.style.format({
                "ExistingHotspotScore": "{:.4f}",
                "TrendScore":           "{:.4f}",
                "FinalRealtimeScore":   "{:.4f}",
            }).background_gradient(subset=["FinalRealtimeScore"], cmap="Purples"),
            use_container_width=True,
            hide_index=True,
        )

        # ── Scatter: Existing vs Final
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            rt_df,
            x="ExistingHotspotScore",
            y="FinalRealtimeScore",
            size="TrendScore",
            color="TrendScore",
            hover_name="Place",
            color_continuous_scale="Purpor",
            labels={
                "ExistingHotspotScore": "Existing Hotspot Score",
                "FinalRealtimeScore":   "Final Real-Time Score",
                "TrendScore":           "Trend Score",
            },
            template=PLOTLY_TEMPLATE,
            size_max=28,
        )
        fig_scatter = apply_chart_style(
            fig_scatter,
            "🔵 Existing Score vs Final Real-Time Score (bubble size = Trend Score)",
            400,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── How-to footer
    with st.expander("ℹ️ How Real-Time Trends Works", expanded=False):
        st.markdown("""
        | Step | Detail |
        |---|---|
        | **1. Hotspot Input** | Top places from `hotspots_place.csv` (hotspot.py output) |
        | **2. Google Trends API** | Fetches 3-month interest-over-time via `pytrends` |
        | **3. Normalisation** | Raw scores (0–100) → normalised to [0, 1] |
        | **4. Blending** | `FinalRealtimeScore = 0.9 × ExistingScore + 0.1 × TrendScore` |
        | **5. Output files** | `google_trends.csv` + `realtime_hotspots.csv` |
        | **Rate limiting** | Exponential back-off on 429 errors; defaults to 0 on failure |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
def main_legacy():
    render_sidebar()

    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🌍 Tourism Trend Analytics</div>
        <div class="hero-sub">
            Hotspot Prediction Using Big Data &nbsp;·&nbsp;
            1.48M Reviews &nbsp;·&nbsp; 1,794 Cities &nbsp;·&nbsp; 14,494 Places
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs([
        "📊 Overview",
        "🏙️ Top Cities",
        "📍 Top Places",
        "💬 Sentiment",
        "🔥 Hotspots",
        "🤖 Prediction",
        "🗺️ Map",
        "📡 Real-Time Trends",
    ])

    with tabs[0]: tab_overview()
    with tabs[1]: tab_top_cities()
    with tabs[2]: tab_top_places()
    with tabs[3]: tab_sentiment()
    with tabs[4]: tab_hotspots()
    with tabs[5]: tab_prediction()
    with tabs[6]: tab_map()
    with tabs[7]: tab_realtime_trends()


def main():
    render_sidebar()

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Tourism Trend Analytics</div>
        <div class="hero-sub">
            Hotspot Prediction Using Big Data &nbsp;|&nbsp;
            1.48M Reviews &nbsp;|&nbsp; 1,794 Cities &nbsp;|&nbsp; 14,494 Places
        </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "Overview",
        "Top Cities",
        "Top Places",
        "Sentiment",
        "Hotspots",
        "Simulator",
        "Discover",
        "Compare",
        "Aspects",
        "Prediction",
        "India Map",
    ])

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_top_cities()
    with tabs[2]:
        tab_top_places()
    with tabs[3]:
        tab_sentiment()
    with tabs[4]:
        tab_hotspots()
    with tabs[5]:
        tab_weight_simulator()
    with tabs[6]:
        tab_discover()
    with tabs[7]:
        tab_compare_recommend()
    with tabs[8]:
        tab_aspect_signals()
    with tabs[9]:
        tab_prediction()
    with tabs[10]:
        tab_map()


if __name__ == "__main__":
    main()
