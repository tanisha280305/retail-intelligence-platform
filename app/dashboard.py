import os
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import networkx as nx
from datetime import datetime
from google.cloud import storage

st.set_page_config(
    page_title="Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0A0A12 !important;
    color: #E2E2F0 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0F0F1A !important;
    border-right: 1px solid #1E1E30 !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #13131F;
    border: 1px solid #1E1E30;
    border-radius: 12px;
    padding: 1.1rem 1.4rem !important;
}
[data-testid="stMetricLabel"] { color: #7B7B9A !important; font-size: 0.78rem !important; letter-spacing: 0.06em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #E2E2F0 !important; font-size: 1.9rem !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] > div { color: #3DBA8C !important; }

/* ── Titles ── */
h1 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: -0.02em; font-size: 1.8rem !important; }
h2, h3 { color: #C8C8E0 !important; font-weight: 600 !important; }
.stMarkdown p { color: #9090B0; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1E1E30 !important; border-radius: 10px !important; overflow: hidden; }
[data-testid="stDataFrame"] thead th { background: #13131F !important; color: #7B7B9A !important; font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; border-bottom: 1px solid #1E1E30 !important; }
[data-testid="stDataFrame"] tbody td { background: #0A0A12 !important; color: #C8C8E0 !important; border-color: #1A1A28 !important; font-size: 0.85rem; }
[data-testid="stDataFrame"] tbody tr:hover td { background: #13131F !important; }

/* ── Buttons ── */
.stButton > button {
    background: #1A1A2E !important;
    color: #A0A0CC !important;
    border: 1px solid #2A2A45 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    background: #22223A !important;
    border-color: #4A4A7A !important;
    color: #E2E2F0 !important;
}

/* ── Selectbox / Radio / Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stRadio"] > div {
    background: #13131F !important;
    border: 1px solid #1E1E30 !important;
    border-radius: 8px !important;
    color: #C8C8E0 !important;
}
[data-testid="stRadio"] label { color: #9090B0 !important; }

/* ── Subheader rule ── */
.stSubheader { border-bottom: 1px solid #1E1E30; padding-bottom: 0.4rem; margin-bottom: 1rem; }

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    background: #13131F !important;
    border: 1px solid #1E1E30 !important;
    border-radius: 10px !important;
    color: #9090B0 !important;
}
[data-testid="stInfo"] { border-left: 3px solid #3D6ABA !important; }
[data-testid="stWarning"] { border-left: 3px solid #BA8C3D !important; }
[data-testid="stSuccess"] { border-left: 3px solid #3DBA8C !important; }

/* ── Tables ── */
table { border-collapse: collapse; width: 100%; }
thead tr { background: #13131F !important; }
thead th { color: #7B7B9A !important; font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; padding: 0.6rem 0.8rem; border-bottom: 1px solid #1E1E30; }
tbody td { color: #C8C8E0 !important; font-size: 0.84rem; padding: 0.55rem 0.8rem; border-bottom: 1px solid #141422; }
tbody tr:hover td { background: #13131F; }

/* ── Line chart ── */
[data-testid="stVegaLiteChart"] { background: #13131F !important; border-radius: 12px; padding: 1rem; border: 1px solid #1E1E30; }

/* ── Dividers ── */
hr { border-color: #1E1E30 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A0A12; }
::-webkit-scrollbar-thumb { background: #2A2A45; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Config ───────────────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_PREFIX = "processed"
LOCAL_PROCESSED_DIR = "processed"

# ─── Page State ────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "last_loaded_forecast" not in st.session_state:
    st.session_state.last_loaded_forecast = None

def navigate_to(p):
    st.session_state.page = p

# ─── Utilities ────────────────────────────────────────────────────────────────
def _download_blob_to_bytes(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket_name}/{blob_name} not found")
    return blob.download_as_bytes(), blob.updated

def _read_parquet_from_source(path, use_gcs_when_available=True):
    if use_gcs_when_available and GCS_BUCKET:
        full_blob = f"{GCS_PREFIX}/{path}"
        try:
            data, updated = _download_blob_to_bytes(GCS_BUCKET, full_blob)
            return pd.read_parquet(io.BytesIO(data)), updated
        except Exception as e:
            st.warning(f"Failed to load from GCS ({full_blob}): {e}. Falling back to local.")
    local_path = os.path.join(LOCAL_PROCESSED_DIR, path)
    if not os.path.exists(local_path):
        return None, None
    return pd.read_parquet(local_path), None

# ─── Data Loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=24*60*60)
def load_transactions():
    local = "data/cleaned_online_retail.parquet"
    if GCS_BUCKET:
        try:
            data, _ = _download_blob_to_bytes(GCS_BUCKET, "raw/cleaned_online_retail.parquet")
            return pd.read_parquet(io.BytesIO(data))
        except Exception:
            pass
    return pd.read_parquet(local)

@st.cache_data(ttl=24*60*60)
def load_precomputed_forecast():
    required = [
        ("daily.parquet", "daily"), ("hist_fc.parquet", "hist_fc"),
        ("future_fc.parquet", "future_fc"), ("hist.parquet", "hist"), ("anoms.parquet", "anoms"),
    ]
    results = {}
    loaded_ts = None
    for fname, key in required:
        df, updated = _read_parquet_from_source(fname)
        if df is None:
            return None, None, None, None, None, None
        results[key] = df
        if updated and (loaded_ts is None or updated > loaded_ts):
            loaded_ts = updated
    for k in ["daily", "hist_fc", "future_fc", "hist", "anoms"]:
        if "ds" in results[k].columns:
            results[k]["ds"] = pd.to_datetime(results[k]["ds"])
    return results["daily"], results["hist_fc"], results["future_fc"], results["hist"], results["anoms"], loaded_ts

@st.cache_data(ttl=24*60*60)
def load_rfm_and_profiles():
    rfm_df = pd.read_csv("data/customer_rfm_clusters.csv")
    profile_df = pd.read_csv("data/cluster_profile.csv")
    profile_df["SegmentName"] = profile_df["Cluster"].map({
        2: "VIPs", 3: "Loyal High-Value", 0: "Moderate", 1: "At-Risk/Lapsed"
    })
    return rfm_df, profile_df

@st.cache_data(ttl=24*60*60)
def load_product_segments():
    sku_df = pd.read_csv("data/product_segments.csv")
    profile_df = pd.read_csv("data/product_segment_profile.csv")
    profile_df["SegmentName"] = profile_df["Segment"].map({
        0: "Steady Sellers", 1: "Dead Stock", 2: "Super Performers", 3: "Niche Big-Tickets"
    })
    return sku_df, profile_df

@st.cache_data(ttl=24*60*60)
def load_rules(path="data/association_rules.csv"):
    df = pd.read_csv(path)


    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "antecedents" in df.columns:
        df["antecedents"] = df["antecedents"].apply(eval)

    if "consequents" in df.columns:
        df["consequents"] = df["consequents"].apply(eval)

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )

    return df

@st.cache_data(ttl=24*60*60)
def get_transactions_for_date(date):
    df = load_transactions()
    if isinstance(date, str):
        dt = pd.to_datetime(date).date()
    elif isinstance(date, datetime):
        dt = date.date()
    else:
        dt = date
    return df[df["InvoiceDay"] == pd.to_datetime(dt).date()]

# ─── Plotly dark theme helper ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#13131F",
    plot_bgcolor="#13131F",
    font=dict(family="Inter, sans-serif", color="#C8C8E0", size=12),
    title_font=dict(size=14, color="#E2E2F0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1E1E30", linecolor="#1E1E30", tickfont=dict(color="#7B7B9A")),
    yaxis=dict(gridcolor="#1E1E30", linecolor="#1E1E30", tickfont=dict(color="#7B7B9A")),
    legend=dict(bgcolor="#13131F", bordercolor="#1E1E30", borderwidth=1, font=dict(color="#9090B0")),
    margin=dict(l=16, r=16, t=48, b=16),
    hoverlabel=dict(bgcolor="#1A1A2E", bordercolor="#2A2A45", font=dict(color="#E2E2F0")),
)

def apply_layout(fig, title=None, height=520):
    layout = dict(PLOTLY_LAYOUT)
    if title:
        layout["title"] = title
    layout["height"] = height
    fig.update_layout(**layout)
    return fig


# Backwards-compatible alias for a common misspelling found at runtime
def plot_rule_netowkr(*args, **kwargs):
    return plot_rule_network(*args, **kwargs)

# ─── Force refresh ────────────────────────────────────────────────────────────
def force_refresh_all():
    st.cache_data.clear()
    st.session_state.last_loaded_forecast = None
    st.rerun()

with st.sidebar:
    if st.button("🔄  Force refresh"):
        force_refresh_all()

# ─── Page: Overview ───────────────────────────────────────────────────────────
def page_overview():
    df = load_transactions()
    st.title("📊 Online Retail Dashboard")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue (GBP)", f"£{df.TotalSales.sum():,.0f}")
    c2.metric("Total Orders",        f"{df.InvoiceNo.nunique():,}")
    c3.metric("Unique Customers",    f"{df.CustomerID.nunique():,}")
    
    total_revenue = df.TotalSales.sum()
    total_orders = df.InvoiceNo.nunique()
    total_customers = df.CustomerID.nunique()
    st.subheader("🤖 AI Business Insights")

    insights = []
    insights.append(
    f"Business generated £{total_revenue:,.0f} revenue from {total_orders:,} orders."
    )
    
    if total_orders / total_customers > 1.5:
        insights.append(
            "Customer repeat purchase behavior appears strong."
            )
        
        top_product = (
            df.groupby("Description")["TotalSales"]
            .sum()
            .idxmax()
            )
        
        insights.append(
            f"Top revenue-generating product is '{top_product}'."
            )
        
        avg_order_value = total_revenue / total_orders
        insights.append(
            f"Average order value is £{avg_order_value:.2f}."
            )
        for insight in insights:
            st.info(insight)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.subheader("Daily Sales Over Time")
    daily_ts = df.groupby("InvoiceDay")["TotalSales"].sum().reset_index()
    fig_line = px.line(daily_ts, x="InvoiceDay", y="TotalSales",
                       labels={"InvoiceDay": "Date", "TotalSales": "Sales (£)"})
    fig_line.update_traces(line=dict(color="#5B8AF5", width=2))
    apply_layout(fig_line, height=380)
    st.plotly_chart(fig_line, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Top 10 Products by Revenue")
        top10 = df.groupby("Description")["TotalSales"].sum().nlargest(10).reset_index()
        fig_bar = px.bar(top10, x="TotalSales", y="Description", orientation="h",
                         labels={"TotalSales": "Revenue (£)", "Description": ""},
                         color_discrete_sequence=["#5B8AF5"])
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        apply_layout(fig_bar, height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("Sales Heatmap (Weekday vs Hour)")
        df["Weekday"] = pd.Categorical(
            df.InvoiceDate.dt.day_name(),
            categories=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
            ordered=True
        )
        df["Hour"] = df.InvoiceDate.dt.hour
        heat = df.pivot_table(index="Weekday", columns="Hour", values="TotalSales", aggfunc="sum").fillna(0)
        fig_heat = go.Figure(go.Heatmap(
            z=heat.values, x=list(heat.columns), y=list(heat.index),
            colorscale=[[0,"#0A0A12"],[0.5,"#1E3A8A"],[1,"#5B8AF5"]],
            hovertemplate="Hour %{x}, %{y}<br>Sales: £%{z:,.0f}<extra></extra>",
            showscale=True
        ))
        apply_layout(fig_heat, height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

# ─── Page: Forecast & Anomalies ───────────────────────────────────────────────
def page_forecast():
    st.header("🔮 Forecast & Anomaly Alerts")
    daily, hist_fc, future_fc, hist, anoms, updated = load_precomputed_forecast()

    if daily is None:
        st.warning("Processed forecast artifacts not found. Run process_daily.py or set GCS_BUCKET.")
        return

    cutoff = daily["ds"].max() if "ds" in daily.columns else None
    hist_fc["type"], future_fc["type"] = "Fitted (history)", "Forecast (30d)"
    plot_df = pd.concat([hist_fc, future_fc])

    fig_fc = px.line(plot_df, x="ds", y="yhat", color="type",
                     color_discrete_map={"Fitted (history)": "#5B8AF5", "Forecast (30d)": "#F5A623"},
                     labels={"ds": "Date", "yhat": "Sales (£)"})
    fig_fc.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], mode="markers", name="Actual",
                                marker=dict(color="#FFFFFF", size=5), opacity=0.6))
    if cutoff is not None:
        fig_fc.add_shape(type="line", x0=cutoff, x1=cutoff, y0=0, y1=1,
                         xref="x", yref="paper", line=dict(color="#3D6ABA", dash="dash", width=1))
        fig_fc.add_annotation(x=cutoff, y=1.03, xref="x", yref="paper",
                              text="Cutoff", showarrow=False, font=dict(color="#7B7B9A", size=11))
    if "yhat_upper" in hist_fc.columns:
        fig_fc.add_traces([go.Scatter(
            x=hist_fc["ds"].tolist()+hist_fc["ds"].iloc[::-1].tolist(),
            y=hist_fc["yhat_upper"].tolist()+hist_fc["yhat_lower"].iloc[::-1].tolist(),
            fill="toself", fillcolor="rgba(91,138,245,0.12)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False)])
    if "yhat_upper" in future_fc.columns:
        fig_fc.add_traces([go.Scatter(
            x=future_fc["ds"].tolist()+future_fc["ds"].iloc[::-1].tolist(),
            y=future_fc["yhat_upper"].tolist()+future_fc["yhat_lower"].iloc[::-1].tolist(),
            fill="toself", fillcolor="rgba(245,166,35,0.12)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False)])
    apply_layout(fig_fc, title="Daily Sales Forecast", height=560)
    st.plotly_chart(fig_fc, use_container_width=True)
    

    st.subheader("📈 Forecast Summary")

    next_30d_sales = future_fc["yhat"].sum()

    best_day = future_fc.loc[
        future_fc["yhat"].idxmax()
    ]

    worst_day = future_fc.loc[
        future_fc["yhat"].idxmin()
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Expected 30-Day Revenue",
            f"£{next_30d_sales:,.0f}"
        )

    with c2:
        st.metric(
            "Highest Forecast Day",
            best_day["ds"].strftime("%d %b %Y")
        )

    with c3:
        st.metric(
            "Peak Day Sales",
            f"£{best_day['yhat']:,.0f}"
        )

    st.info(
        f"""
        📌 Forecast suggests the strongest sales day will be
        **{best_day['ds'].strftime('%d %b %Y')}**
        with expected revenue of
        **£{best_day['yhat']:,.0f}**.

        Lowest expected sales:
        **{worst_day['ds'].strftime('%d %b %Y')}**
        (£{worst_day['yhat']:,.0f})
        """
    )

    fig_ra = go.Figure()
    fig_ra.add_trace(go.Scatter(x=hist["ds"], y=hist["residual"], mode="lines", name="Residual",
                                line=dict(color="#5B8AF5", width=1.5)))
    fig_ra.add_trace(go.Scatter(x=anoms["ds"], y=anoms["residual"], mode="markers", name="Anomaly",
                                marker=dict(color="#E2504A", size=8, symbol="circle")))
    apply_layout(fig_ra, title="Residuals & Detected Anomalies", height=440)
    st.plotly_chart(fig_ra, use_container_width=True)

    st.subheader("Top 5 Anomalous Days")
    top5 = (anoms[["ds","y","yhat","residual"]]
            .sort_values("residual", key=abs, ascending=False).head(5)
            .rename(columns={"ds":"Date","y":"Actual","yhat":"Predicted","residual":"Residual"}))
    st.table(top5)

    st.subheader("🔍 Drill into Anomalous Transactions")
    anoms["DateOnly"] = anoms["ds"].dt.date
    top_pos = anoms[anoms["residual"]>0].nlargest(5,"residual")
    top_neg = anoms[anoms["residual"]<0].nsmallest(5,"residual")
    mode = st.radio("Choose anomaly type:", ("Above-Expectation","Below-Expectation"))
    choices = (top_pos["DateOnly"].tolist() if mode.startswith("Above") else top_neg["DateOnly"].tolist())
    if not choices:
        st.info("No anomalies found of the selected type.")
        return
    sel = st.selectbox("Select date to inspect:", choices)
    tx = get_transactions_for_date(sel)
    st.markdown(f"### Transactions on {sel}")
    st.markdown(
        f"- Number of orders: **{tx.InvoiceNo.nunique()}**  \n"
        f"- Total transactions: **{len(tx)}**  \n"
        f"- Total sales: **£{tx.TotalSales.sum():,.2f}**  \n"
        f"- Avg. order value: **£{tx.TotalSales.sum()/tx.InvoiceNo.nunique():,.2f}**"
    )
    st.dataframe(tx[["InvoiceNo","StockCode","Description","Quantity","UnitPrice","TotalSales"]],
                 use_container_width=True)

# ─── Page: Customer Segments ──────────────────────────────────────────────────
def page_customers():
    rfm_df, profile_df = load_rfm_and_profiles()
    st.header("👥 Customer Segmentation (RFM + K-Means)")

    CLUSTER_COLORS = {0: "#5B8AF5", 1: "#E2504A", 2: "#3DBA8C", 3: "#F5A623"}
    fig = px.scatter(rfm_df, x="Recency", y="Monetary", color="Cluster",
                     hover_data=["CustomerID","Frequency"],
                     color_discrete_map=CLUSTER_COLORS,
                     labels={"Recency": "Recency (days)", "Monetary": "Monetary (£)"})
    apply_layout(fig, title="Recency vs Monetary by Cluster", height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Profiles")
    st.table(profile_df)

    st.subheader("🎯 Target a Segment")
    seg_names = profile_df["SegmentName"].tolist()
    sel_name  = st.selectbox("Choose a segment:", seg_names)
    cid       = profile_df.loc[profile_df["SegmentName"]==sel_name,"Cluster"].iloc[0]
    seg_customers = rfm_df[rfm_df["Cluster"]==cid]
    prof      = profile_df.query("Cluster==@cid").iloc[0]
    st.markdown(f"**Segment:** {sel_name}")
    st.markdown(
        f"- **Avg Frequency:** {prof.AvgFrequency:.1f} orders  \n"
        f"- **Avg Monetary:** £{prof.AvgMonetary:,.2f}"
    )
    st.markdown("**Sample Customers:**")
    st.dataframe(seg_customers.head(10))
    if st.button(f"📧 Send Special Offer to {sel_name}"):
        st.success(f"Offer emailed to {int(prof.Count)} {sel_name}!")

# ─── Page: Product Segmentation ───────────────────────────────────────────────
def page_products():
    sku_df, profile_df = load_product_segments()
    st.header("📦 Product Segmentation")
    sku_df["SegmentName"] = sku_df["Segment"].map(profile_df.set_index("Segment")["SegmentName"])

    color_map = {
        "Steady Sellers":    "#5B8AF5",
        "Dead Stock":        "#E2504A",
        "Super Performers":  "#3DBA8C",
        "Niche Big-Tickets": "#F5A623",
    }
    fig = px.scatter(sku_df, x="UnitsSold", y="AvgPrice", color="SegmentName",
                     color_discrete_map=color_map,
                     hover_data=["StockCode","Description","TotalRevenue","Recency","SalesFreq"],
                     labels={"UnitsSold": "Units Sold", "AvgPrice": "Avg Price (£)", "SegmentName": "Segment"})
    apply_layout(fig, title="Units Sold vs Average Price by Segment", height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Profiles")
    st.table(profile_df.set_index("Segment")[["Count","AvgRevenue","AvgUnits","AvgRecency","AvgSalesDays","SegmentName"]])

    st.subheader("🔍 Explore SKUs in a Segment")
    seg_choice = st.selectbox("Choose a segment:", profile_df["SegmentName"])
    seg_id     = profile_df.loc[profile_df["SegmentName"]==seg_choice,"Segment"].iloc[0]
    seg_skus   = sku_df[sku_df["Segment"] == seg_id]
    st.markdown(f"**Showing {len(seg_skus)} SKUs in {seg_choice}**")
    st.dataframe(seg_skus[["StockCode","Description","TotalRevenue","UnitsSold","AvgPrice","Recency","SalesFreq"]],
                 use_container_width=True)

    interpretations = {
        "Steady Sellers":    "Consistent revenue and volume — keep these reliably in stock.",
        "Dead Stock":        "Old, slow-moving products — consider clearance or bundles.",
        "Super Performers":  "High-velocity sellers — prioritize availability and marketing.",
        "Niche Big-Tickets": "Rare but high-value items — promote in targeted channels."
    }
    st.info(f"**{seg_choice}**: {interpretations[seg_choice]}")



def page_market_basket():
    st.header("🛒 Market Basket / Association Rules")

    rules_df = load_rules()

    if rules_df.empty:
        st.warning("No association rules available.")
        return

    if "confidence" not in rules_df.columns:
        st.error(
            f"Confidence column not found. Available columns: {rules_df.columns.tolist()}"
        )
        return

    rules_df["confidence"] = pd.to_numeric(
        rules_df["confidence"],
        errors="coerce"
    )

    if "support" in rules_df.columns:
        rules_df["support"] = pd.to_numeric(
            rules_df["support"],
            errors="coerce"
        )

    if "lift" in rules_df.columns:
        rules_df["lift"] = pd.to_numeric(
            rules_df["lift"],
            errors="coerce"
        )

    rules_df = rules_df.dropna(subset=["confidence"])

    if rules_df.empty:
        st.warning("No valid rules found after cleaning.")
        return

    st.subheader("Top 10 Rules by Confidence")

    top10 = rules_df.nlargest(10, "confidence").copy()

    top10["antecedents"] = top10["antecedents"].apply(
        lambda s: ", ".join(sorted(s))
    )
    top10["consequents"] = top10["consequents"].apply(
        lambda s: ", ".join(sorted(s))
    )

    top10["support"] = top10["support"].round(3)
    top10["confidence"] = top10["confidence"].round(3)
    top10["lift"] = top10["lift"].round(2)

    display_df = top10[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].rename(
        columns={
            "antecedents": "If basket contains",
            "consequents": "Then also contains",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
        }
    )

    st.dataframe(display_df, use_container_width=True, height=300)

    st.subheader("💡 Get Recommendations")

    all_items = sorted(
        {
            item
            for ants in rules_df["antecedents"]
            for item in ants
        }
    )

    if not all_items:
        st.warning("No products available.")
        return

    selected = st.selectbox(
        "Select a product:",
        all_items
    )

    filtered_rules = rules_df[
        rules_df["antecedents"].apply(
            lambda ants: selected in ants
        )
    ].copy()


    if filtered_rules.empty:
        st.info(
            f"No strong associations found for {selected}."
        )
    else:
        if "confidence" not in filtered_rules.columns:
            st.error(
                f"Confidence column missing in filtered data. Columns: {filtered_rules.columns.tolist()}"
            )
            return

        recs = (
            filtered_rules
            .sort_values(
                "confidence",
                ascending=False
            )
            .head(5)
            .copy()
        )

        recs["consequents"] = recs["consequents"].apply(
            lambda s: ", ".join(sorted(s))
        )

        recs["confidence"] = recs["confidence"].round(3)
        recs["lift"] = recs["lift"].round(2)

        st.markdown(
            f"**Customers who bought {selected} also bought:**"
        )

        for _, r in recs.iterrows():
            st.markdown(
                f"- **{r['consequents']}** (conf={r['confidence']}, lift={r['lift']})"
            )

    st.subheader("🌐 Association Network")

    fig_net = plot_rule_network(
        rules_df,
        top_n=25
    )

    st.plotly_chart(
        fig_net,
        use_container_width=True
    )

def plot_rule_network(rules_df, top_n=25):

    rules_df["confidence"] = pd.to_numeric(
        rules_df["confidence"],
        errors="coerce"
    )

    rules_df = rules_df.dropna(subset=["confidence"])

    sub = rules_df.nlargest(top_n, "confidence")

    G = nx.DiGraph()

    for _, row in sub.iterrows():
        for ant in row["antecedents"]:
            for cons in row["consequents"]:
                G.add_node(ant)
                G.add_node(cons)
                G.add_edge(
                    ant,
                    cons,
                    weight=row["confidence"]
                )

    pos = nx.spring_layout(
        G,
        k=1,
        seed=42
    )

    edge_x, edge_y = [], []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="#1E3A8A", width=1),
        hoverinfo="none"
    )

    node_x, node_y, node_text = [], [], []

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        textfont=dict(
            color="#9090B0",
            size=10
        ),
        marker=dict(
            color="#5B8AF5",
            size=18,
            line=dict(
                color="#1E3A8A",
                width=1.5
            )
        )
    )

    fig = go.Figure([
        edge_trace,
        node_trace
    ])

    fig.update_layout(
        title="Association Rules Network",
        showlegend=False,
        hovermode="closest",
        margin=dict(
            b=20,
            l=5,
            r=5,
            t=40
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        paper_bgcolor="#13131F",
        plot_bgcolor="#13131F",
        font=dict(color="#C8C8E0"),
        height=560
    )

    return fig

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h3 style='color:#E2E2F0; margin-top:0.5rem; margin-bottom:1.2rem; font-size:1.2rem; letter-spacing:-0.01em;'>"
        "🔗 Navigation</h3>",
        unsafe_allow_html=True
    )
    page = option_menu(
        menu_title=None,
        options=["Overview","Forecast & Anomalies","Customer Segments","Product Segments","Market Basket Analysis"],
        icons=["house","graph-up","people-fill","box-seam","cart"],
        menu_icon="cast", default_index=0, orientation="vertical",
        styles={
            "container":         {"padding":"0px","background-color":"#0F0F1A","margin-top":"0px"},
            "nav-link":          {"font-size":"0.9rem","text-align":"left","margin":"0.15rem 0",
                                  "color":"#7B7B9A","border-radius":"8px","padding":"0.55rem 0.9rem",
                                  "--hover-color":"#1A1A2E"},
            "nav-link-selected": {"background-color":"#1A1A2E","font-weight":"600","color":"#E2E2F0",
                                  "border-left":"2px solid #5B8AF5"},
            "icon":              {"font-size":"1rem","color":"#5B8AF5"},
        }
    )
    st.markdown(
        "<div style='position:absolute;bottom:1rem;width:80%;text-align:center;"
        "color:#3A3A5A;font-size:0.75rem;'>Dashboard © Omar Medhat</div>",
        unsafe_allow_html=True
    )

# ─── Main ─────────────────────────────────────────────────────────────────────
if page == "Overview":
    page_overview()
elif page == "Forecast & Anomalies":
    page_forecast()
elif page == "Customer Segments":
    page_customers()
elif page == "Product Segments":
    page_products()
elif page == "Market Basket Analysis":
    page_market_basket()