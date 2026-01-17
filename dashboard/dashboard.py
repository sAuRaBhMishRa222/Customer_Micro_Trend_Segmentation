import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Customer Micro-Trend Intelligence",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CUSTOM CSS (NO OVERLAP)
# =====================================================
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1551288049-bebda4e38f71");
    background-size: cover;
    background-repeat: no-repeat;
}

/* Glass container */
.block-container {
    background: rgba(15, 15, 15, 0.70);
    backdrop-filter: blur(12px);
    padding: 2.5rem;
    border-radius: 18px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: rgba(0,0,0,0.75);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
}

/* Tabs spacing */
[data-testid="stTabs"] {
    margin-top: 30px;
}

/* Plot spacing */
.js-plotly-plot {
    margin-top: 25px;
}

/* Titles */
h1, h2, h3 {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/raw/transactions.csv")
    df_clusters = pd.read_csv("data/processed/customer_micro_trend_segments.csv")
    return df_raw, df_clusters

df, customer_df = load_data()

# =====================================================
# PREPROCESS TIME FEATURES
# =====================================================
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["hour"] = df["TransactionDate"].dt.hour
df["is_weekend"] = df["TransactionDate"].dt.dayofweek.isin([5, 6]).astype(int)

# =====================================================
# HEADER
# =====================================================
st.title("🧠 Customer Micro-Trend Intelligence Dashboard")
st.markdown("### Transforming Raw Transactions into Actionable Business Insights")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# KPI SECTION
# =====================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", customer_df["CustomerID"].nunique())
col2.metric("Total Transactions", df.shape[0])
col3.metric("Clusters Found", customer_df["dbscan_cluster"].nunique() - 1)
col4.metric("Outlier Customers", (customer_df["dbscan_cluster"] == -1).sum())

st.markdown("<br><br>", unsafe_allow_html=True)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Exploratory Analysis",
    "🧬 Clustering Insights",
    "💰 Revenue & Marketing"
])

# =====================================================
# TAB 1 — EDA
# =====================================================
with tab1:
    st.subheader("Purchases by Hour")

    hourly = df.groupby("hour").size().reset_index(name="count")
    fig = px.line(hourly, x="hour", y="count", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weekend vs Weekday Purchases")
        weekend = df["is_weekend"].value_counts().reset_index()
        weekend.columns = ["is_weekend", "count"]
        fig = px.bar(weekend, x="is_weekend", y="count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Order Value Distribution")
        fig = px.histogram(df, x="TotalAmount", nbins=50)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2 — CLUSTERING
# =====================================================
with tab2:
    st.subheader("Elbow Method (K-Means Baseline)")

    X = customer_df.drop(
        ["CustomerID", "dbscan_cluster", "marketing_action"], axis=1
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig = px.line(
        x=list(range(2, 10)),
        y=inertia,
        markers=True,
        labels={"x": "Number of Clusters", "y": "Inertia"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Micro-Trend Clusters (PCA Projection)")

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(pca_data, columns=["PCA_1", "PCA_2"])
    pca_df["Cluster"] = customer_df["dbscan_cluster"]

    fig = px.scatter(
        pca_df,
        x="PCA_1",
        y="PCA_2",
        color="Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 3 — REVENUE & MARKETING
# =====================================================
with tab3:
    st.subheader("Revenue Contribution per Cluster")

    df_rev = df.merge(
        customer_df[["CustomerID", "dbscan_cluster"]],
        on="CustomerID"
    )

    revenue = df_rev.groupby("dbscan_cluster")["TotalAmount"].sum().reset_index()

    fig = px.bar(
        revenue,
        x="dbscan_cluster",
        y="TotalAmount"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Marketing Recommendation Engine")

    st.dataframe(
        customer_df[
            ["CustomerID", "dbscan_cluster", "marketing_action"]
        ].head(25),
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<center>🚀 Built with Python, Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)