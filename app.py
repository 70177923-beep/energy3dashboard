import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from prophet import Prophet
from streamlit_option_menu import option_menu

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Global Energy Intelligence",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# SIMPLE STYLING
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.metric-card {
    padding: 1rem;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA LOAD
# --------------------------------------------------

@st.cache_data
def load_data():

    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

    df = pd.read_csv(url)

    return df


df = load_data()

# --------------------------------------------------
# COUNTRY LIST
# --------------------------------------------------

exclude = [
    "World",
    "Asia",
    "Europe",
    "Africa",
    "North America",
    "South America",
    "OECD"
]

countries = sorted([
    c for c in df["country"].dropna().unique()
    if c not in exclude
])

# --------------------------------------------------
# ESG SCORE
# --------------------------------------------------

def calculate_esg(row):

    renewable = row.get(
        "renewables_share_energy",
        0
    )

    fossil = row.get(
        "fossil_share_energy",
        0
    )

    if pd.isna(renewable):
        renewable = 0

    if pd.isna(fossil):
        fossil = 0

    score = (
        renewable * 0.6
        +
        (100 - fossil) * 0.4
    )

    return round(score, 1)

# --------------------------------------------------
# AI INSIGHT
# --------------------------------------------------

def generate_insight(country_df):

    if len(country_df) < 5:
        return "Not enough data"

    start = country_df.iloc[0]
    end = country_df.iloc[-1]

    try:

        renewable_change = (
            end["renewables_share_energy"]
            -
            start["renewables_share_energy"]
        )

        if renewable_change > 10:

            return (
                "Strong renewable growth detected."
            )

        elif renewable_change > 0:

            return (
                "Moderate renewable growth."
            )

        else:

            return (
                "Renewable growth is stagnant."
            )

    except:
        return "No insight available."

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚡ Energy Intel")

    page = option_menu(
        menu_title="Navigation",
        options=[
            "Executive Overview",
            "Country Intelligence",
            "Global Rankings",
            "AI Forecasting",
            "ESG & Sustainability"
        ],
        icons=[
            "speedometer2",
            "globe",
            "bar-chart",
            "graph-up",
            "leaf"
        ],
        default_index=0
    )

    st.markdown("---")

    selected_country = st.selectbox(
        "Country",
        countries,
        index=countries.index("Pakistan")
        if "Pakistan" in countries
        else 0
    )

# --------------------------------------------------
# COMMON DATA
# --------------------------------------------------

latest_year = int(df["year"].max())

latest = df[
    df["year"] == latest_year
].copy()

country_df = df[
    df["country"] == selected_country
].copy()

country_df = country_df.sort_values(
    "year"
)

st.title("⚡ Global Energy Intelligence Platform")

st.caption(
    "Powered by Our World in Data"
)
# ==================================================
# EXECUTIVE OVERVIEW
# ==================================================

if page == "Executive Overview":

    st.header("🌍 Executive Overview")

    total_energy = latest[
        "primary_energy_consumption"
    ].fillna(0).sum()

    renewable_pct = latest[
        "renewables_share_energy"
    ].fillna(0).mean()

    total_co2 = latest[
        "co2"
    ].fillna(0).sum() if "co2" in latest.columns else 0

    avg_esg = latest.apply(
        calculate_esg,
        axis=1
    ).mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "⚡ Total Energy",
            f"{total_energy:,.0f} TWh"
        )

    with c2:
        st.metric(
            "🌱 Renewable %",
            f"{renewable_pct:.1f}%"
        )

    with c3:
        st.metric(
            "🏭 CO₂ Emissions",
            f"{total_co2:,.0f}"
        )

    with c4:
        st.metric(
            "♻ ESG Score",
            f"{avg_esg:.1f}"
        )

    st.markdown("---")

    st.subheader("🌎 Global Energy Map")

    map_df = latest[
        ["country", "primary_energy_consumption"]
    ].dropna()

    fig_map = px.choropleth(
        map_df,
        locations="country",
        locationmode="country names",
        color="primary_energy_consumption",
        color_continuous_scale="Viridis"
    )

    fig_map.update_layout(
        height=550,
        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0
        )
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("⚡ Global Energy Trend")

    global_energy = (
        df.groupby("year")
        ["primary_energy_consumption"]
        .sum()
        .reset_index()
    )

    fig_energy = px.line(
        global_energy,
        x="year",
        y="primary_energy_consumption",
        markers=True
    )

    fig_energy.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_energy,
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🌱 Renewable Growth")

        renewable = (
            df.groupby("year")
            ["renewables_share_energy"]
            .mean()
            .reset_index()
        )

        fig_renew = px.area(
            renewable,
            x="year",
            y="renewables_share_energy"
        )

        fig_renew.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_renew,
            use_container_width=True
        )

    with col2:

        st.subheader("🏭 CO₂ Trend")

        if "co2" in df.columns:

            co2 = (
                df.groupby("year")
                ["co2"]
                .sum()
                .reset_index()
            )

            fig_co2 = px.line(
                co2,
                x="year",
                y="co2",
                markers=True,
                color_discrete_sequence=["red"]
            )

            fig_co2.update_layout(
                height=400
            )

            st.plotly_chart(
                fig_co2,
                use_container_width=True
            )

    st.markdown("---")

    st.subheader("🤖 AI Executive Insight")

    insight = f"""
    Global renewable energy share is currently
    {renewable_pct:.1f}%.

    Average sustainability score is
    {avg_esg:.1f}/100.

    Energy consumption continues to rise globally,
    while renewable adoption is improving.
    """

    st.info(insight)

# ==================================================
# COUNTRY INTELLIGENCE
# ==================================================

elif page == "Country Intelligence":

    st.header(f"📊 Country Intelligence: {selected_country}")

    c_df = country_df.copy()

    if len(c_df) == 0:
        st.warning("No data available for this country.")
        st.stop()

    latest_row = c_df.iloc[-1]

    esg = calculate_esg(latest_row)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "⚡ Energy",
            f"{latest_row.get('primary_energy_consumption',0):,.0f}"
        )

    with c2:
        st.metric(
            "🌱 Renewable %",
            f"{latest_row.get('renewables_share_energy',0):.1f}%"
        )

    with c3:
        st.metric(
            "♻ ESG Score",
            f"{esg}"
        )

    st.markdown("---")

    # -------------------------
    # ENERGY TREND
    # -------------------------

    st.subheader("⚡ Energy Trend")

    fig1 = px.line(
        c_df,
        x="year",
        y="primary_energy_consumption",
        markers=True
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # -------------------------
    # RENEWABLE TREND
    # -------------------------

    st.subheader("🌱 Renewable Trend")

    if "renewables_share_energy" in c_df.columns:

        fig2 = px.area(
            c_df,
            x="year",
            y="renewables_share_energy"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # -------------------------
    # CO2 TREND
    # -------------------------

    st.subheader("🏭 CO₂ Trend")

    if "co2" in c_df.columns:

        fig3 = px.line(
            c_df,
            x="year",
            y="co2",
            color_discrete_sequence=["red"]
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # -------------------------
    # ENERGY MIX (SIMPLE VIEW)
    # -------------------------

    st.subheader("⚡ Energy Mix Snapshot")

    mix_cols = [
        "coal_share_energy",
        "oil_share_energy",
        "gas_share_energy",
        "renewables_share_energy"
    ]

    mix_data = {}

    for col in mix_cols:

        if col in c_df.columns:

            mix_data[col.replace("_share_energy","")] = c_df[col].iloc[-1]

    if mix_data:

        fig4 = px.pie(
            values=list(mix_data.values()),
            names=list(mix_data.keys()),
            hole=0.5
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    # -------------------------
    # AI INSIGHT
    # -------------------------

    st.subheader("🤖 AI Country Insight")

    insight = generate_insight(c_df)

    st.info(
        f"""
        {selected_country} Analysis:

        {insight}

        ESG Score: {esg}/100
        """
    )

# ==================================================
# GLOBAL RANKINGS
# ==================================================

elif page == "Global Rankings":

    st.header("🏆 Global Rankings")

    latest_clean = latest.copy()

    st.subheader("⚡ Top Energy Consumers")

    top_energy = latest_clean.sort_values(
        "primary_energy_consumption",
        ascending=False
    ).head(10)

    fig1 = px.bar(
        top_energy,
        x="primary_energy_consumption",
        y="country",
        orientation="h",
        color="primary_energy_consumption",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.dataframe(
        top_energy[[
            "country",
            "primary_energy_consumption"
        ]]
    )

    st.markdown("---")

    # -------------------------
    # TOP RENEWABLE COUNTRIES
    # -------------------------

    st.subheader("🌱 Top Renewable Countries")

    if "renewables_share_energy" in latest_clean.columns:

        top_renew = latest_clean.sort_values(
            "renewables_share_energy",
            ascending=False
        ).head(10)

        fig2 = px.bar(
            top_renew,
            x="renewables_share_energy",
            y="country",
            orientation="h",
            color="renewables_share_energy",
            color_continuous_scale="Greens"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            top_renew[[
                "country",
                "renewables_share_energy"
            ]]
        )

    st.markdown("---")

    # -------------------------
    # TOP EMITTERS
    # -------------------------

    st.subheader("🏭 Top CO₂ Emitters")

    if "co2" in latest_clean.columns:

        top_co2 = latest_clean.sort_values(
            "co2",
            ascending=False
        ).head(10)

        fig3 = px.bar(
            top_co2,
            x="co2",
            y="country",
            orientation="h",
            color="co2",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(
            top_co2[[
                "country",
                "co2"
            ]]
        )

    st.markdown("---")

    # -------------------------
    # ESG RANKING
    # -------------------------

    st.subheader("♻ ESG Leaders")

    latest_clean["ESG"] = latest_clean.apply(
        calculate_esg,
        axis=1
    )

    top_esg = latest_clean.sort_values(
        "ESG",
        ascending=False
    ).head(10)

    fig4 = px.bar(
        top_esg,
        x="ESG",
        y="country",
        orientation="h",
        color="ESG",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(
        top_esg[[
            "country",
            "ESG"
        ]]
    )

# ==================================================
# AI FORECASTING (PROPHET)
# ==================================================

elif page == "AI Forecasting":

    st.header("🔮 AI Energy Forecasting")

    f_df = country_df[[
        "year",
        "primary_energy_consumption"
    ]].dropna()

    if len(f_df) < 10:
        st.warning("Not enough data for forecasting.")
        st.stop()

    # -------------------------
    # Prepare data for Prophet
    # -------------------------

    prophet_df = f_df.rename(
        columns={
            "year": "ds",
            "primary_energy_consumption": "y"
        }
    )

    prophet_df["ds"] = pd.to_datetime(
        prophet_df["ds"].astype(str)
    )

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=10,
        freq="Y"
    )

    forecast = model.predict(future)

    # -------------------------
    # Plot forecast
    # -------------------------

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=prophet_df["ds"],
            y=prophet_df["y"],
            name="Actual",
            line=dict(color="blue")
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Forecast",
            line=dict(color="purple", dash="dash")
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="Confidence Interval",
            fillcolor="rgba(128,0,128,0.2)"
        )
    )

    fig1.update_layout(
        height=500,
        title=f"{selected_country} Energy Forecast"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # -------------------------
    # METRICS
    # -------------------------

    last_actual = prophet_df["y"].iloc[-1]
    future_pred = forecast["yhat"].iloc[-1]

    change_pct = (
        (future_pred - last_actual)
        / last_actual
    ) * 100

    risk = (
        "High"
        if abs(change_pct) > 20
        else "Medium"
        if abs(change_pct) > 10
        else "Low"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "📊 Current Value",
            f"{last_actual:,.0f}"
        )

    with c2:
        st.metric(
            "🔮 Forecast",
            f"{future_pred:,.0f}"
        )

    with c3:
        st.metric(
            "⚠ Risk Level",
            risk
        )

    # -------------------------
    # INSIGHT
    # -------------------------

    st.subheader("🤖 AI Insight")

    st.info(
        f"""
        {selected_country} is expected to change by
        {change_pct:.2f}% in next 10 years.

        Risk level is classified as {risk}.

        Forecast is based on historical energy trends
        using Facebook Prophet model.
        """
    )

# ==================================================
# ESG & SUSTAINABILITY
# ==================================================

elif page == "ESG & Sustainability":

    st.header("🌿 ESG & Sustainability Dashboard")

    c_df = country_df.copy()

    if len(c_df) == 0:
        st.warning("No data available for this country.")
        st.stop()

    # -------------------------
    # ESG SCORE OVER TIME
    # -------------------------

    c_df["ESG"] = c_df.apply(calculate_esg, axis=1)

    latest_esg = c_df["ESG"].iloc[-1]
    prev_esg = c_df["ESG"].iloc[-2] if len(c_df) > 1 else latest_esg
    esg_delta = round(latest_esg - prev_esg, 1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "♻ ESG Score",
            f"{latest_esg}",
            delta=f"{esg_delta:+.1f} vs last year"
        )

    with col2:
        renewable_val = c_df["renewables_share_energy"].iloc[-1] if "renewables_share_energy" in c_df.columns else 0
        st.metric(
            "🌱 Renewable Share",
            f"{renewable_val:.1f}%"
        )

    with col3:
        fossil_val = c_df["fossil_share_energy"].iloc[-1] if "fossil_share_energy" in c_df.columns else 0
        st.metric(
            "🏭 Fossil Share",
            f"{fossil_val:.1f}%"
        )

    st.markdown("---")

    st.subheader("📈 ESG Score Over Time")

    fig_esg = px.line(
        c_df,
        x="year",
        y="ESG",
        markers=True,
        color_discrete_sequence=["#22c55e"]
    )

    fig_esg.update_layout(
        height=400,
        yaxis_title="ESG Score"
    )

    st.plotly_chart(fig_esg, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🌱 Renewable vs Fossil Over Time")

        if "renewables_share_energy" in c_df.columns and "fossil_share_energy" in c_df.columns:

            compare_df = c_df[["year", "renewables_share_energy", "fossil_share_energy"]].dropna()

            fig_compare = px.line(
                compare_df,
                x="year",
                y=["renewables_share_energy", "fossil_share_energy"],
                markers=True,
                color_discrete_map={
                    "renewables_share_energy": "#22c55e",
                    "fossil_share_energy": "#ef4444"
                }
            )

            fig_compare.update_layout(
                height=400,
                legend_title="Energy Type"
            )

            st.plotly_chart(fig_compare, use_container_width=True)

    with col2:

        st.subheader("🥧 Latest Energy Mix")

        mix_cols = [
            "coal_share_energy",
            "oil_share_energy",
            "gas_share_energy",
            "renewables_share_energy",
            "nuclear_share_energy"
        ]

        mix_data = {}

        for col in mix_cols:
            if col in c_df.columns:
                val = c_df[col].iloc[-1]
                if not pd.isna(val) and val > 0:
                    mix_data[col.replace("_share_energy", "").capitalize()] = val

        if mix_data:

            fig_pie = px.pie(
                values=list(mix_data.values()),
                names=list(mix_data.keys()),
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            fig_pie.update_layout(height=400)

            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # GLOBAL ESG COMPARISON
    # -------------------------

    st.subheader("🌍 Global ESG Comparison")

    latest_all = df[df["year"] == latest_year].copy()
    latest_all["ESG"] = latest_all.apply(calculate_esg, axis=1)

    top_esg = latest_all.sort_values("ESG", ascending=False).head(15)

    # Highlight selected country
    top_esg["color"] = top_esg["country"].apply(
        lambda x: "Selected" if x == selected_country else "Other"
    )

    fig_global = px.bar(
        top_esg,
        x="ESG",
        y="country",
        orientation="h",
        color="color",
        color_discrete_map={
            "Selected": "#f59e0b",
            "Other": "#22c55e"
        }
    )

    fig_global.update_layout(
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # ESG INSIGHT
    # -------------------------

    st.subheader("🤖 AI Sustainability Insight")

    insight = generate_insight(c_df)

    esg_rating = (
        "Excellent 🟢" if latest_esg >= 60
        else "Good 🟡" if latest_esg >= 40
        else "Needs Improvement 🔴"
    )

    st.info(
        f"""
        {selected_country} — Sustainability Analysis

        ESG Score: {latest_esg}/100 — Rating: {esg_rating}

        Renewable Trend: {insight}

        Renewable Share: {renewable_val:.1f}% | Fossil Share: {fossil_val:.1f}%

        Higher ESG scores reflect greater adoption of
        clean energy and lower dependence on fossil fuels.
        """
    )
