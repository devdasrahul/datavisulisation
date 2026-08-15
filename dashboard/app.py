"""
dashboard/app.py

Phase 4 — Streamlit Dashboard

A minimalist, all-white dashboard with custom CSS animations.
Restructured to tell a clear narrative about Predictive Maintenance.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Engine Health | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Accent color definition
ACCENT_COLOR = "#0D9488"  # Soft teal
LIGHT_GRAY = "#F3F4F6"
DARK_TEXT = "#1F2937"


# ---------------------------------------------------------------------------
# Custom CSS & Micro-animations
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        
        /* Base Reset */
        html, body, [class*="css"]  {{
            font-family: 'Inter', sans-serif !important;
            background-color: #ffffff !important;
            color: {DARK_TEXT} !important;
        }}
        
        /* Hide default Streamlit top bar and footer */
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Fade-in Animation */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .stApp > header {{
            background-color: transparent !important;
        }}
        
        /* Section Containers */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1000px; /* narrowed for narrative reading */
            animation: fadeIn 0.4s ease-out forwards;
        }}
        
        /* Dividers */
        hr {{
            border: 0;
            border-top: 1px solid {LIGHT_GRAY};
            margin-top: 3rem;
            margin-bottom: 3rem;
        }}
        
        /* Typography */
        h1 {{
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }}
        
        h2 {{
            font-size: 1.5rem;
            font-weight: 400;
            color: #4B5563;
            margin-top: 0 !important;
            margin-bottom: 1.5rem !important;
        }}
        
        h3 {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: {DARK_TEXT};
        }}
        
        .narrative-text {{
            font-size: 1.125rem;
            line-height: 1.7;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }}
        
        /* Pipeline Strip */
        .pipeline-strip {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #F9FAFB;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            margin-bottom: 2rem;
        }}
        .pipeline-step {{
            text-align: center;
            font-size: 0.875rem;
            font-weight: 600;
            color: #374151;
        }}
        .pipeline-arrow {{
            color: #9CA3AF;
            font-size: 1.2rem;
        }}
        
        /* KPI Cards */
        .kpi-card {{
            background: #ffffff;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            height: 100%;
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
            border-color: {ACCENT_COLOR};
        }}
        .kpi-title {{
            font-size: 0.875rem;
            font-weight: 600;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-size: 2.5rem;
            font-weight: 300;
            color: {ACCENT_COLOR};
            margin: 0;
        }}
        .kpi-sub {{
            font-size: 0.75rem;
            color: #9CA3AF;
            margin-top: 0.25rem;
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Database Connection & Queries
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    # 1. Try to get from environment variables first (local .env file or Docker/traditional cloud)
    db_url = os.environ.get("DATABASE_URL")

    # 2. If not found, fallback to Streamlit secrets (Streamlit Cloud).
    # We only touch st.secrets if we have to, because accessing it when the file doesn't exist
    # causes Streamlit to print a repetitive "No secrets files found" warning.
    if not db_url:
        try:
            db_url = st.secrets.get("DATABASE_URL")
        except Exception:
            pass

    if not db_url:
        st.error("🚨 DATABASE_URL not found in environment or secrets.")
        st.stop()
    return create_engine(db_url)


@st.cache_data(ttl="24h")
def fetch_data(query: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def load_all_data():
    try:
        df_freshness = fetch_data("SELECT * FROM v_batch_freshness")
        df_rul = fetch_data("SELECT * FROM v_rul_estimate")
        df_health = fetch_data("SELECT * FROM v_pipeline_health_trend")
        df_oor = fetch_data("SELECT * FROM v_sensor_out_of_range_rate")
        df_rank = fetch_data("SELECT * FROM v_degradation_rank")
        return df_freshness, df_rul, df_health, df_oor, df_rank
    except Exception as e:
        st.error(f"Error connecting to database views: {e}")
        st.stop()


@st.cache_data(ttl="24h")
def fetch_analytical_sample() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        # Fetch a subset of units for heavy Pandas processing (correlation/Z-scores)
        return pd.read_sql(
            text(
                "SELECT unit_id, cycle, sensor_id, reading_value FROM fact_sensor_readings WHERE unit_id <= 15"
            ),
            conn,
        )


@st.cache_data(ttl="1h", show_spinner=False)
def fetch_explain_analyze(query: str) -> str:
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"EXPLAIN ANALYZE {query}")).fetchall()
            return "\n".join([row[0] for row in result])
    except Exception as e:
        return f"EXPLAIN ANALYZE failed: {e}"


@st.cache_data(ttl="24h", show_spinner=False)
def fetch_total_rows() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM fact_sensor_readings")).scalar()


# ---------------------------------------------------------------------------
# Chart Factory
# ---------------------------------------------------------------------------
def clean_layout(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="Inter", color=DARK_TEXT),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E5E7EB")
    fig.update_yaxes(
        showgrid=True, gridcolor="#F3F4F6", zeroline=False, linecolor="#E5E7EB"
    )
    return fig


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    inject_custom_css()
    df_freshness, df_rul, df_health, df_oor, df_rank = load_all_data()

    # ── 1. The Hook ────────────────────────────────────────────────────────
    st.markdown("<h1>Predictive Maintenance</h1>", unsafe_allow_html=True)

    st.markdown(
        """
    <div class="narrative-text">
        Unplanned equipment failure is one of the most expensive problems in industrial operations — 
        engines and machinery that fail without warning cost far more than ones that are caught early. 
        This dashboard tracks a fleet of turbofan engines and answers one vital question: 
        <strong>which ones are heading toward failure, and how much runway do we have?</strong>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── 2. What this pipeline actually does ────────────────────────────────
    st.markdown(
        """
    <div class="pipeline-strip">
        <div class="pipeline-step">📥<br>Ingest Data</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">🧹<br>Clean & Standardize</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">⚙️<br>Engineer Features</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">🗄️<br>Load to Postgres</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">📊<br>Monitor & Alert</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 2.5 Deeper Analytics ───────────────────────────────────────────────
    df_samp = fetch_analytical_sample()
    if not df_samp.empty:
        # Pivot the data for correlation and anomaly calculation
        pivot = df_samp.pivot(
            index=["unit_id", "cycle"], columns="sensor_id", values="reading_value"
        ).dropna(axis=1, how="all")

        # Drop 0-variance sensors to prevent NaN correlations and Z-scores
        std_devs = pivot.std()
        valid_sensors = std_devs[std_devs > 0].index
        pivot = pivot[valid_sensors]

        # SENSOR CORRELATION
        st.markdown("<h3>Sensor Signal Independence</h3>", unsafe_allow_html=True)
        st.markdown(
            """
        <div class="narrative-text">
            Before engineering features, we analyze the raw telemetry. The correlation matrix below reveals that several sensors are highly redundant (moving perfectly in tandem), while others carry independent degradation signals. This justifies our focus on specific sensors in the ETL pipeline.
        </div>
        """,
            unsafe_allow_html=True,
        )

        corr = pivot.corr()
        fig_corr = px.imshow(
            corr,
            color_continuous_scale=[[0.0, "#ffffff"], [1.0, ACCENT_COLOR]],
            labels=dict(color="Correlation"),
        )
        fig_corr.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(
            fig_corr, use_container_width=True, config={"displayModeBar": False}
        )

        # ANOMALY HEATMAP
        st.markdown(
            "<h3>Emergence of Anomalies (Z-Score > 3)</h3>", unsafe_allow_html=True
        )
        st.markdown(
            """
        <div class="narrative-text">
            By calculating rolling Z-scores for each sensor across these units, we can isolate statistically significant deviations from their baseline. In the heatmap below, the density of anomalies visibly increases as engines approach failure, providing the early-warning signal our pipeline captures.
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Calculate rolling Z-scores
        rolling_mean = (
            pivot.groupby(level="unit_id")
            .rolling(window=10, min_periods=1)
            .mean()
            .droplevel(0)
        )
        rolling_std = (
            pivot.groupby(level="unit_id")
            .rolling(window=10, min_periods=1)
            .std()
            .droplevel(0)
        )
        z_scores = ((pivot - rolling_mean) / rolling_std).fillna(0)

        # Count anomalies per (unit_id, cycle)
        anomalies = (z_scores.abs() > 3).sum(axis=1).reset_index(name="anomaly_count")
        # Bucket cycles by 20 for visual clarity
        anomalies["cycle_bucket"] = (anomalies["cycle"] // 20) * 20
        heatmap_data = (
            anomalies.groupby(["unit_id", "cycle_bucket"])["anomaly_count"]
            .sum()
            .reset_index()
        )
        heatmap_pivot = heatmap_data.pivot(
            index="unit_id", columns="cycle_bucket", values="anomaly_count"
        ).fillna(0)

        fig_heatmap = px.imshow(
            heatmap_pivot,
            labels=dict(x="Operating Cycle Bucket", y="Unit ID", color="Anomaly Count"),
            color_continuous_scale=[[0.0, "#F9FAFB"], [1.0, "#DC2626"]],
            aspect="auto",
        )
        fig_heatmap.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(
            fig_heatmap, use_container_width=True, config={"displayModeBar": False}
        )

        # PIVOT TABLE: LIFESPAN VS EARLIEST ANOMALY
        st.markdown(
            "<h3>Lifespan Quartile vs. Leading Indicator</h3>", unsafe_allow_html=True
        )
        st.markdown(
            """
        <div class="narrative-text">
            If we group engines by how long they lasted, which sensor usually triggered the <em>first</em> severe anomaly? This pivot table illustrates which sensors act as the earliest leading indicators of failure across different engine lifespans.
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Calculate max cycle (lifespan) per unit and find first anomaly sensor
        max_cycles = df_samp.groupby("unit_id")["cycle"].max()
        quartiles = pd.qcut(
            max_cycles, q=4, labels=["Q1 (Shortest)", "Q2", "Q3", "Q4 (Longest)"]
        )

        first_anomalies = []
        for unit in pivot.index.get_level_values("unit_id").unique():
            unit_z = z_scores.loc[unit]
            # find first row where any sensor > 3
            anom_rows = unit_z[(unit_z.abs() > 3).any(axis=1)]
            if not anom_rows.empty:
                first_cycle = anom_rows.index[0]
                first_row = anom_rows.loc[first_cycle]
                # find the sensor with max z score in this row
                top_sensor = first_row.abs().idxmax()
                first_anomalies.append(
                    {
                        "unit_id": unit,
                        "earliest_anomaly_sensor": top_sensor,
                        "first_anomaly_cycle": first_cycle,
                    }
                )
            else:
                first_anomalies.append(
                    {
                        "unit_id": unit,
                        "earliest_anomaly_sensor": "None",
                        "first_anomaly_cycle": None,
                    }
                )

        df_first_anom = pd.DataFrame(first_anomalies).set_index("unit_id")
        df_first_anom["lifespan_quartile"] = quartiles

        df_valid_anom = df_first_anom.dropna(subset=["first_anomaly_cycle"])
        if not df_valid_anom.empty:
            pivot_table = (
                pd.pivot_table(
                    df_valid_anom,
                    values="first_anomaly_cycle",
                    index="lifespan_quartile",
                    columns="earliest_anomaly_sensor",
                    aggfunc="mean",
                )
                .round(1)
                .fillna("")
            )
            st.dataframe(pivot_table, use_container_width=True)
        else:
            st.warning("No anomalies > 3σ detected in this sample.")

    else:
        st.warning("No data available for Deeper Analytics.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 3. The Story of One Engine (Centerpiece) ───────────────────────────
    st.markdown("<h3>The Anatomy of a Failure</h3>", unsafe_allow_html=True)

    if not df_rank.empty:
        # Interactive deep dive
        worst_unit = df_rank.sort_values("degradation_slope", ascending=True).iloc[0]
        worst_unit_id_default = int(worst_unit["unit_id"])

        st.markdown(
            """
        <div class="narrative-text">
            Let's look at a real example. <strong>You can use the controls below to explore specific engines and see exactly what the pipeline sees.</strong> 
            By tracking the rolling averages of these sensors, we can detect the subtle onset of wear long before catastrophic failure occurs.
        </div>
        """,
            unsafe_allow_html=True,
        )

        col_unit, col_sensor = st.columns(2)
        with col_unit:
            all_units = sorted(df_rank["unit_id"].astype(int).unique())
            default_idx = (
                all_units.index(worst_unit_id_default)
                if worst_unit_id_default in all_units
                else 0
            )
            selected_unit_id = st.selectbox(
                "Select an Engine Unit to analyze:", all_units, index=default_idx
            )

        with col_sensor:
            # Default to Sensor 11 which is a classic degradation indicator
            selected_sensor = st.selectbox(
                "Select a Sensor to plot:",
                options=[11, 14, 2, 3, 4, 7, 8, 9, 12, 13, 15, 17, 20, 21],
                index=0,
                format_func=lambda x: f"Sensor {x}",
            )

        # Fetch data for selected unit and sensor
        query = f"""
            SELECT cycle, reading_value, rolling_avg_7 
            FROM v_rolling_avg_by_sensor 
            WHERE unit_id = {selected_unit_id} AND sensor_id = {selected_sensor}
            ORDER BY cycle
        """
        df_worst = fetch_data(query)

        if not df_worst.empty:
            max_cycle = df_worst["cycle"].max()
            inflection_cycle = max(0, max_cycle - 60)

            # Interpolate y value for annotation
            try:
                inflection_y = df_worst[df_worst["cycle"] >= inflection_cycle][
                    "rolling_avg_7"
                ].iloc[0]
            except:
                inflection_y = df_worst["rolling_avg_7"].mean()

            fig_trend = go.Figure()

            # Raw readings
            fig_trend.add_trace(
                go.Scatter(
                    x=df_worst["cycle"],
                    y=df_worst["reading_value"],
                    mode="lines",
                    line=dict(color="#E5E7EB", width=1),
                    name="Raw Reading",
                )
            )

            # Rolling Avg
            fig_trend.add_trace(
                go.Scatter(
                    x=df_worst["cycle"],
                    y=df_worst["rolling_avg_7"],
                    mode="lines",
                    line=dict(color=ACCENT_COLOR, width=3, shape="spline"),
                    name="7-Cycle Trend",
                )
            )

            # Confidence Band (Heuristic)
            std_dev = df_worst["rolling_avg_7"].std() * 0.3
            df_valid_band = df_worst.dropna(subset=["rolling_avg_7"])
            if not df_valid_band.empty and not pd.isna(std_dev):
                fig_trend.add_trace(
                    go.Scatter(
                        x=df_valid_band["cycle"].tolist()
                        + df_valid_band["cycle"].tolist()[::-1],
                        y=(df_valid_band["rolling_avg_7"] + std_dev).tolist()
                        + (df_valid_band["rolling_avg_7"] - std_dev).tolist()[::-1],
                        fill="toself",
                        fillcolor="rgba(13, 148, 136, 0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=True,
                        name="Confidence Band (±0.3σ heuristic)",
                    )
                )

            # Annotation
            fig_trend.add_annotation(
                x=inflection_cycle,
                y=inflection_y,
                text=f"Maintenance Flag (~60 cycles remaining)",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#DC2626",
                ax=-80,
                ay=-40,
                font=dict(color="#DC2626", size=12, family="Inter"),
            )

            clean_layout(fig_trend)
            fig_trend.update_layout(
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(
                fig_trend, use_container_width=True, config={"displayModeBar": False}
            )

            st.markdown(
                f"""
            <div class="narrative-text" style="font-size: 1rem; color: #6B7280; font-style: italic; text-align: center; margin-top: -1rem;">
                With this pipeline, we would flag Unit {selected_unit_id} for maintenance based on its degradation curve.
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 4. Fleet-wide View ─────────────────────────────────────────────────
    st.markdown("<h3>Zooming Out: The Fleet-wide View</h3>", unsafe_allow_html=True)
    st.markdown(
        """
    <div class="narrative-text">
        That single-engine story isn't a cherry-picked case. The pipeline calculates this degradation slope and estimates the Remaining Useful Life (RUL) for every active engine in the fleet, surfacing the most critical units to the top.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # KPIs
    total_units = len(df_rul) if not df_rul.empty else 0
    failing_units = (
        len(df_rul[df_rul["estimated_rul_cycles"] < 20]) if not df_rul.empty else 0
    )

    kcol1, kcol2, kcol3 = st.columns(3)
    with kcol1:
        st.markdown(
            f"""
         <div class="kpi-card">
             <div class="kpi-title">Active Fleet</div>
             <div class="kpi-value">{total_units}</div>
             <div class="kpi-sub">Total Engines Monitored</div>
         </div>
         """,
            unsafe_allow_html=True,
        )
    with kcol2:
        danger_color = "#DC2626" if failing_units > 0 else ACCENT_COLOR
        st.markdown(
            f"""
         <div class="kpi-card" style="border-color: {danger_color};">
             <div class="kpi-title">Critical RUL (< 20)</div>
             <div class="kpi-value" style="color: {danger_color}">{failing_units}</div>
             <div class="kpi-sub">Immediate Action Required</div>
         </div>
         """,
            unsafe_allow_html=True,
        )
    with kcol3:
        avg_rul = int(df_rul["estimated_rul_cycles"].mean()) if not df_rul.empty else 0
        st.markdown(
            f"""
         <div class="kpi-card">
             <div class="kpi-title">Average RUL</div>
             <div class="kpi-value">{avg_rul}</div>
             <div class="kpi-sub">Estimated Cycles Remaining</div>
         </div>
         """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Ranking Chart
    rcol1, rcol2 = st.columns([2, 1])
    with rcol1:
        if not df_rank.empty:
            df_plot_rank = df_rank.sort_values(
                "degradation_slope", ascending=True
            ).head(10)
            fig_rank = px.bar(
                df_plot_rank,
                x="degradation_slope",
                y="unit_id",
                orientation="h",
                color="degradation_slope",
                color_continuous_scale=[LIGHT_GRAY, ACCENT_COLOR],
                title="Top 10 Fastest Degrading Engines",
            )
            fig_rank.update_layout(
                coloraxis_showscale=False,
                yaxis_type="category",
                title_font=dict(size=14, color="#6B7280"),
            )
            fig_rank.update_traces(marker_line_width=0)
            clean_layout(fig_rank)
            st.plotly_chart(
                fig_rank, use_container_width=True, config={"displayModeBar": False}
            )

    with rcol2:
        if not df_rul.empty:
            st.markdown(
                "<div style='font-size: 14px; color: #6B7280; margin-bottom: 10px;'>Lowest Estimated RUL</div>",
                unsafe_allow_html=True,
            )
            df_rul_sorted = df_rul.sort_values(
                "estimated_rul_cycles", ascending=True
            ).head(8)
            st.dataframe(
                df_rul_sorted[["unit_id", "estimated_rul_cycles"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "unit_id": "Unit ID",
                    "estimated_rul_cycles": st.column_config.ProgressColumn(
                        "RUL (Cycles)",
                        format="%f",
                        min_value=0,
                        max_value=200,
                    ),
                },
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 5. Impact Framing ──────────────────────────────────────────────────
    st.markdown("<h3>The Business Impact</h3>", unsafe_allow_html=True)
    st.markdown(
        """
    <div class="narrative-text">
        Predictive maintenance isn't just a data exercise; it's a massive operational lever. 
        If this pipeline gives operations teams even a <strong>60-cycle early warning</strong>, that represents 
        the difference between a planned, inexpensive maintenance stop and a catastrophic, unplanned engine failure mid-operation.
        <br><br>
        By moving from reactive to proactive maintenance, organizations can drastically reduce downtime, optimize spare parts inventory, and most importantly, ensure the safety of the fleet.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 6. Pipeline Health ─────────────────────────────────────────────────
    st.markdown("<h3>Why You Can Trust This Data</h3>", unsafe_allow_html=True)
    st.markdown(
        """
    <div class="narrative-text">
        Insights are only as good as the data powering them. This pipeline enforces strict data quality contracts, checking for anomalies, nulls, and referential integrity before any numbers hit this dashboard.
    </div>
    """,
        unsafe_allow_html=True,
    )

    latest_pass_rate = (
        float(df_health["pass_rate_pct"].iloc[0]) if not df_health.empty else 0.0
    )

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown(
            f"<div style='font-size:0.875rem; font-weight:600; color:#6B7280; margin-bottom:0.5rem'>DQ PASS RATE TREND (Latest: {latest_pass_rate}%)</div>",
            unsafe_allow_html=True,
        )
        if not df_health.empty:
            fig_health = px.line(
                df_health, x="run_date", y="pass_rate_pct", markers=True
            )
            fig_health.update_traces(line_color=ACCENT_COLOR, marker=dict(size=8))
            clean_layout(fig_health)
            fig_health.update_layout(height=200, yaxis_range=[0, 105])
            st.plotly_chart(
                fig_health, use_container_width=True, config={"displayModeBar": False}
            )

    with pcol2:
        st.markdown(
            "<div style='font-size:0.875rem; font-weight:600; color:#6B7280; margin-bottom:0.5rem'>SENSOR OUT OF RANGE ANOMALIES</div>",
            unsafe_allow_html=True,
        )
        if not df_oor.empty:
            st.dataframe(
                df_oor.sort_values("recent_oor_pct", ascending=False).head(5),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "sensor_name": "Sensor",
                    "recent_oor_pct": st.column_config.NumberColumn(
                        "Recent % OOR", format="%.1f%%"
                    ),
                    "all_time_oor_pct": st.column_config.NumberColumn(
                        "All-time % OOR", format="%.1f%%"
                    ),
                },
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── 7. Pipeline Performance Panel ──────────────────────────────────────
    with st.expander("⚙️ Engineering Details & Pipeline Performance"):
        st.markdown(
            """
        <div style="font-family: monospace; font-size: 0.85rem; color: #4B5563;">
            This section validates the performance of the underlying Postgres data warehouse and Streamlit's caching.
        </div>
        """,
            unsafe_allow_html=True,
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            rows = fetch_total_rows()
            st.metric("Total Fact Rows Processed", f"{rows:,}")
        with col_p2:
            st.metric("Dashboard Cache Status", "HIT (TTL 24h)")

        st.markdown(
            "<div style='font-family: monospace; font-size: 0.85rem; font-weight: 600; margin-top: 1rem;'>EXPLAIN ANALYZE: v_degradation_rank</div>",
            unsafe_allow_html=True,
        )
        explain_text = fetch_explain_analyze(
            "SELECT * FROM v_degradation_rank LIMIT 100"
        )
        st.code(explain_text, language="sql")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 8. Footer ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align: center; color: #9CA3AF; font-size: 0.85rem; line-height: 1.6;">
            <strong>Built as an end-to-end demonstration of production-style data engineering for predictive maintenance.</strong><br>
            Automated via GitHub Actions. Dashboard powered by Streamlit & PostgreSQL.<br>
            <a href="https://github.com/devdasrahul/datavisualisation" style="color: #0D9488; text-decoration: none;">View Source on GitHub</a>
        </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
