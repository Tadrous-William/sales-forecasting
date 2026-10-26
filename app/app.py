"""Demand forecast & stock planner (Streamlit).  streamlit run app/app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA = ROOT / "output", ROOT / "data"
INK, BLUE, BLUE_LIGHT, GRID = "#0b0b0b", "#2a78d6", "rgba(42,120,214,0.18)", "#e7e6e1"

st.set_page_config(page_title="Demand Forecast & Stock Planner", page_icon="📦", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1400px}</style>", unsafe_allow_html=True)


@st.cache_data
def load():
    return (pd.read_parquet(DATA / "products.parquet"), pd.read_parquet(DATA / "weekly_product.parquet"),
            pd.read_parquet(OUT / "forecast.parquet"), pd.read_parquet(OUT / "backtest.parquet"),
            pd.read_csv(OUT / "stock_plan.csv"), json.loads((OUT / "metrics.json").read_text()))


products, wp, fc, bt, plan, m = load()
first_week = pd.Timestamp(m["forecast_weeks"][0])
st.title("📦 Demand Forecast & Stock Planner")
st.caption(f"UK online gift store · {m['products']} best-selling products · forecast made after the week of "
           f"{first_week - pd.Timedelta(weeks=1):%d %b %Y} for the next {m['horizon_weeks']} weeks (Christmas closure included)")

w4 = m["wape"]["store_4_weeks"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected units, next 4 weeks", f"{plan['demand_4w'].sum():,.0f}")
c2.metric("Stock to cover 4 weeks (90%)", f"{plan['stock_90'].sum():,.0f}")
c3.metric("Store 4-week forecast error (tested)", f"{w4['combined']:.1%}", f"{w4['combined'] - w4['avg_8w']:+.1%} vs 8-week average",
          delta_color="inverse")
c4.metric("Weeks inside the forecast range (tested)", f"{m['range_check_on_last_3_dates']['inside_low_high']:.0%}")

tab1, tab2, tab3 = st.tabs(["Product forecast", "Stock plan", "Accuracy test"])
with tab1:
    labels = {r.stock_code: f"{r.product_name.title()} ({r.stock_code})" for r in products.itertuples()}
    code = st.selectbox("Product", list(labels), format_func=labels.get)
    h = wp[(wp["stock_code"] == code) & (wp["week"] >= "2011-01-01")]
    f = fc[fc["stock_code"] == code]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(f["week"]) + list(f["week"][::-1]), y=list(f["high"]) + list(f["low"][::-1]),
                             fill="toself", fillcolor=BLUE_LIGHT, line=dict(width=0), mode="lines", name="Likely range (8 in 10 weeks)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h["week"], y=h["units"], name="Units sold", line=dict(color=INK, width=2)))
    fig.add_trace(go.Scatter(x=f["week"], y=f["forecast"].round(), name="Forecast", mode="lines+markers",
                             line=dict(color=BLUE, width=2.5), marker=dict(size=7)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis=dict(gridcolor=GRID, rangemode="tozero"))
    st.plotly_chart(fig, width="stretch")
    p = plan[plan["stock_code"] == code].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Sold per week (last 8 weeks)", f"{p['avg_week_last_8']:,.0f}")
    b.metric("Expected demand, next 4 weeks", f"{p['demand_4w']:,.0f}")
    c.metric("Stock to cover 4 weeks (90%)", f"{p['stock_90']:,.0f}")
    st.dataframe(f.assign(week=f["week"].dt.strftime("%d %b %Y"))[["week", "trading_days", "forecast", "low", "high"]]
                 .rename(columns={"week": "Week of", "trading_days": "Trading days", "forecast": "Forecast units",
                                  "low": "Low", "high": "High"}).round(0), hide_index=True, width="stretch")
with tab2:
    st.markdown("**Stock to cover 4 weeks (90%)**: enough units that demand in the next 4 weeks is covered 9 times out of 10.")
    show = plan.rename(columns={"stock_code": "Stock code", "product_name": "Product", "category": "Category",
                                "avg_week_last_8": "Sold per week (last 8 wk)", "demand_4w": "Demand next 4 wk",
                                "stock_90": "Stock to cover 4 wk (90%)", "demand_8w": "Demand next 8 wk",
                                "revenue_last_52w": "Revenue last 52 wk (£)"})
    st.dataframe(show.round(0), hide_index=True, width="stretch", height=520)
    st.download_button("⬇ Download the Excel forecast", (OUT / "forecast_next_8_weeks.xlsx").read_bytes(),
                       file_name="forecast_next_8_weeks.xlsx", type="primary")
with tab3:
    names = {"combined": "Combined forecast (used)", "avg_8w": "8-week average", "avg_8w_per_day": "8-week avg per trading day",
             "same_week_ly": "Same week last year", "ly_x_trend": "Last year x trend", "last_week": "Last week"}
    rows = [{"Method": names[k], **{lbl: m["wape"][lvl][k] for lvl, lbl in
             [("product_4_weeks", "Each product, 4 weeks"), ("store_4_weeks", "Whole store, 4 weeks"),
              ("product_week", "Each product, 1 week")]}} for k in names]
    st.markdown(f"Forecast error (lower is better) on **{m['product_weeks_tested']:,} product-weeks**: forecasts were made on "
                f"{len(m['forecast_dates'])} dates in 2011, each using only the data before that date.")
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.1%}" for c in rows[0] if c != "Method"}), hide_index=True,
                 width="stretch")
    st.image(str(ROOT / "portfolio" / "backtest_store.png"), width="stretch")
