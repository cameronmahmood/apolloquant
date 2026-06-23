# stress_testing.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

CRISIS_SCENARIOS = {
    "🦠 COVID Crash (Feb–Mar 2020)":        {"start": "2020-02-19", "end": "2020-03-23", "description": "Global pandemic panic. S&P 500 fell 34% in 33 days — fastest bear market in history.", "spy_return": -0.34},
    "💥 2008 Financial Crisis (Peak)":       {"start": "2007-10-09", "end": "2009-03-09", "description": "Lehman Brothers collapse, housing bubble burst. S&P 500 fell 57% peak to trough.", "spy_return": -0.57},
    "📉 2022 Rate Shock (Jan–Oct 2022)":     {"start": "2022-01-03", "end": "2022-10-12", "description": "Fed hiked rates from 0% to 4.25% in 9 months. S&P 500 fell 25%, bonds fell 30%.", "spy_return": -0.25},
    "💻 Dot-Com Bust (2000–2002)":           {"start": "2000-03-24", "end": "2002-10-09", "description": "Tech bubble burst. Nasdaq fell 78%, S&P 500 fell 49%. Tech stocks devastated.", "spy_return": -0.49},
    "🏦 SVB / Regional Bank Crisis (2023)":  {"start": "2023-03-08", "end": "2023-03-14", "description": "Silicon Valley Bank collapsed. Regional banks fell 30-50% in a week.", "spy_return": -0.05},
    "⚡ Flash Crash (May 6, 2010)":          {"start": "2010-05-06", "end": "2010-05-06", "description": "Dow fell 1,000 points (9%) in minutes due to algorithmic trading cascade.", "spy_return": -0.09},
    "🇨🇳 China Devaluation (Aug 2015)":      {"start": "2015-08-18", "end": "2015-08-25", "description": "China devalued yuan. Global markets fell 10-15% in one week.", "spy_return": -0.11},
    "🛢️ Oil Crash (Nov 2014–Feb 2016)":      {"start": "2014-11-28", "end": "2016-02-11", "description": "WTI oil fell from $75 to $26. Energy stocks fell 40-60%.", "spy_return": -0.14},
    "🔴 Custom Scenario":                    {"start": None, "end": None, "description": "Define your own stress scenario with custom shock values.", "spy_return": None},
}

ASSET_BETAS = {
    "SPY":  {"COVID": -0.34, "2008": -0.57, "2022": -0.25, "DotCom": -0.49, "SVB": -0.05},
    "QQQ":  {"COVID": -0.28, "2008": -0.53, "2022": -0.32, "DotCom": -0.78, "SVB": -0.06},
    "TLT":  {"COVID": +0.22, "2008": +0.33, "2022": -0.29, "DotCom": +0.15, "SVB": +0.03},
    "GLD":  {"COVID": -0.02, "2008": +0.05, "2022": -0.02, "DotCom": +0.12, "SVB": +0.02},
    "HYG":  {"COVID": -0.22, "2008": -0.35, "2022": -0.15, "DotCom": -0.18, "SVB": -0.04},
    "XLF":  {"COVID": -0.45, "2008": -0.79, "2022": -0.12, "DotCom": -0.22, "SVB": -0.18},
    "XLE":  {"COVID": -0.51, "2008": -0.42, "2022": +0.31, "DotCom": -0.28, "SVB": -0.03},
    "XLK":  {"COVID": -0.28, "2008": -0.54, "2022": -0.33, "DotCom": -0.76, "SVB": -0.06},
    "XLV":  {"COVID": -0.12, "2008": -0.31, "2022": -0.06, "DotCom": -0.15, "SVB": -0.02},
    "XLP":  {"COVID": -0.15, "2008": -0.28, "2022": -0.04, "DotCom": +0.05, "SVB": -0.01},
    "GDX":  {"COVID": -0.10, "2008": -0.30, "2022": -0.08, "DotCom": +0.20, "SVB": +0.03},
    "USO":  {"COVID": -0.65, "2008": -0.48, "2022": +0.40, "DotCom": -0.20, "SVB": -0.02},
}

def _dark_ax(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

def run_stress_testing():
    st.subheader("💥 Stress Testing")
    st.markdown(
        "Simulates how your portfolio would perform during **historical market crises**. "
        "Every risk manager at a bank runs stress tests daily. "
        "Knowing your downside before it happens is the foundation of professional risk management."
    )

    with st.expander("📖 How Stress Testing Works", expanded=False):
        st.markdown("""
**What stress testing does:**
Takes your current portfolio holdings and estimates how they would have performed
during historical crisis periods using asset-specific historical returns.

**Why it matters:**
- During the 2008 crisis, a 60/40 portfolio lost ~25% even with bonds
- During 2022, both stocks AND bonds fell simultaneously — the "diversification" failed
- COVID showed that even "safe" assets can fall initially before recovering

**How to use results:**
- If your portfolio would lose >20% in a crisis scenario, consider adding hedges
- GLD and TLT have historically been the best crisis hedges (except 2022)
- High-beta positions (XLK, QQQ) amplify both gains and losses

**Key lesson from 2022:**
The traditional 60% stocks / 40% bonds portfolio failed because rising rates hurt BOTH.
This is why diversification across uncorrelated assets (including commodities) matters.
""")

    # ---- Portfolio Input ----
    st.markdown("### 💼 Portfolio")
    col1, col2 = st.columns(2)
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", value=100000, min_value=1000, step=1000, key="st_value")
    with col2:
        scenario_name = st.selectbox("Crisis Scenario", list(CRISIS_SCENARIOS.keys()), key="st_scenario")

    scenario = CRISIS_SCENARIOS[scenario_name]

    n_assets = st.number_input("Number of Assets", min_value=1, max_value=10, value=4, key="st_n")
    tickers_input = []
    weights_input = []

    cols = st.columns(2)
    defaults = [("SPY", 40), ("TLT", 30), ("GLD", 20), ("QQQ", 10)]
    for i in range(int(n_assets)):
        with cols[i % 2]:
            t = st.text_input(f"Ticker {i+1}", value=defaults[i][0] if i < 4 else "", key=f"st_t{i}").upper().strip()
            w = st.number_input(f"Weight {i+1} (%)", value=float(defaults[i][1]) if i < 4 else 0.0,
                                min_value=0.0, max_value=100.0, key=f"st_w{i}")
            if t:
                tickers_input.append(t)
                weights_input.append(w / 100)

    # Custom scenario
    if "Custom" in scenario_name:
        st.markdown("#### Custom Scenario Shocks")
        custom_shocks = {}
        shock_cols = st.columns(min(len(tickers_input), 4))
        for i, t in enumerate(tickers_input):
            with shock_cols[i % 4]:
                shock = st.number_input(f"{t} Shock (%)", value=-20.0, min_value=-100.0,
                                         max_value=100.0, key=f"st_shock{i}")
                custom_shocks[t] = shock / 100

    run = st.button("▶ Run Stress Test", key="st_run")
    if not run:
        st.info("Enter your portfolio and click Run Stress Test.")
        return

    total_w = sum(weights_input)
    if abs(total_w - 1.0) > 0.01:
        st.error(f"Weights sum to {total_w*100:.1f}%. Must sum to 100%.")
        return

    # ---- Run All Scenarios ----
    st.markdown("### 📊 Stress Test Results")
    st.markdown(f"**Scenario:** {scenario_name}")
    st.caption(scenario["description"])

    # Compute scenario returns for each asset
    results = []
    total_pnl = 0
    for t, w in zip(tickers_input, weights_input):
        pos_value = portfolio_value * w
        if "Custom" in scenario_name:
            shock = custom_shocks.get(t, -0.20)
        else:
            # Use known historical returns or estimate from SPY beta
            if t in ASSET_BETAS:
                scenario_key = "COVID" if "COVID" in scenario_name else \
                               "2008" if "2008" in scenario_name else \
                               "2022" if "2022" in scenario_name else \
                               "DotCom" if "Dot" in scenario_name else \
                               "SVB" if "SVB" in scenario_name else "COVID"
                shock = ASSET_BETAS[t].get(scenario_key, scenario["spy_return"] * 0.9)
            else:
                shock = scenario["spy_return"] * 0.9 if scenario["spy_return"] else -0.20

        pnl = pos_value * shock
        total_pnl += pnl
        results.append({
            "Ticker": t,
            "Weight": f"{w*100:.1f}%",
            "Position ($)": f"${pos_value:,.0f}",
            "Estimated Shock": f"{shock*100:+.1f}%",
            "P&L ($)": f"${pnl:,.0f}",
            "Signal": "🔴 Loss" if pnl < 0 else "🟢 Gain"
        })

    total_return = total_pnl / portfolio_value * 100
    ending_value = portfolio_value + total_pnl

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Value (Before)", f"${portfolio_value:,.0f}")
    c2.metric("Estimated P&L", f"${total_pnl:,.0f}", delta=f"{total_return:.1f}%")
    c3.metric("Portfolio Value (After)", f"${ending_value:,.0f}")

    if total_return < -30:
        st.error(f"⚠️ Catastrophic loss scenario: {total_return:.1f}%. Portfolio would need {abs(total_return/(100+total_return)*100):.1f}% gain to recover.")
    elif total_return < -15:
        st.warning(f"⚠️ Severe drawdown scenario: {total_return:.1f}%. Consider adding hedges (GLD, TLT, or short positions).")
    elif total_return < -5:
        st.info(f"Portfolio would experience a moderate drawdown of {total_return:.1f}% in this scenario.")
    else:
        st.success(f"✅ Portfolio is relatively resilient: {total_return:.1f}% in this scenario.")

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    # ---- All Scenarios Comparison ----
    st.markdown("### 📊 All Scenarios Comparison")
    scenario_results = {}
    for sname, sdata in CRISIS_SCENARIOS.items():
        if sdata["spy_return"] is None or "Custom" in sname:
            continue
        port_return = 0
        for t, w in zip(tickers_input, weights_input):
            if t in ASSET_BETAS:
                skey = "COVID" if "COVID" in sname else \
                       "2008" if "2008" in sname else \
                       "2022" if "2022" in sname else \
                       "DotCom" if "Dot" in sname else \
                       "SVB" if "SVB" in sname else "COVID"
                shock = ASSET_BETAS[t].get(skey, sdata["spy_return"] * 0.9)
            else:
                shock = sdata["spy_return"] * 0.9
            port_return += w * shock
        scenario_results[sname.split("(")[0].strip()] = port_return * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e1117"); _dark_ax(ax)
    labels = list(scenario_results.keys())
    values = list(scenario_results.values())
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.6)
    for bar, val in zip(bars, values):
        ax.text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
                color="white", fontsize=8)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_title("Portfolio Performance Across All Crisis Scenarios", color="white", fontsize=11)
    ax.set_xlabel("Estimated Return (%)", color="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
        "Historical returns estimated from actual crisis data"
    )
