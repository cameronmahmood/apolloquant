# risk_rules.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def run_risk_rules():
    st.subheader("Risk Rules & Strategy Guide")
    st.markdown(
        "Apollo Quant's trading rules. These govern every paper trade on Investopedia. "
        "Discipline without rules is just gambling."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚠️ Risk Rules",
        "📐 Position Sizing",
        "📊 Strategy Rules",
        "🛑 Stop Loss Guide"
    ])

    # =========================
    # TAB 1 — RISK RULES
    # =========================
    with tab1:
        st.markdown("### ⚠️ Apollo Quant Risk Rules")
        st.markdown("These rules apply to every single paper trade. No exceptions.")

        rules = [
            ("🔢 Position Limits",
             "No single position above **25%** of portfolio.\nNo single sector above **40%** of portfolio.\nMinimum 4 positions when fully invested.",
             "#f0b429"),
            ("📉 VaR Limit",
             "1-day portfolio VaR must stay **below 2%** of total account value.\nIf VaR exceeds 2%, reduce position sizes before entering new trades.\nRun VaR Calculator before every new position.",
             "#e74c3c"),
            ("💥 Stress Test Limit",
             "Portfolio must survive a 2008-style crash with **less than 30% loss**.\nIf stress test shows >30% loss, add hedges (GLD, TLT) or reduce equities.\nRun Stress Testing page monthly.",
             "#e67e22"),
            ("📅 Economic Calendar Rule",
             "**Never enter a new position within 48 hours of:**\n- FOMC meeting\n- CPI release\n- NFP (jobs report)\n- GDP release\nReduce existing positions by 50% before FOMC and CPI.",
             "#3498db"),
            ("📈 Regime Filter",
             "Always confirm the **Market Regime** before entering a trade.\n- Risk-On → bias long equities\n- Risk-Off → bias TLT, GLD, defensives\n- Recessionary → cash, TLT, avoid equities\n- Never trade against the dominant regime.",
             "#2ecc71"),
            ("🔁 Rebalance Discipline",
             "Rebalance **monthly** — not randomly.\nFor Dual Momentum: check signal on the first trading day of each month.\nFor other strategies: review every Sunday before the trading week.",
             "#9b59b6"),
            ("📊 SPY Benchmark",
             "Every trade must be compared to simply holding SPY.\nIf your paper portfolio underperforms SPY by >5% over 3 months, review your strategy.\nTrack in the Performance Dashboard.",
             "#1abc9c"),
            ("✋ Max Drawdown Exit",
             "If any single position loses **15% from entry**, exit immediately regardless of thesis.\nIf total portfolio is down **10% from peak**, stop trading new positions and review.",
             "#e74c3c"),
        ]

        for title, detail, color in rules:
            st.markdown(
                f"""<div style="border-left:4px solid {color};padding:12px 16px;
                margin-bottom:10px;background:{color}11;border-radius:0 6px 6px 0;">
                <div style="font-weight:700;color:{color};font-size:0.95rem;">{title}</div>
                <div style="color:#ccc;font-size:0.85rem;margin-top:6px;white-space:pre-line;">{detail}</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("### 🚦 Quick Decision Rules")
        st.markdown("""
| Situation | Rule |
|-----------|------|
| MACD says Sell but RSI < 30 | Only buy if it's a **Mean Reversion** setup, not momentum |
| Strong Laggard but cheap | Do NOT buy. Wait for Mean Reversion confirmation (RSI < 30, Z < -2) |
| Strong Leader but RSI > 70 | Wait for a pullback. Use MACD for entry timing |
| Market Regime is Risk-Off | No new long positions in cyclicals or growth stocks |
| VIX above 25 | Reduce all position sizes by 50% |
| Economic event in 48hrs | No new positions. Reduce size on existing |
| Portfolio up >15% in 1 month | Consider taking partial profits |
""")

    # =========================
    # TAB 2 — POSITION SIZING
    # =========================
    with tab2:
        st.markdown("### 📐 Position Sizing Calculator")
        st.markdown("Use this before every Investopedia trade to calculate exactly how many shares to buy.")

        col1, col2 = st.columns(2)
        with col1:
            account = st.number_input("Account Size ($)", value=100000, min_value=1000, step=1000, key="ps_account")
            max_risk_pct = st.slider("Max Risk Per Trade (% of account)", 0.5, 5.0, 1.0, 0.5, key="ps_risk")
            entry_price = st.number_input("Entry Price ($)", value=100.0, min_value=0.01, step=0.01, key="ps_entry")
        with col2:
            stop_method = st.radio("Stop Loss Method", ["ATR-Based (2x ATR)", "Percentage", "Manual Price"], key="ps_method")
            if stop_method == "ATR-Based (2x ATR)":
                atr = st.number_input("ATR (14-day)", value=2.50, min_value=0.01, step=0.01, key="ps_atr")
                stop_price = round(entry_price - 2 * atr, 2)
                st.metric("Calculated Stop Loss", f"${stop_price:.2f}")
            elif stop_method == "Percentage":
                stop_pct = st.slider("Stop Loss %", 1.0, 20.0, 5.0, 0.5, key="ps_pct")
                stop_price = round(entry_price * (1 - stop_pct/100), 2)
                st.metric("Calculated Stop Loss", f"${stop_price:.2f}")
            else:
                stop_price = st.number_input("Stop Loss Price ($)", value=95.0, min_value=0.01, step=0.01, key="ps_stop")

        risk_per_share = round(entry_price - stop_price, 2)
        max_risk_dollar = account * max_risk_pct / 100
        shares = int(max_risk_dollar / risk_per_share) if risk_per_share > 0 else 0
        position_value = round(shares * entry_price, 2)
        pct_of_portfolio = position_value / account * 100

        # Target prices
        target_1r = round(entry_price + risk_per_share, 2)
        target_2r = round(entry_price + 2 * risk_per_share, 2)
        target_3r = round(entry_price + 3 * risk_per_share, 2)

        st.markdown("### 📊 Position Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry Price",     f"${entry_price:.2f}")
        c2.metric("Stop Loss",       f"${stop_price:.2f}")
        c3.metric("Risk Per Share",  f"${risk_per_share:.2f}")
        c4.metric("Max Risk $",      f"${max_risk_dollar:,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Shares to Buy",   f"{shares}")
        c6.metric("Position Value",  f"${position_value:,.0f}")
        c7.metric("% of Portfolio",  f"{pct_of_portfolio:.1f}%",
                  delta="✅ Within 25% limit" if pct_of_portfolio <= 25 else "❌ Exceeds 25% limit")
        c8.metric("Risk/Reward",     "1:2 min recommended")

        st.markdown("### 🎯 Target Prices (Risk/Reward)")
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("1R Target (1:1)",  f"${target_1r:.2f}", delta=f"+{risk_per_share:.2f}")
        tc2.metric("2R Target (1:2)",  f"${target_2r:.2f}", delta=f"+{2*risk_per_share:.2f}")
        tc3.metric("3R Target (1:3)",  f"${target_3r:.2f}", delta=f"+{3*risk_per_share:.2f}")

        st.info("**Rule:** Only take trades where the potential gain is at least 2x the risk (1:2 R/R minimum).")

        if pct_of_portfolio > 25:
            st.error("⚠️ Position exceeds 25% limit. Reduce shares or use a tighter stop loss.")

    # =========================
    # TAB 3 — STRATEGY RULES
    # =========================
    with tab3:
        st.markdown("### 📊 Strategy Entry & Exit Rules")
        st.markdown("These define exactly when each model generates a valid trade signal.")

        strategies = [
            {
                "name": "📡 Market Regime — Risk Filter",
                "color": "#9b59b6",
                "entry": [
                    "Check regime BEFORE running any other model",
                    "Risk-On → bias long equities and cyclicals",
                    "Risk-Off → bias TLT, GLD, XLU, XLP",
                    "Inflationary → bias XLE, GLD, commodities",
                    "Recessionary → cash, TLT, avoid all equities",
                ],
                "exit": [
                    "Regime shifts to Risk-Off → exit all equity longs within 2-3 days",
                    "Regime shifts to Recessionary → exit immediately, move to cash",
                ],
                "note": "Used as a risk FILTER, not a standalone trade signal."
            },
            {
                "name": "🔁 Relative Strength & Momentum",
                "color": "#2ecc71",
                "entry": [
                    "Signal = 🟢 Leader or 🟡 Watch (Relative Strength Scanner)",
                    "Outperforming SPY on both 1M and 3M",
                    "Trend score at least 2/3 (above SMA20 and SMA50)",
                    "RSI between 40-65 (not overbought)",
                    "MACD not showing Sell signal",
                    "Market Regime is Risk-On",
                ],
                "exit": [
                    "Signal drops to Neutral or Laggard",
                    "MACD turns bearish (crossover below signal line)",
                    "Price closes below 50-day SMA",
                    "RSI rises above 75 (overbought — take partial profit)",
                    "Stop loss hit (2x ATR below entry)",
                ],
                "note": "Best strategy for trending Risk-On markets."
            },
            {
                "name": "📉 Mean Reversion",
                "color": "#f0b429",
                "entry": [
                    "RSI (14) below 30 — oversold",
                    "Z-Score below -2 — more than 2 standard deviations below mean",
                    "Price near or below lower Bollinger Band",
                    "Market Regime is NOT Recessionary (avoid catching falling knives)",
                    "No major economic event in next 48 hours",
                ],
                "exit": [
                    "Z-Score returns to 0 (mean) — take profit",
                    "RSI rises back above 50",
                    "Price returns to 20-day moving average",
                    "Stop loss hit (1x ATR below entry — tighter stop for mean reversion)",
                ],
                "note": "Mean reversion fails during strong trends. Only use in range-bound or neutral regimes."
            },
            {
                "name": "🧮 Dual Momentum",
                "color": "#3498db",
                "entry": [
                    "Signal shows US, INTL, or CASH at start of each month",
                    "Buy the signaled ETF on the first trading day of the month",
                    "Hold until next month's signal",
                    "Do NOT change position mid-month based on noise",
                ],
                "exit": [
                    "Signal switches at monthly rebalance — exit old position, enter new",
                    "If signal = CASH, move to IEF or BIL immediately",
                    "Never hold a losing Dual Momentum position past the rebalance date",
                ],
                "note": "Monthly rebalance only. Do not overtrade this strategy."
            },
            {
                "name": "🔗 Pairs Trading",
                "color": "#e67e22",
                "entry": [
                    "Cointegration p-value < 0.05 (Strict Mode) OR correlation > 0.7 (Correlation Mode)",
                    "Z-Score of spread above +2 (short Asset1, long Asset2) OR below -2 (long Asset1, short Asset2)",
                    "ADF p-value < 0.05 (spread is stationary)",
                    "Backtest Sharpe ratio is positive",
                ],
                "exit": [
                    "Z-Score of spread returns to 0 — take profit",
                    "Z-Score moves further against position beyond -3/+3 — stop loss",
                    "Cointegration breaks down on re-test",
                ],
                "note": "Pairs trading is market-neutral. Works in any regime. Use smaller position sizes."
            },
        ]

        for strat in strategies:
            with st.expander(strat["name"], expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✅ Entry Rules:**")
                    for rule in strat["entry"]:
                        st.markdown(f"- {rule}")
                with col_b:
                    st.markdown("**🚪 Exit Rules:**")
                    for rule in strat["exit"]:
                        st.markdown(f"- {rule}")
                st.caption(f"📝 {strat['note']}")

    # =========================
    # TAB 4 — STOP LOSS GUIDE
    # =========================
    with tab4:
        st.markdown("### 🛑 Stop Loss Guide")
        st.markdown("Never enter a trade without a stop loss. This is the most important rule.")

        st.markdown("""
**Stop Loss Methods by Strategy:**

| Strategy | Stop Loss Method | Typical Size |
|----------|-----------------|--------------|
| Momentum / Relative Strength | 2x ATR below entry | 4-8% |
| Mean Reversion | 1x ATR below entry (tighter) | 2-4% |
| Dual Momentum | Exit at monthly rebalance if signal flips | N/A |
| Pairs Trading | Z-Score exceeds ±3 | Varies |
| Any trade | Hard stop: 15% below entry | 15% |

**ATR-Based Stop Loss:**
ATR (Average True Range) measures how much an asset moves on a typical day.
A 2x ATR stop means you're giving the trade 2 average days of movement before stopping out.
This prevents being stopped out by normal volatility.

**How to calculate:**
1. Find ATR on the MACD & Technical Signals page
2. Multiply by 2
3. Subtract from entry price
4. That's your stop loss

**Example:**
- SPY entry: $740
- ATR (14): $10.76
- Stop loss: $740 - (2 × $10.76) = $718.48
- Risk per share: $21.52
- Shares (1% risk on $100k): 46 shares
- Position value: $34,040 (34% of portfolio — consider reducing)

**Take Profit Targets:**
- Minimum 1:2 risk/reward on all trades
- Take 50% profit at 1R, let rest run to 2R
- Exit full position at 3R or when signal reverses

**Never move your stop loss further away.**
You can move it closer (trailing stop) but never wider once a trade is open.
""")

        st.info("""
**Disclaimer:** Apollo Quant is an educational research and paper-trading platform.
All models, signals, and trade suggestions are for learning purposes only and do not constitute investment advice.
Past performance of backtested strategies does not guarantee future results.
Data sourced from Yahoo Finance. Models are built on historical data and may not predict future market behavior.
""")

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} | Apollo Quant — For educational use only")
