import streamlit as st
import momentum_projects as mp
import option_projects as op
import macro_projects as mrp
import regime_classifier as rc
import mean_reversion as mr
import pairs_trading as pt
import technical_analysis as ta

CAMLINK = "https://www.linkedin.com/in/cameron-mahmood-86334628a"
GITHUBLINK = "https://github.com/cameronmahmood"

LOGO_SVG = """<svg width="100%" viewBox="0 0 680 300" role="img" xmlns="http://www.w3.org/2000/svg"><rect style="fill:#080c14" width="680" height="300" rx="12"/><rect x="26" y="26" width="628" height="248" rx="8" style="fill:#0d1420"/><circle cx="134" cy="148" r="85" style="fill:#f0b429;opacity:0.06"/><circle cx="134" cy="148" r="52" style="fill:#f0b429;opacity:0.12"/><line x1="134" y1="52" x2="134" y2="70" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="134" y1="226" x2="134" y2="244" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="48" y1="148" x2="66" y2="148" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="202" y1="148" x2="220" y2="148" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="73" y1="87" x2="86" y2="100" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="195" y1="87" x2="182" y2="100" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="73" y1="209" x2="86" y2="196" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="195" y1="209" x2="182" y2="196" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><circle cx="134" cy="148" r="18" style="fill:#f0b429;opacity:0.22"/><circle cx="134" cy="148" r="10" style="fill:#f0b429"/><ellipse cx="98" cy="126" rx="8" ry="4.5" transform="rotate(-40 98 126)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="90" cy="140" rx="8" ry="4.5" transform="rotate(-20 90 140)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="90" cy="155" rx="8" ry="4.5" transform="rotate(15 90 155)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="96" cy="168" rx="8" ry="4.5" transform="rotate(38 96 168)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="106" cy="178" rx="8" ry="4.5" transform="rotate(58 106 178)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="170" cy="126" rx="8" ry="4.5" transform="rotate(40 170 126)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="178" cy="140" rx="8" ry="4.5" transform="rotate(20 178 140)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="178" cy="155" rx="8" ry="4.5" transform="rotate(-15 178 155)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="172" cy="168" rx="8" ry="4.5" transform="rotate(-38 172 168)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="162" cy="178" rx="8" ry="4.5" transform="rotate(-58 162 178)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="125" cy="110" rx="8" ry="4.5" transform="rotate(85 125 110)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="134" cy="109" rx="8" ry="4.5" transform="rotate(90 134 109)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="143" cy="110" rx="8" ry="4.5" transform="rotate(-85 143 110)" style="fill:#c8a94a;opacity:0.8"/><polygon points="68,218 82,210 100,215 118,198 136,192 154,178 172,162 188,148 188,218" style="fill:#f0b429;opacity:0.06"/><polyline points="68,218 82,210 100,215 118,198 136,192 154,178 172,162 188,148" style="fill:none;stroke:#f0b429;stroke-width:2.8;stroke-linecap:round;stroke-linejoin:round"/><circle cx="188" cy="148" r="10" style="fill:#f0b429;opacity:0.15"/><circle cx="188" cy="148" r="5" style="fill:#f0b429"/><path d="M 181 141 L 188 134 L 195 141" fill="none" style="stroke:#f0b429;stroke-width:2;stroke-linecap:round;stroke-linejoin:round"/><text x="240" y="155" style="font-family:Georgia,serif;font-size:52px;font-weight:700;fill:#ffffff;letter-spacing:-1px">APOLLO</text><text x="240" y="213" style="font-family:Georgia,serif;font-size:52px;font-weight:400;fill:#f0b429;letter-spacing:-1px">QUANT</text><line x1="240" y1="229" x2="648" y2="229" style="stroke:#f0b429;stroke-width:0.6;opacity:0.22"/><text x="242" y="248" style="font-family:Arial,sans-serif;font-size:10.5px;fill:#ffffff;opacity:0.32;letter-spacing:4px">QUANTITATIVE FINANCE RESEARCH</text></svg>"""

st.set_page_config(page_title="Apollo Quant", layout="wide")
st.title("📊 Apollo Quant")

st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Trading Tools",
    "Research & Macro",
    "Risk & Portfolio",
    "Performance"
])

# ─────────────────────────────────────────
# HOME
# ─────────────────────────────────────────
if page == "Home":
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("A live quantitative finance dashboard built in Python by **Cameron Mahmood**. Use the sidebar to navigate between sections.")
    st.info("💡 If the app is loading slowly, it may be warming up from sleep mode. Please allow 30–60 seconds on first visit.")

    st.markdown("---")
    st.subheader("📂 What's Inside")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**⚙️ Trading Tools**
Options pricing (Black-Scholes, Monte Carlo, Implied Move), momentum strategies (Cross-Sectional, Dual Momentum), mean reversion scanner (RSI, Bollinger Bands, Z-Score), MACD & technical signals, pairs trading, and relative strength scanner.

**🌍 Research & Macro**
Original cross-asset macro research with live price feeds, AI-driven market regime classifier, yield curve visualizer, sector rotation heatmap, economic calendar, and Fear & Greed index.
""")
    with col2:
        st.markdown("""
**🛡️ Risk & Portfolio**
Value at Risk (VaR) calculator, correlation matrix, stress testing across historical crisis scenarios, Fama-French factor model, and portfolio optimizer with efficient frontier.

**📊 Performance**
Quantitative trading performance dashboard tracking paper trades vs S&P 500 benchmark with Sharpe ratio, win rate, sector attribution, and trade journal with reasoning.
""")

    st.markdown("---")
    st.subheader("🛠️ Built With")
    st.markdown("Python · Streamlit · pandas · NumPy · SciPy · yfinance · matplotlib · seaborn · scikit-learn")

    st.markdown("---")
    st.subheader("🔗 Connect")
    st.markdown(f"- **Cameron Mahmood** — [LinkedIn]({CAMLINK}) | [GitHub]({GITHUBLINK})\n- **Providence College** — B.A. Quantitative Finance, Expected May 2027")
    st.caption("Last updated: June 2026 | apolloquant.streamlit.app")

# ─────────────────────────────────────────
# TRADING TOOLS
# ─────────────────────────────────────────
elif page == "Trading Tools":
    st.header("⚙️ Trading Tools")
    tool = st.selectbox("Choose a tool:", [
        "Option Pricing & Derivatives",
        "Momentum Strategies",
        "Mean Reversion Scanner",
        "MACD & Technical Signals",
        "Pairs Trading",
        "Relative Strength Scanner",
    ])

    if tool == "Option Pricing & Derivatives":
        st.markdown("## 📈 Option Pricing & Derivatives")
        with st.expander("📊 Black-Scholes & Binomial Tree Model", expanded=True):
            op.run_black_scholes()
        with st.expander("🎲 Monte Carlo Simulation for Option Pricing", expanded=False):
            op.run_monte_carlo()
        with st.expander("📈 Market-Implied Move (from ATM IV)", expanded=True):
            if hasattr(op, "run_implied_move"):
                op.run_implied_move()
            elif hasattr(op, "run_implied_move_table"):
                op.run_implied_move_table()

    elif tool == "Momentum Strategies":
        st.markdown("## 💡 Momentum Strategies")
        with st.expander("📘 Cross-Sectional Momentum Across Sectors", expanded=True):
            mp.run_cross_sectional()
        with st.expander("📊 Dual Momentum Strategy", expanded=False):
            mp.run_dual_momentum()

    elif tool == "Mean Reversion Scanner":
        mr.run_mean_reversion()

    elif tool == "MACD & Technical Signals":
        ta.run_technical_analysis()

    elif tool == "Pairs Trading":
        pt.run_pairs_trading()

    elif tool == "Relative Strength Scanner":
        rs.run_relative_strength()

# ─────────────────────────────────────────
# RESEARCH & MACRO
# ─────────────────────────────────────────
elif page == "Research & Macro":
    st.header("🌍 Research & Macro")
    tool = st.selectbox("Choose a section:", [
        "Macro Research",
        "Market Regime Classifier",
        "Yield Curve Visualizer",
        "Sector Rotation Heatmap",
        "Economic Calendar",
        "Fear & Greed Index",
    ])

    if tool == "Macro Research":
        mrp.run_macro_research()

    elif tool == "Market Regime Classifier":
        rc.run_regime_classifier()

    elif tool == "Yield Curve Visualizer":
        st.info("🔧 Yield Curve Visualizer — Coming Soon")

    elif tool == "Sector Rotation Heatmap":
        st.info("🔧 Sector Rotation Heatmap — Coming Soon")

    elif tool == "Economic Calendar":
        st.info("🔧 Economic Calendar — Coming Soon")

    elif tool == "Fear & Greed Index":
        st.info("🔧 Fear & Greed Index — Coming Soon")

# ─────────────────────────────────────────
# RISK & PORTFOLIO
# ─────────────────────────────────────────
elif page == "Risk & Portfolio":
    st.header("🛡️ Risk & Portfolio")
    tool = st.selectbox("Choose a tool:", [
        "VaR Calculator",
        "Correlation Matrix",
        "Stress Testing",
        "Factor Model (Fama-French)",
        "Portfolio Optimizer",
    ])

    if tool == "VaR Calculator":
        st.info("🔧 VaR Calculator — Coming Soon")

    elif tool == "Correlation Matrix":
        st.info("🔧 Correlation Matrix — Coming Soon")

    elif tool == "Stress Testing":
        st.info("🔧 Stress Testing — Coming Soon")

    elif tool == "Factor Model (Fama-French)":
        st.info("🔧 Factor Model — Coming Soon")

    elif tool == "Portfolio Optimizer":
        st.info("🔧 Portfolio Optimizer — Coming Soon")

# ─────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────
elif page == "Performance":
    st.header("📊 Trading Performance")
    st.info("🔧 Trading Performance Dashboard — Coming Soon")
