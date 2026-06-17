import streamlit as st
import momentum_projects as mp
import option_projects as op
import macro_projects as mrp
import regime_classifier as rc

CAMLINK = "https://www.linkedin.com/in/cameron-mahmood-86334628a"

st.set_page_config(page_title="Apollo Quant", layout="wide")
st.title("📊 Quantitative Finance Projects")

st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Projects", "Macro Research", "Market Regime"])

if page == "Home":
    st.markdown("""
<svg width="100%" viewBox="0 0 680 300" role="img" xmlns="http://www.w3.org/2000/svg">
<rect style="fill:#080c14" width="680" height="300" rx="12"/>
<rect x="26" y="26" width="628" height="248" rx="8" style="fill:#0d1420"/>
<circle cx="134" cy="148" r="85" style="fill:#f0b429;opacity:0.06"/>
<circle cx="134" cy="148" r="52" style="fill:#f0b429;opacity:0.12"/>
<line x1="134" y1="52" x2="134" y2="70" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/>
<line x1="134" y1="226" x2="134" y2="244" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/>
<line x1="48" y1="148" x2="66" y2="148" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/>
<line x1="202" y1="148" x2="220" y2="148" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/>
<line x1="73" y1="87" x2="86" y2="100" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/>
<line x1="195" y1="87" x2="182" y2="100" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/>
<line x1="73" y1="209" x2="86" y2="196" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/>
<line x1="195" y1="209" x2="182" y2="196" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/>
<circle cx="134" cy="148" r="18" style="fill:#f0b429;opacity:0.22"/>
<circle cx="134" cy="148" r="10" style="fill:#f0b429"/>
<ellipse cx="98" cy="126" rx="8" ry="4.5" transform="rotate(-40 98 126)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="90" cy="140" rx="8" ry="4.5" transform="rotate(-20 90 140)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="90" cy="155" rx="8" ry="4.5" transform="rotate(15 90 155)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="96" cy="168" rx="8" ry="4.5" transform="rotate(38 96 168)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="106" cy="178" rx="8" ry="4.5" transform="rotate(58 106 178)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="170" cy="126" rx="8" ry="4.5" transform="rotate(40 170 126)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="178" cy="140" rx="8" ry="4.5" transform="rotate(20 178 140)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="178" cy="155" rx="8" ry="4.5" transform="rotate(-15 178 155)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="172" cy="168" rx="8" ry="4.5" transform="rotate(-38 172 168)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="162" cy="178" rx="8" ry="4.5" transform="rotate(-58 162 178)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="125" cy="110" rx="8" ry="4.5" transform="rotate(85 125 110)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="134" cy="109" rx="8" ry="4.5" transform="rotate(90 134 109)" style="fill:#c8a94a;opacity:0.8"/>
<ellipse cx="143" cy="110" rx="8" ry="4.5" transform="rotate(-85 143 110)" style="fill:#c8a94a;opacity:0.8"/>
<polygon points="68,218 82,210 100,215 118,198 136,192 154,178 172,162 188,148 188,218" style="fill:#f0b429;opacity:0.06"/>
<polyline points="68,218 82,210 100,215 118,198 136,192 154,178 172,162 188,148" style="fill:none;stroke:#f0b429;stroke-width:2.8;stroke-linecap:round;stroke-linejoin:round"/>
<circle cx="188" cy="148" r="10" style="fill:#f0b429;opacity:0.15"/>
<circle cx="188" cy="148" r="5" style="fill:#f0b429"/>
<path d="M 181 141 L 188 134 L 195 141" fill="none" style="stroke:#f0b429;stroke-width:2;stroke-linecap:round;stroke-linejoin:round"/>
<text x="240" y="155" style="font-family:Georgia,serif;font-size:52px;font-weight:700;fill:#ffffff;letter-spacing:-1px">APOLLO</text>
<text x="240" y="213" style="font-family:Georgia,serif;font-size:52px;font-weight:400;fill:#f0b429;letter-spacing:-1px">QUANT</text>
<line x1="240" y1="229" x2="648" y2="229" style="stroke:#f0b429;stroke-width:0.6;opacity:0.22"/>
<text x="242" y="248" style="font-family:Arial,sans-serif;font-size:10.5px;fill:#ffffff;opacity:0.32;letter-spacing:4px">QUANTITATIVE FINANCE RESEARCH</text>
</svg>
""", unsafe_allow_html=True)
st.header("🏠 Welcome to Apollo Quant")
    st.markdown(
        "A live quantitative finance dashboard built in Python by **Cameron Mahmood**. "
        "Use the sidebar to navigate between sections."
    )
    st.info("💡 If the app is loading slowly, it may be warming up from sleep mode. Please allow 30–60 seconds on first visit.")

    st.markdown("---")
    st.subheader("📂 What's Inside")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**📈 Projects — Option Pricing & Derivatives**
Black-Scholes option pricing with P&L heatmaps, Monte Carlo simulation via Geometric Brownian Motion, and real-time market-implied move calculations from ATM implied volatility.

**💡 Projects — Momentum Strategies**
Cross-sectional momentum backtesting across equities and commodities, and a Dual Momentum strategy engine — both computing Sharpe, Sortino, Calmar, max drawdown, and CAGR.
""")
    with col2:
        st.markdown("""
**🌍 Macro Research**
Original cross-asset bullish and bearish analysis across Oil, the US Dollar, 2Y/10Y Treasury yields, and the S&P 500 — with live market price feeds updated every 5 minutes.

**🧠 Market Regime Classifier**
AI-driven regime classifier using nine cross-asset signals (VIX, yield curve, credit spreads, oil, gold, DXY) to identify Risk-On, Risk-Off, Inflationary, and Recessionary environments in real time.
""")

    st.markdown("---")
    st.subheader("🛠️ Built With")
    st.markdown("Python · Streamlit · pandas · NumPy · SciPy · yfinance · matplotlib · seaborn")

    st.markdown("---")
    st.subheader("🔗 Connect")
    st.markdown(f"""
- **Cameron Mahmood** — [LinkedIn]({CAMLINK}) 
- **Providence College** — B.A. Quantitative Finance, Expected May 2027
    """)

    st.caption(f"Last updated: June 2026 | apolloquant.streamlit.app")

elif page == "Projects":
    st.header("💼 Quant Finance Projects")
    project = st.selectbox("Choose a project:", ["Momentum Strategies", "Option Pricing & Derivatives"], index=0)

    if project == "Momentum Strategies":
        st.markdown("## 💡 Momentum Strategies")
        with st.expander("📘 Cross-Sectional Momentum Across Sectors", expanded=True):
            mp.run_cross_sectional()
        with st.expander("📊 Dual Momentum Strategy", expanded=False):
            mp.run_dual_momentum()

    elif project == "Option Pricing & Derivatives":
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

elif page == "Macro Research":
    mrp.run_macro_research()

elif page == "Market Regime":
    rc.run_regime_classifier()
