cat > /workspaces/apolloquant/streamlit_app.py << 'EOF'
import streamlit as st
import momentum_projects as mp
import option_projects as op
import macro_projects as mrp
import regime_classifier as rc

CAMLINK = "https://www.linkedin.com/in/cameron-mahmood-86334628a"
GITHUBLINK = "https://github.com/cameronmahmood"

st.set_page_config(page_title="Apollo Quant", layout="wide")
st.title("📊 Quantitative Finance Projects")

st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Projects", "Macro Research", "Market Regime"])

if page == "Home":
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
- **Cameron Mahmood** — [LinkedIn]({CAMLINK}) | [GitHub]({GITHUBLINK})
- **Providence College** — B.A. Quantitative Finance, Expected May 2027
- **Delphi Macro** — Head of Quantitative Modeling
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
EOF