import streamlit as st
import momentum_projects as mp
import option_projects as op
import macro_projects as mrp
import regime_classifier as rc
import mean_reversion as mr
import pairs_trading as pt
import technical_analysis as ta
import relative_strength as rs
import yield_curve as yc
import sector_rotation as sr
import economic_calendar as ec
import fear_greed as fg
import var_calculator as vc
import correlation_matrix as cm
import stress_testing as st_test
import factor_model as fm
import portfolio_optimizer as po
import performance_dashboard as perf_dash
import trade_decision as td
import watchlist as wl
import risk_rules as rr

CAMLINK    = "https://www.linkedin.com/in/cameron-mahmood-86334628a"
GITHUBLINK = "https://github.com/cameronmahmood"

LOGO_SVG = """<svg width="100%" viewBox="0 0 680 300" role="img" xmlns="http://www.w3.org/2000/svg"><rect style="fill:#080c14" width="680" height="300" rx="12"/><rect x="26" y="26" width="628" height="248" rx="8" style="fill:#0d1420"/><circle cx="134" cy="148" r="85" style="fill:#f0b429;opacity:0.06"/><circle cx="134" cy="148" r="52" style="fill:#f0b429;opacity:0.12"/><line x1="134" y1="52" x2="134" y2="70" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="134" y1="226" x2="134" y2="244" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="48" y1="148" x2="66" y2="148" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="202" y1="148" x2="220" y2="148" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="73" y1="87" x2="86" y2="100" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="195" y1="87" x2="182" y2="100" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="73" y1="209" x2="86" y2="196" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><line x1="195" y1="209" x2="182" y2="196" style="stroke:#f0b429;stroke-width:1.3;stroke-linecap:round;opacity:0.45"/><circle cx="134" cy="148" r="18" style="fill:#f0b429;opacity:0.22"/><circle cx="134" cy="148" r="10" style="fill:#f0b429"/><ellipse cx="98" cy="126" rx="8" ry="4.5" transform="rotate(-40 98 126)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="90" cy="140" rx="8" ry="4.5" transform="rotate(-20 90 140)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="90" cy="155" rx="8" ry="4.5" transform="rotate(15 90 155)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="96" cy="168" rx="8" ry="4.5" transform="rotate(38 96 168)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="106" cy="178" rx="8" ry="4.5" transform="rotate(58 106 178)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="170" cy="126" rx="8" ry="4.5" transform="rotate(40 170 126)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="178" cy="140" rx="8" ry="4.5" transform="rotate(20 178 140)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="178" cy="155" rx="8" ry="4.5" transform="rotate(-15 178 155)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="172" cy="168" rx="8" ry="4.5" transform="rotate(-38 172 168)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="162" cy="178" rx="8" ry="4.5" transform="rotate(-58 162 178)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="125" cy="110" rx="8" ry="4.5" transform="rotate(85 125 110)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="134" cy="109" rx="8" ry="4.5" transform="rotate(90 134 109)" style="fill:#c8a94a;opacity:0.8"/><ellipse cx="143" cy="110" rx="8" ry="4.5" transform="rotate(-85 143 110)" style="fill:#c8a94a;opacity:0.8"/><polygon points="68,218 82,210 100,215 118,198 136,192 154,178 172,162 188,148 188,218" style="fill:#f0b429;opacity:0.06"/><polyline points="68,218 82,210 100,215 118,198 136,192 154,178 172,162 188,148" style="fill:none;stroke:#f0b429;stroke-width:2.8;stroke-linecap:round;stroke-linejoin:round"/><circle cx="188" cy="148" r="10" style="fill:#f0b429;opacity:0.15"/><circle cx="188" cy="148" r="5" style="fill:#f0b429"/><path d="M 181 141 L 188 134 L 195 141" fill="none" style="stroke:#f0b429;stroke-width:2;stroke-linecap:round;stroke-linejoin:round"/><text x="240" y="155" style="font-family:Georgia,serif;font-size:52px;font-weight:700;fill:#ffffff;letter-spacing:-1px">APOLLO</text><text x="240" y="213" style="font-family:Georgia,serif;font-size:52px;font-weight:400;fill:#f0b429;letter-spacing:-1px">QUANT</text><line x1="240" y1="229" x2="648" y2="229" style="stroke:#f0b429;stroke-width:0.6;opacity:0.22"/><text x="242" y="248" style="font-family:Arial,sans-serif;font-size:10.5px;fill:#ffffff;opacity:0.32;letter-spacing:4px">QUANTITATIVE FINANCE RESEARCH</text></svg>"""

st.set_page_config(page_title="Apollo Quant", layout="wide")
st.title("📊 Apollo Quant")

st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "🎯 Trade Decision",
    "📋 Watchlist",
    "Trading Tools",
    "Research & Macro",
    "Risk & Portfolio",
    "Performance",
    "📜 Risk Rules",
])

# ─────────────────────────────────────────
# HOME
# ─────────────────────────────────────────
if page == "Home":
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "A live quantitative finance research and paper-trading analytics platform "
        "built in Python by **Cameron Mahmood**. Use the sidebar to navigate."
    )
    st.info("💡 If the app is loading slowly, it may be warming up from sleep mode. Please allow 30–60 seconds on first visit.")

    st.markdown("---")
    st.subheader("🗺️ Apollo Quant Workflow")
    st.markdown("Use these six steps in order before every Investopedia paper trade:")

    steps = [
        ("1️⃣", "Check Market Regime",    "Is the environment Risk-On, Risk-Off, Inflationary, or Recessionary?",      "Research & Macro → Market Regime Classifier"),
        ("2️⃣", "Check the Watchlist",    "Which assets are leading, lagging, or oversold right now?",                  "📋 Watchlist"),
        ("3️⃣", "Run Trade Decision",     "Enter your ticker — all models run automatically for a Buy/Watch/Avoid.",    "🎯 Trade Decision"),
        ("4️⃣", "Check the Macro",        "Is the bull or bear case supported by rates, oil, and the dollar?",          "Research & Macro → Macro Research + Yield Curve"),
        ("5️⃣", "Size the Position",      "How many shares? What's the stop loss? Is VaR within limits?",              "📜 Risk Rules → Position Sizing"),
        ("6️⃣", "Log the Trade",          "Record your thesis, entry price, and tools used. Track vs SPY.",             "Performance → Trading Performance Dashboard"),
    ]

    for icon, title, desc, tool in steps:
        st.markdown(
            f"""<div style="border-left:4px solid #f0b429;padding:10px 16px;margin-bottom:8px;
            background:#f0b42908;border-radius:0 6px 6px 0;">
            <span style="font-size:1.1rem;font-weight:700;color:#f0b429;">{icon} {title}</span><br>
            <span style="color:#ccc;font-size:0.88rem;">{desc}</span><br>
            <span style="color:#888;font-size:0.8rem;">📍 {tool}</span>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📂 What's Inside")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div style="border:1px solid #f0b42944;border-radius:8px;padding:16px;margin-bottom:12px;">
            <div style="font-size:1rem;font-weight:700;color:#f0b429;">🎯 Trade Decision + 📋 Watchlist</div>
            <div style="color:#ccc;font-size:0.85rem;margin-top:6px;">
            One-click trade decision combining all models into a Buy/Watch/Avoid verdict.
            Auto-generated watchlist showing leaders, laggards, and mean reversion candidates.
            </div></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="border:1px solid #f0b42944;border-radius:8px;padding:16px;margin-bottom:12px;">
            <div style="font-size:1rem;font-weight:700;color:#f0b429;">⚙️ Trading Signals</div>
            <div style="color:#ccc;font-size:0.85rem;margin-top:6px;">
            Momentum (Cross-Sectional, Dual), Relative Strength Scanner, Mean Reversion (RSI, Bollinger, Z-Score),
            MACD & Technical Signals, Pairs Trading with cointegration testing.
            </div></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="border:1px solid #3498db44;border-radius:8px;padding:16px;margin-bottom:12px;">
            <div style="font-size:1rem;font-weight:700;color:#3498db;">🌍 Macro Research</div>
            <div style="color:#ccc;font-size:0.85rem;margin-top:6px;">
            AI Market Regime Classifier, cross-asset macro outlook, live yield curve,
            sector rotation heatmap, economic calendar, and Fear & Greed Index.
            </div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style="border:1px solid #2ecc7144;border-radius:8px;padding:16px;margin-bottom:12px;">
            <div style="font-size:1rem;font-weight:700;color:#2ecc71;">🛡️ Portfolio Risk</div>
            <div style="color:#ccc;font-size:0.85rem;margin-top:6px;">
            Value at Risk (3 methods), stress testing across 8 historical crises,
            correlation matrix, Fama-French factor model, and Markowitz portfolio optimizer.
            </div></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="border:1px solid #9b59b644;border-radius:8px;padding:16px;margin-bottom:12px;">
            <div style="font-size:1rem;font-weight:700;color:#9b59b6;">📊 Performance Tracking</div>
            <div style="color:#ccc;font-size:0.85rem;margin-top:6px;">
            Paper trade journal with thesis logging, live P&L vs S&P 500 benchmark,
            Sharpe ratio, win rate, sector attribution, and CSV export.
            </div></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="border:1px solid #e74c3c44;border-radius:8px;padding:16px;margin-bottom:12px;">
            <div style="font-size:1rem;font-weight:700;color:#e74c3c;">📜 Risk Rules</div>
            <div style="color:#ccc;font-size:0.85rem;margin-top:6px;">
            Position sizing calculator, entry/exit rules for every strategy,
            stop loss guide, and trading discipline rules.
            </div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🛠️ Built With")
    st.markdown("Python · Streamlit · pandas · NumPy · SciPy · yfinance · matplotlib · seaborn · statsmodels · scikit-learn")

    st.markdown("---")
    st.subheader("🔗 Connect")
    st.markdown(
        f"- **Cameron Mahmood** — [LinkedIn]({CAMLINK}) | [GitHub]({GITHUBLINK})\n"
        f"- **Providence College** — B.A. Quantitative Finance, Expected May 2027\n"
        f"- **Bally's Casino** — Finance Intern, June 2026–Present"
    )
    st.caption(
        "⚠️ Disclaimer: Apollo Quant is for educational research and paper-trading only. "
        "It is not investment advice. All data via Yahoo Finance."
    )
    st.caption("Last updated: June 2026 | apolloquant.streamlit.app")

# ─────────────────────────────────────────
# TRADE DECISION
# ─────────────────────────────────────────
elif page == "🎯 Trade Decision":
    td.run_trade_decision()

# ─────────────────────────────────────────
# WATCHLIST
# ─────────────────────────────────────────
elif page == "📋 Watchlist":
    wl.run_watchlist()

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
        with st.expander("📊 Black-Scholes Option Pricer", expanded=True):
            op.run_black_scholes()
        with st.expander("🎲 Monte Carlo Simulation", expanded=False):
            op.run_monte_carlo()
        with st.expander("📈 Market-Implied Move (from ATM IV)", expanded=True):
            if hasattr(op, "run_implied_move"):
                op.run_implied_move()
            elif hasattr(op, "run_implied_move_table"):
                op.run_implied_move_table()
    elif tool == "Momentum Strategies":
        st.markdown("## 💡 Momentum Strategies")
        with st.expander("📘 Cross-Sectional Momentum", expanded=True):
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
    if tool == "Macro Research":               mrp.run_macro_research()
    elif tool == "Market Regime Classifier":   rc.run_regime_classifier()
    elif tool == "Yield Curve Visualizer":     yc.run_yield_curve()
    elif tool == "Sector Rotation Heatmap":    sr.run_sector_rotation()
    elif tool == "Economic Calendar":          ec.run_economic_calendar()
    elif tool == "Fear & Greed Index":         fg.run_fear_greed()

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
    if tool == "VaR Calculator":              vc.run_var_calculator()
    elif tool == "Correlation Matrix":        cm.run_correlation_matrix()
    elif tool == "Stress Testing":            st_test.run_stress_testing()
    elif tool == "Factor Model (Fama-French)": fm.run_factor_model()
    elif tool == "Portfolio Optimizer":       po.run_portfolio_optimizer()

# ─────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────
elif page == "Performance":
    perf_dash.run_performance_dashboard()

# ─────────────────────────────────────────
# RISK RULES
# ─────────────────────────────────────────
elif page == "📜 Risk Rules":
    rr.run_risk_rules()