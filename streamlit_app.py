import streamlit as st
import momentum_projects as mp
import option_projects as op

st.set_page_config(page_title="Quant Finance Projects", layout="wide")
st.title("📊 Quantitative Finance Projects")

# Sidebar navigation
st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Projects"])

if page == "Home":
    st.header("🏠 Welcome to the Quantitative Finance Dashboard")
    st.write("This app showcases multiple quantitative finance projects built in Python. Use the sidebar to navigate.")

elif page == "About":
    st.header("ℹ️ About")
    st.write(
        "Created by Cameron Mahmood and Joseph Panaro. "
        "This dashboard includes option pricing, momentum strategies, portfolio optimization, "
        "and machine learning applications in finance."
    )

elif page == "Projects":
    st.header("💼 Quant Finance Projects")
    project = st.selectbox(
        "Choose a project:",
        [
            "Momentum Strategies",
            "Option Pricing & Derivatives",
        ],
        index=0
    )

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

        # NEW: Market-Implied Move tool (ATM IV -> expected 1W/1M move)
        with st.expander("📈 Market-Implied Move (from ATM IV)", expanded=True):
            if hasattr(op, "run_implied_move"):
                op.run_implied_move()
            elif hasattr(op, "run_implied_move_table"):
                op.run_implied_move_table()
            else:
                st.warning(
                    "Add `run_implied_move()` (or `run_implied_move_table()`) "
                    "to `option_projects.py` to render the implied move table."
                )
