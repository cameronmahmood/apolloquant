# option_projects.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def run_black_scholes():
    st.subheader("📈 Black-Scholes Option Pricer with P&L Heatmap")
    st.markdown("Use this tool to visualize **option prices** and **P&L surfaces** using the Black-Scholes model.")

    # Sidebar Inputs
    st.sidebar.header("Option Inputs")
    S = st.sidebar.number_input("Asset Price (S)", value=100.0, step=1.0)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
    T = st.sidebar.number_input("Time to Maturity (in years)", value=1.0, step=0.1)
    r = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100
    sigma = st.sidebar.number_input("Volatility (%)", value=20.0, step=0.1) / 100

    st.sidebar.header("Optional Trade Inputs")
    call_price_paid = st.sidebar.number_input("Call Purchase Price", value=0.0)
    put_price_paid = st.sidebar.number_input("Put Purchase Price", value=0.0)

    plot_type = st.sidebar.radio("Heatmap Type", options=["Option Value", "Call P&L"])

    # Calculate Prices
    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")

    st.markdown(f"""
    ### 🧮 Option Prices
    - **Call Price:** ${call_price:.2f}  
    - **Put Price:** ${put_price:.2f}
    """)

    if call_price_paid > 0 or put_price_paid > 0:
        st.markdown(f"""
        ### 💸 Implied P&L (Based on Purchase Price)
        - Call P&L: ${call_price - call_price_paid:.2f}
        - Put P&L: ${put_price - put_price_paid:.2f}
        """)

    # Heatmap
    st.subheader("📊 Heatmap Visualization")
    S_range = np.linspace(S * 0.8, S * 1.2, 30)
    sigma_range = np.linspace(sigma * 0.5, sigma * 1.5, 30)
    heatmap = np.zeros((len(S_range), len(sigma_range)))

    for i, s_val in enumerate(S_range):
        for j, sig_val in enumerate(sigma_range):
            price = black_scholes(s_val, K, T, r, sig_val, "call")
            heatmap[i, j] = price - call_price_paid if plot_type == "Call P&L" and call_price_paid > 0 else price

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap,
        xticklabels=np.round(sigma_range, 2),
        yticklabels=np.round(S_range, 2),
        cmap="RdYlGn" if plot_type == "Call P&L" and call_price_paid > 0 else "YlGnBu",
        ax=ax
    )
    plt.xlabel("Volatility")
    plt.ylabel("Asset Price")
    plt.title(f"{plot_type} Heatmap")
    st.pyplot(fig)


# --- Utility function ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# Placeholder for second project

def run_monte_carlo():
    st.subheader("🎲 Monte Carlo Simulation for Option Pricing")
    st.info("This section is under construction.")
