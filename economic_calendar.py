# economic_calendar.py
import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta

def run_economic_calendar():
    st.subheader("📅 Economic Calendar")
    st.markdown(
        "Key upcoming **market-moving economic events** for traders. "
        "Never place a major trade without checking what data is coming out that week. "
        "Update this monthly alongside your Macro Research report."
    )

    with st.expander("📖 How to Use the Economic Calendar", expanded=False):
        st.markdown("""
**Why the Economic Calendar matters:**

Every major economic release can move markets 1-3% in minutes. 
Before placing any trade on Investopedia, check what events are coming:

**High Impact Events (🔴 Red — Avoid large positions the day before):**
- **FOMC Meeting** — Fed rate decision. Most important event in markets.
- **CPI** — Inflation data. Moves bonds, stocks, and dollar significantly.
- **NFP (Non-Farm Payrolls)** — Jobs report. First Friday of every month at 8:30 AM ET.
- **GDP** — Quarterly economic growth. Major market mover.
- **PCE** — Fed's preferred inflation gauge. Released monthly.

**Medium Impact Events (🟡 Yellow — Be aware, reduce size):**
- **PPI** — Producer price inflation. Leading indicator for CPI.
- **Retail Sales** — Consumer spending. 70% of US GDP.
- **ISM Manufacturing/Services** — Business activity indicators.
- **JOLTS** — Job openings. Labor market health.
- **Consumer Confidence** — Leading indicator for spending.

**Trading Rules Around Events:**
1. **Never enter a new position the day before a high impact event**
2. **Reduce existing positions by 50% before FOMC and CPI**
3. **Wait 30 minutes after the release** before trading the reaction
4. **Use your Implied Move tool** to size options around events
""")

    # ---- Current Month Events ----
    st.markdown("### 🗓️ June–July 2026 Key Events")
    st.caption("Update this section monthly alongside your Macro Research report.")

    events = [
        {"Date": "Jun 16-17", "Event": "FOMC Meeting — Kevin Warsh First Meeting", "Impact": "🔴 High", "Asset": "All Markets",
         "Expected": "Hold at 4.25-4.50%", "Notes": "Warsh's first meeting as Fed Chair. Markets watching for hawkish signals. Any rate hike language = USD up, bonds down, stocks down."},
        {"Date": "Jun 18", "Event": "CPI (May 2026)", "Impact": "🔴 High", "Asset": "Bonds, USD, Stocks",
         "Expected": "3.7% YoY", "Notes": "Current CPI at 3.8%. Any upside surprise = yields up, stocks down, USD up. Downside = yields down, stocks up."},
        {"Date": "Jun 20", "Event": "PCE Price Index (May 2026)", "Impact": "🔴 High", "Asset": "Bonds, USD",
         "Expected": "2.8% YoY", "Notes": "Fed's preferred inflation gauge. Above 3% = hawkish concern. Below 2.5% = dovish pivot expectations."},
        {"Date": "Jun 27", "Event": "GDP (Q1 2026 Final)", "Impact": "🔴 High", "Asset": "USD, Stocks",
         "Expected": "+0.8% annualized", "Notes": "Final revision. Below 0% = recession confirmation. Watch for consumer spending component."},
        {"Date": "Jul 3", "Event": "NFP (June 2026)", "Impact": "🔴 High", "Asset": "All Markets",
         "Expected": "185,000 jobs", "Notes": "Above 250k = hawkish Fed concern. Below 100k = recession fear. Wage growth above 4% = inflation concern."},
        {"Date": "Jul 9", "Event": "CPI (June 2026)", "Impact": "🔴 High", "Asset": "Bonds, USD, Stocks",
         "Expected": "3.5% YoY", "Notes": "Key for H2 2026 Fed path. If below 3%, rate cut expectations return. Above 4% = possible hike."},
        {"Date": "Jul 15", "Event": "Retail Sales (June 2026)", "Impact": "🟡 Medium", "Asset": "Consumer stocks, USD",
         "Expected": "+0.3% MoM", "Notes": "70% of GDP is consumer spending. Weak print = recession risk elevated."},
        {"Date": "Jul 18", "Event": "Michigan Consumer Sentiment", "Impact": "🟡 Medium", "Asset": "Consumer stocks",
         "Expected": "67.5", "Notes": "Below 65 = significant consumer stress. Inflation expectations component watched closely."},
        {"Date": "Jul 28-29", "Event": "FOMC Meeting", "Impact": "🔴 High", "Asset": "All Markets",
         "Expected": "Hold", "Notes": "Second Warsh meeting. If CPI has fallen, first cut hints possible. If sticky, possible hike discussion."},
        {"Date": "Jul 30", "Event": "GDP (Q2 2026 Advance)", "Impact": "🔴 High", "Asset": "USD, Stocks",
         "Expected": "+1.2% annualized", "Notes": "First read on Q2. Below 0% = two consecutive negative quarters = technical recession."},
    ]

    df_events = pd.DataFrame(events)

    # Color by impact
    impact_colors = {"🔴 High": "#e74c3c", "🟡 Medium": "#f0b429", "🟢 Low": "#2ecc71"}

    for _, row in df_events.iterrows():
        color = "#e74c3c" if "High" in row["Impact"] else "#f0b429"
        st.markdown(
            f"""<div style="border-left:4px solid {color};padding:12px 16px;
            margin-bottom:8px;background:{color}11;border-radius:0 6px 6px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:700;color:white;font-size:1rem;">{row['Event']}</span>
                <span style="color:{color};font-weight:600;">{row['Impact']}</span>
            </div>
            <div style="color:#aaa;font-size:0.85rem;margin-top:4px;">
                📅 {row['Date']} &nbsp;|&nbsp; 📊 {row['Asset']} &nbsp;|&nbsp; 
                Expected: <span style="color:#f0b429;">{row['Expected']}</span>
            </div>
            <div style="color:#ccc;font-size:0.82rem;margin-top:6px;">{row['Notes']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    # ---- Weekly Checklist ----
    st.markdown("### ✅ Weekly Pre-Trade Checklist")
    st.markdown("""
Before placing any trade on Investopedia this week, confirm:

- [ ] **No FOMC meeting or CPI within 48 hours**
- [ ] **NFP week? Reduce position sizes by 50%**
- [ ] **Check VIX** — above 20 = elevated risk, above 25 = high risk event possible
- [ ] **Check Market Regime** — is current signal consistent with your trade direction?
- [ ] **Check Implied Move** — for options trades, is the move already priced in?
- [ ] **Check Technical Signals** — MACD and RSI confirm your direction?
""")

    # ---- Key Data Release Times ----
    st.markdown("### ⏰ Key Release Times (ET)")
    times_data = {
        "Release": ["FOMC Decision", "CPI", "NFP", "GDP", "PCE", "Retail Sales", "PPI", "ISM Mfg", "JOLTS"],
        "Time (ET)": ["2:00 PM", "8:30 AM", "8:30 AM (1st Fri)", "8:30 AM", "8:30 AM", "8:30 AM", "8:30 AM", "10:00 AM", "10:00 AM"],
        "Frequency": ["8x/year", "Monthly", "Monthly", "Quarterly", "Monthly", "Monthly", "Monthly", "Monthly", "Monthly"],
        "Impact": ["🔴🔴🔴", "🔴🔴🔴", "🔴🔴🔴", "🔴🔴", "🔴🔴🔴", "🔴🔴", "🔴🔴", "🔴🔴", "🔴🔴"],
    }
    st.dataframe(pd.DataFrame(times_data), use_container_width=True)

    # ---- Fed Meeting Dates 2026 ----
    st.markdown("### 🏦 FOMC Meeting Dates 2026")
    fomc_dates = {
        "Meeting": ["Jan 28-29", "Mar 18-19", "May 6-7", "Jun 16-17 ← NOW", "Jul 28-29", "Sep 15-16", "Oct 27-28", "Dec 9-10"],
        "Decision Date": ["Jan 29", "Mar 19", "May 7", "Jun 17", "Jul 29", "Sep 16", "Oct 28", "Dec 10"],
        "Expected Action": ["Hold", "Hold", "Hold", "Hold", "Hold", "Possible Cut", "Possible Cut", "TBD"],
        "Probability": ["95% Hold", "90% Hold", "88% Hold", "92% Hold", "75% Hold", "60% Hold / 40% Cut", "55% Hold / 45% Cut", "TBD"],
    }
    st.dataframe(pd.DataFrame(fomc_dates), use_container_width=True)

    st.warning("""
⚠️ **Important:** This calendar is manually maintained. 
Update it monthly when you update your Macro Research report. 
For real-time economic calendar data, use **Investing.com/economic-calendar** or **Forex Factory**.
""")

    st.caption(f"Last updated: June 18, 2026 | Update monthly alongside Macro Research report")
