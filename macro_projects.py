# macro_projects.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# =========================
# Live Price Fetching
# =========================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_live_prices():
    """Fetch live market prices for macro dashboard."""
    tickers = {
        "WTI Oil": "CL=F",
        "DXY": "DX-Y.NYB",
        "10Y Yield": "^TNX",
        "2Y Yield": "^IRX",
        "S&P 500": "^GSPC",
        "VIX": "^VIX",
    }
    prices = {}
    changes = {}
    for name, ticker in tickers.items():
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="2d", auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if not hist.empty and "Close" in hist.columns:
                latest = float(hist["Close"].dropna().iloc[-1])
                prev = float(hist["Close"].dropna().iloc[-2]) if len(hist) > 1 else latest
                prices[name] = latest
                changes[name] = ((latest - prev) / prev) * 100 if prev != 0 else 0.0
            else:
                prices[name] = None
                changes[name] = None
        except Exception:
            prices[name] = None
            changes[name] = None
    return prices, changes

def _display_live_prices():
    """Display live price metrics at the top of the macro dashboard."""
    st.markdown("### 📡 Live Market Snapshot")
    with st.spinner("Fetching live prices..."):
        prices, changes = _fetch_live_prices()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    cols = [col1, col2, col3, col4, col5, col6]
    labels = ["WTI Oil", "DXY", "10Y Yield", "2Y Yield", "S&P 500", "VIX"]
    formats = ["${:.2f}", "{:.2f}", "{:.2f}%", "{:.2f}%", "{:,.0f}", "{:.2f}"]

    for col, label, fmt in zip(cols, labels, formats):
        price = prices.get(label)
        change = changes.get(label)
        with col:
            if price is not None:
                delta = f"{change:+.2f}%" if change is not None else None
                col.metric(
                    label=label,
                    value=fmt.format(price),
                    delta=delta,
                )
            else:
                col.metric(label=label, value="N/A")

    st.caption(
        f"Last updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} "
        "| Prices delayed ~15 min | Source: Yahoo Finance"
    )

# =========================
# Main Macro Research Page
# =========================

def run_macro_research():
    st.subheader("🌍 Cross-Asset Market Outlook")
    st.markdown(
        "**Updated June 16, 2026 | Cameron Mahmood — Delphi Macro**\n\n"
        "Bullish and bearish cases across Oil, the US Dollar, Treasury Yields, and Equities. "
        "Live market prices are pulled automatically at the top of each section."
    )

    # Live prices
    _display_live_prices()
    st.divider()

    # Big Picture
    with st.expander("🔭 Big Picture Framework", expanded=True):
        _big_picture()

    # Oil
    with st.expander("🛢️ Oil (WTI / Brent)", expanded=False):
        _oil_section()

    # Dollar
    with st.expander("💵 US Dollar (DXY)", expanded=False):
        _dollar_section()

    # 2Y Yield
    with st.expander("📈 2-Year Treasury Yield", expanded=False):
        _two_year_section()

    # 10Y Yield
    with st.expander("📉 10-Year Treasury Yield", expanded=False):
        _ten_year_section()

    # S&P 500
    with st.expander("📊 S&P 500 (SPX)", expanded=False):
        _spx_section()

    # Final Positioning
    with st.expander("📋 Final Positioning Summary", expanded=False):
        _positioning_summary()

    # Indicators
    with st.expander("🔍 Key Indicators to Monitor Weekly", expanded=False):
        _indicators_section()

    st.caption(
        "Sources: EIA STEO (June 2026), Federal Reserve FOMC Minutes (April 2026), "
        "Goldman Sachs Research, Morgan Stanley Mid-Year Outlook, Citi Equity Strategy, "
        "UBS, Nuveen Fixed Income, Charles Schwab Mid-Year 2026, FactSet EarningsInsight, "
        "TradingEconomics, OPEC Monthly Oil Market Report (June 2026)"
    )

# =========================
# Section Functions
# =========================

def _big_picture():
    st.markdown("""
The dominant macro shift from early 2026 is the **US–Iran peace agreement announced June 16, 2026**, 
triggering a sharp repricing across all asset classes. Brent crude fell toward the low-$80s, 
the 10-year Treasury yield touched a one-month low at ~4.43%, the S&P 500 rallied sharply, 
and the dollar slipped to its weakest level in over a week.

**The single most important cross-asset relationship right now:**

> 🟢 **Oil down → inflation expectations down → yields down → S&P up → dollar weaker**

> 🔴 **Oil up → inflation expectations up → Fed hawkish → yields up → S&P pressured → dollar stronger**

Everything flows from oil. The key question for H2 2026 is whether the Strait of Hormuz reopens 
smoothly and Iranian barrels return to market, or whether geopolitical risk re-emerges and energy 
prices rebound. That single variable will determine the path of inflation, the Fed, equities, and the dollar.

**Key macro backdrop entering June 16, 2026:**
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
- WTI near **$80/barrel** — down ~5% on peace deal; Brent averaged $103–$107 in May
- DXY **~99.5** — firmer than 2026 forecasts but softening on risk-on
- 10Y Treasury yield **~4.43%** — one-month low after peace deal
- 2Y Treasury yield **~4.06%**
""")
    with col2:
        st.markdown("""
- S&P 500 **~7,554** — near all-time highs; +13% since late March 2026
- CPI **3.8% YoY** — well above Fed's 2% target
- Fed on hold — **Kevin Warsh's first FOMC** June 16–17, 2026
- Some traders pricing a **possible December 2026 rate hike**
""")


def _oil_section():
    st.markdown("### Current View: Neutral to Bearish Short-Term | Bullish if Geopolitical Risk Returns")
    st.markdown("""
Oil is no longer just an OPEC+ supply story. The biggest current driver is the Strait of Hormuz 
situation. The near-closure since late February 2026 disrupted ~20% of global oil shipments. 
The US–Iran peace agreement signed June 19 removes the largest risk premium in the market — at least temporarily.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Bullish Case")
        st.markdown("""
**Geopolitical Risk Remains**
- Peace deal details and timeline remain uncertain — full Hormuz normalization not guaranteed
- Mine clearance, infrastructure repairs, and production restoration could take months
- Any renewed disruption immediately restores the risk premium
- OPEC March 2026 output fell to its lowest level since June 2020

**OPEC+ Discipline**
- OPEC Reference Basket averaged $114.55/barrel in May 2026
- Saudi Arabia, Russia, Iraq need $80–$90/barrel to fund national budgets
- OPEC still expects global demand growth of ~970,000 bpd in 2026
- Retains spare capacity as a policy lever; willing to cut if prices fall below $80

**Structural Supply Underinvestment**
- ESG pressure limits non-OPEC output growth to US, Brazil, and Guyana only
- US shale rig count remains subdued — slow reactivation pace
- Long-term capex in fossil fuel exploration structurally lower across the industry

**Demand Resilience**
- EIA projects US net exports averaging 4.2M b/d in 2026 — up 1.4M b/d from 2025
- Diesel and jet fuel wholesale prices up 60%+ vs. pre-conflict February STEO
- India, Vietnam, Brazil driving demand growth; SPR restocking adds demand below $80

**Indicators to Watch**
- Brent back above $85–$90
- Any renewed Strait of Hormuz disruption
- Falling US crude inventories (EIA weekly)
- OPEC+ delaying supply increases
- Strong China/India crude import data
- Brent-WTI spread tightening (signal of global demand)

**Trade Structures**
- Long crude WTI/Brent futures (CL, BZ)
- Call spreads on USO or Brent ETFs
- Long XLE, XOM, CVX
""")

    with col2:
        st.markdown("#### 🔴 Bearish Case")
        st.markdown("""
**Geopolitical De-escalation**
- US–Iran peace agreement June 16 — signed in Switzerland June 19
- Includes lifting blockades, sanctions relief for Iran, dismantling nuclear program
- WTI dropped 5%+ to ~$80 on announcement
- Iranian barrels returning to market could add 1M+ bpd of supply
- JP Morgan sees Brent averaging ~$60/bbl under full normalization

**Demand Weakness**
- US GDP tracked 1.1–1.7% YoY through 2025–2026
- China crude imports from Saudi Arabia: 1.4M → 333k bpd
- German industrial recession; Eurozone GDP at 0.9–1.1%
- PMI sub-50 across Eurozone and US manufacturing for 17 of past 18 months

**Supply Recovery Risk**
- US shale reactivates quickly above $75–$80 — rapid supply response
- EIA expects 2026 oil demand growth of only 0.8M bpd (downward revision)
- IEA cut global demand forecast to 720k bpd
- Futures curves reflecting expectations of supply normalization

**Indicators to Watch**
- Brent breaks below $80 (key OPEC floor)
- Hormuz shipping normalizes — tanker traffic data
- OPEC+ restoring supply faster than expected
- Rising DOE/EIA inventory builds
- Baker Hughes rig count rising
- Energy equities lagging broader market

**Trade Structures**
- Short CL/BZ via futures
- Put spreads on USO
- Short XLE or oil-beta equities (SLB, HAL)
""")


def _dollar_section():
    st.markdown("### Current View: Neutral to Bearish | Unless Fed Turns Hawkish Again")
    st.markdown("""
DXY near 99.5 — firmer than the Goldman Sachs/JPMorgan consensus of low-90s at the start of 2026, 
but softening after the US–Iran deal reduced safe-haven demand. The dollar is caught between two 
competing forces: **hot inflation supporting the dollar** and **risk-on/de-escalation weakening it.**
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Bullish Case")
        st.markdown("""
**Fed on Hold — New Chair Kevin Warsh**
- Warsh's first FOMC meeting: June 16–17, 2026
- Rate cuts off the table; some pricing a possible December rate hike
- April 2026 FOMC minutes: ~30% probability of rate hike by Q1 2027
- Nuveen pushed first cut to H1 2027 after resilient May jobs + sticky CPI at 3.8%
- Payrolls averaging ~190k/month — labor market not breaking

**Hot Inflation = Hawkish Fed = Strong Dollar**
- May CPI rose 4.2% YoY — fastest in three years
- Services inflation 5.3% YoY; shelter still 3.9%+ YoY (OER)
- Structural forces: deglobalization, reshoring, wage pressure, green transition costs

**US Relative Advantage**
- US real yields still higher than Bunds, Gilts, and JGBs
- Largest consumer economy with better growth relative to EU/Japan
- Safe-haven bid from residual geopolitical risk (Taiwan, Middle East uncertainty)

**Indicators to Watch**
- CPI remains above 4%; Core PCE reaccelerates
- Warsh signals hawkish framework at June FOMC
- NFP above 200k — strong labor = Fed holds
- DXY retakes 100–102
- Risk-off flows return

**Trade Structures**
- Long DXY futures or UUP ETF
- Long USDJPY
- Short EURUSD if Fed stays hawkish
""")

    with col2:
        st.markdown("#### 🔴 Bearish Case")
        st.markdown("""
**Peace Deal = Risk-On = Dollar Weakness**
- US–Iran deal sent DXY to weakest in over a week
- Oil falling reduces US inflation fears → space for eventual cuts
- Risk-on pushes capital into EM and non-USD assets

**Fed Eventually Cuts — Long-Term Dollar Headwind**
- Median FOMC survey: two 25bps cuts over next year (Q3/Q4 2026 and Q1 2027)
- MUFG projects DXY down ~5% in 2026 (EUR/USD to 1.24, USD/JPY to 146)
- BoJ hiking this week; ECB already hiked in June — closing rate differential with USD

**US Fiscal and Sovereign Risk**
- Moody's downgraded US to Aa1 in May 2025
- US debt/GDP near 100%; FY2025 deficit $1.8–$2 trillion
- Net interest outlays approaching $1 trillion annually
- Jamie Dimon warning of "crack in bond market"
- Ray Dalio predicts "economic heart attack" within 3 years without deficit reduction
- Dollar's convenience yield and safety premium have eroded

**Indicators to Watch**
- Oil continues falling below $80
- Fed hike odds fade after June FOMC
- EUR/USD and AUD/USD strengthen further
- DXY breaks below 98–99
- Global equities rally — reduces safe-haven bid

**Trade Structures**
- Short DXY/UUP
- Long EURUSD, AUDUSD
- Long EMFX (BRL, ZAR, MXN)
- Long gold or bitcoin as anti-dollar hedges
""")


def _two_year_section():
    st.markdown("### Current View: Neutral to Bullish Yield Risk | Most Fed-Sensitive Instrument")
    st.markdown("""
The 2-year Treasury yield (~4.06%) is the most sensitive instrument to Fed policy expectations. 
It remains elevated because markets are pricing a prolonged hold — not cuts. 
The 2Y is now a fight between **hot inflation** (keeping yields up) and 
**falling oil/geopolitical de-escalation** (which could cool inflation and bring yields down).
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Bullish Case (Yields Up)")
        st.markdown("""
**Fed On Hold Longer Than Expected**
- Nuveen: no rate cut in 2026 — first move pushed to H1 2027
- May jobs averaged ~190k/month — above ~100k neutral threshold
- CPI at 3.8% removes case for any pivot under Warsh
- April 2026 FOMC: ~30% probability of rate hike by Q1 2027
- SOFR futures barely pricing cuts in 2026

**Structural Inflation Forces**
- Services inflation 5.3% YoY; shelter (OER) still 3.9%+ YoY
- Wage growth 4.0–4.5% YoY — structural labor shortage in healthcare, construction, education
- Deglobalization and reshoring (CHIPS Act, IRA) raising domestic production costs
- Green transition creating commodity bottlenecks — copper up 20% YTD 2025

**Indicators to Watch**
- CPI MoM > 0.3%; Core PCE > 2.5% YoY
- Warsh signals hawkish at June FOMC
- JOLTS, NFP, wage growth staying elevated
- Fed Dot Plot shifting higher
- 2Y moves above 4.15%–4.25%

**Trade Structures**
- Short TU (2Y Treasury futures)
- Pay fixed in 2Y swaps
- SOFR puts (rates rising bets)
""")

    with col2:
        st.markdown("#### 🔴 Bearish Case (Yields Down)")
        st.markdown("""
**Peace Deal Disinflation**
- Oil to $80/barrel → reduces energy inflation component directly
- Falling oil relieves pressure on goods inflation and transportation costs
- If Hormuz fully reopens, global energy costs normalize → disinflation path clears
- Lower oil → lower CPI → Fed gets cover to cut → 2Y yield falls rapidly

**Growth Slowing — Recession Risk**
- US GDP Q1 2025 contracted; Q2 2025 +0.9% annualized — below trend
- Conference Board LEI fell 23 straight months — longest since 2007–2009
- ISM Manufacturing at 47.3 — 17 of 18 months below 50
- Payrolls slowing to 130k/month; unemployment from 3.7% to 4.1%
- Sahm Rule recession trigger at 4.4% unemployment — getting close

**Indicators to Watch**
- Weak NFP (below 100k) or rising jobless claims
- Falling Core PCE (below 2.5%)
- Oil stays below $80–$83
- Dovish signals from Warsh at June FOMC
- 2Y breaks below 4.00%

**Trade Structures**
- Long TU or EDZ4
- Receive SOFR 1Y–2Y
- Call spreads on ED or FF futures
""")


def _ten_year_section():
    st.markdown("### Current View: Neutral | Upside Risk from Inflation and Fiscal Concerns")
    st.markdown("""
The 10-year yield (~4.43%) is less about the Fed than the 2-year. It is driven by 
**inflation expectations, term premium, Treasury supply, and recession risk.** 
The peace deal pushed yields to a one-month low, but structural forces keep long-end yields elevated.
Charles Schwab mid-year: *"We see more risks to the upside than the downside."*
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Bullish Case (Yields Up)")
        st.markdown("""
**Term Premium Rising — Fiscal Risk**
- NY Fed term premium model: ~0% in 2023 → 0.56–0.84% by mid-2025
- Moody's downgraded US to Aa1 in May 2025 — fading confidence in dollar sovereignty
- FY2025 deficit: $1.8–$2 trillion (6.5% of GDP) — historic in a non-crisis year
- Net interest outlays approaching $1 trillion annually — debt spiral dynamic
- August 2026 refunding expected to show continued long-end issuance

**Heavy Treasury Supply + Waning Demand**
- US Treasury issued ~$500B net new debt in Q2 2025 alone
- Recent 10Y auctions: bid-to-cover 2.52 — below 2.56 long-run average
- China's Treasury holdings: $1.1T (2021) → $775B (2025)
- Japan reduced holdings by $25B Jan–June 2025
- Fed QT running off $60B/month — removing largest historical buyer

**Global Inflation Persistence**
- Services inflation 5.3% YoY; shelter rising 4.5–5% YoY
- Deglobalization, green transition, wage pressure keeping inflation elevated structurally
- 5y5y forward inflation expectations at 2.35–2.45% — above Fed's 2% target
- Copper up 20% YTD 2025; EV battery metals in structural deficit by 2026

**Indicators to Watch**
- CPI stays above 4%; oil rebounds above $85–$90
- Fed refuses to cut; Warsh signals restrictive stance
- Treasury auctions show weak demand (low bid-to-cover, wide tails)
- 10Y breaks above 4.60%–4.70%
- 2s10s spread steepening

**Trade Structures**
- Short TY futures / TLT
- 2s10s steepener (buy 2Y, short 10Y)
- Bear call spreads on TLT
""")

    with col2:
        st.markdown("#### 🔴 Bearish Case (Yields Down)")
        st.markdown("""
**Peace Deal = Oil Down = Disinflation = Yields Fall**
- 10Y already fell to ~4.43% on June 16 after peace announcement
- Oil to $80/barrel removes largest inflation risk premium in bond market
- If Brent normalizes to $65–$70 (JP Morgan base case), disinflation accelerates sharply

**Recession or Growth Scare**
- GDP tracking 0.9–1.7% — well below 1.8–2.0% neutral growth
- Conference Board LEI: 23 consecutive monthly declines
- ISM Manufacturing: 47.3 — New Orders at 45.1; Employment at 47.8
- Unemployment from 3.7% to 4.1%; approaching Sahm Rule trigger at 4.4%
- Historical: COVID (10Y 1.9% → 0.5%); SVB collapse (4.0% → 3.4% in days)

**CRE Credit Risk — Systemic Concern**
- Office vacancy: 19.8% nationally (record); San Francisco 30%, Manhattan 22%
- $1.5T in commercial mortgages maturing by 2026 — underwritten at 3–4%, refinancing at 7–8%
- Regional banks hold 70% of CRE loans — default spike = contagion risk
- CRE CMBS delinquency rate >6% — highest since 2013

**Duration Underweight = Squeeze Risk**
- CFTC CoT: leveraged funds net short ~200,000 TY contracts
- Asset managers 1 std dev below average duration (JPMorgan Client Survey)
- Any growth scare forces underweights to buy duration rapidly

**Indicators to Watch**
- Brent falls below $80 and stays there
- CPI cools to 3% or below
- Growth slows; credit spreads widen
- VIX spikes above 20
- 10Y breaks below 4.40% then 4.25%

**Trade Structures**
- Long TY futures or TLT
- 10s30s flattener
- Long ZF/TY call spreads
""")


def _spx_section():
    st.markdown("### Current View: Bullish Momentum | But Valuation Risk is Real")
    st.markdown("""
The S&P 500 closed around **7,554** on June 15, near all-time highs, after rallying ~13% since 
late March 2026 — the sharpest rise since April 2020. Goldman Sachs raised its year-end 2026 
target to **8,000**; Citi raised to **8,100**; UBS to **7,900**; Morgan Stanley projects a 
**12% advance** over the next 12 months. But Forward P/E at 20–21x is above the 20-year 
average of 16.5x — the market is **not cheap.**
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Bullish Case")
        st.markdown("""
**AI-Driven CapEx Super-Cycle — The Core Driver**
- Goldman: AI infrastructure to drive ~40–50% of S&P 500 EPS growth in 2026
- Hyperscaler CapEx: $670B for 2026 — 83% increase from 2025; 2027 estimates $905B
- FactSet: "AI" cited on 337 S&P 500 earnings calls — highest ever; 5-year average was 164
- Q1 2026 earnings beat rate: 6% above expectations — strongest beat in 4 years

**Mega-Cap AI Leaders**
- NVIDIA: Data center revenue >$20B/quarter; Blackwell GPUs sold out into 2026; $3.6T market cap
- Microsoft: >$50B CapEx FY2025 for Azure AI; Copilot monetization driving recurring revenue
- Amazon: $12B committed to India AI alone; Trainium chips and Bedrock scaling
- Google: $45–$50B CapEx in 2026 for Gemini, Vertex AI, TPUs; Cloud AI >27% YoY

**Earnings and Valuation**
- Goldman projects S&P 500 EPS of $340 in 2026 (24% YoY) and $385 in 2027
- Citi raised 2025 EPS forecast to $350/share after stronger-than-expected results
- Peace deal → lower oil → lower costs → margin expansion across sectors

**Market Breadth and Positioning**
- Rotation beyond Mag 7 into small caps, industrials, financials, healthcare
- Russell 2000 at new highs; small cap EPS growth expected at +20%
- Record $1.1T in corporate buybacks projected for 2026
- BoA Fund Manager Survey: net equity allocation at -10% vs. +20% long-term average — dry powder
- CTAs only 60–70% long equities — below historical exposure; can add longs on trend signals

**Indicators to Watch**
- S&P holds above 7,400–7,500
- 10Y yield stays below 4.5%
- Oil remains below $85
- AI/semis recover and lead
- HY credit spreads remain tight (OAS below 400 bps)
- VIX stays below 15–18

**Trade Structures**
- Long SPX or QQQ
- SPY call spreads
- Long XLK (tech/AI), XLI (industrials/AI CapEx)
- Long IWM (small cap on rate cut expectations)
""")

    with col2:
        st.markdown("#### 🔴 Bearish Case")
        st.markdown("""
**Valuation — Priced for Perfection**
- Forward P/E at ~20–21x — above 20-year historical average of 16.5x
- 2021 precedent: P/E hit 23x → inflation spiked → Fed pivoted → SPX fell 20% in 2022
- 10Y yield at 4.4% provides meaningful risk-free competition for equities
- Equity risk premium is compressed — little margin for error

**Earnings Warning Signs Beneath the Surface**
- Q2 2025 EPS growth slowed to 4.9% — lowest since Q4 2023
- 59 of 110 S&P 500 companies issued negative EPS guidance for Q2 2025
- GM: $1.1B tariff drag; Lockheed Martin EPS: $6.52 expected → $1.46 actual
- Coca-Cola volumes down 1% YoY; Nike, FedEx, Intel all lowered guidance

**Concentration and AI Dependency**
- Magnificent 7 = 30–35% of S&P 500 market cap
- AI accounts for ~40–50% of expected 2026 EPS growth — if cycle disappoints, damage is severe
- Average S&P 500 member max drawdown -21% YTD despite index resilience

**Sticky Inflation = No Fed Rescue**
- June 2026 CPI at 3.8% — well above 2% target
- Warsh FOMC unlikely to signal cuts; some risk of eventual hike
- Credit card APRs at 22–24%; 30-year mortgage rates near 7%

**Geopolitical Risk — China/Taiwan**
- China conducted largest air/naval drills since August 2022 in July 2025
- 78 sorties into Taiwan ADIZ in a single day
- TSMC, ASML, NVDA supply chains at risk if conflict escalates
- July 2025: semiconductor stocks pulled back 6% on Taiwan uncertainty alone

**CRE and Credit Risk**
- CRE CMBS delinquency >6% — highest since 2013
- $540B in CRE loans maturing next 12 months — refinancing at 7–8% from 3–4%
- HY spreads: if OAS widens past 450 bps, equities historically follow
- Regional banks (NYCB, ZION) under pressure — echoing SVB/Signature stress

**Indicators to Watch**
- 10Y yield moves back above 4.6%
- CPI reaccelerates above 4%
- Warsh signals hawkish at June FOMC
- HY credit spreads widen past 400–450 bps
- S&P breaks below 7,300–7,400
- VIX spikes above 20
- CBOE SKEW Index above 140

**Trade Structures**
- Short SPX or long SPX puts
- Long VIX or VIX calls
- Short XLF (financials — CRE exposure)
- Short XRE (real estate — rate sensitivity)
- Long XLV (healthcare), XLP (consumer staples) as defensives
""")


def _positioning_summary():
    st.markdown("""
| Asset | Short-Term Bias | Key Risk | Catalyst to Watch |
|-------|----------------|----------|-------------------|
| **WTI/Brent Oil** | Neutral–Bearish | Hormuz reopens faster than expected | US–Iran deal signing June 19 |
| **DXY (Dollar)** | Neutral | Warsh hawkish surprise | June 16–17 FOMC |
| **2Y Treasury Yield** | Neutral–Bullish | Inflation stays hot | CPI, NFP, Warsh signals |
| **10Y Treasury Yield** | Neutral | Fiscal/supply pressure vs. disinflation | Treasury auctions, oil price |
| **S&P 500** | Bullish | Priced for perfection; concentration risk | AI earnings, 10Y yield path |
""")


def _indicators_section():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
**🛢️ Oil & Energy**
- EIA Weekly Crude Inventory Report (Wednesday)
- Baker Hughes Rig Count (Friday)
- Brent-WTI spread (tighter = bullish global demand)
- Strait of Hormuz tanker traffic data
- CFTC CoT managed money positioning in crude futures
- China crude import data (monthly)
""")

    with col2:
        st.markdown("""
**💵 Dollar & Rates**
- Fed Funds Futures — CME FedWatch rate probabilities
- SOFR futures — near-term rate path pricing
- CPI MoM and YoY (monthly)
- Core PCE (monthly — Fed's preferred measure)
- NFP and jobless claims (weekly/monthly)
- DXY technical: 98–99 support; 101–102 resistance
- Kevin Warsh FOMC statements and press conference
""")

    with col3:
        st.markdown("""
**📊 Equities**
- S&P 500 forward P/E (FactSet weekly)
- HY credit spreads — ICE BofA OAS (Bloomberg)
- VIX — fear gauge; above 20 = caution zone
- CBOE SKEW Index — tail risk; above 140 = elevated
- Earnings revision trends (FactSet EarningsInsight)
- AI capex guidance: NVDA, MSFT, GOOGL, AMZN
- 10Y Treasury yield vs. S&P forward earnings yield
""")