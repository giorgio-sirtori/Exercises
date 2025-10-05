import streamlit as st
import yfinance as yf
from datetime import datetime


st.set_page_config(
    page_title="Live Cost of Carry Calculator",
    layout="wide"
)

# --- Data Constants ---
INTEREST_RATES = {
    'EUR': 0.0425, # ECB Deposit Facility Rate
    'AUD': 0.0435, # Reserve Bank of Australia Cash Rate
    'DKK': 0.0360, # Danmarks Nationalbank Certificates of Deposit Rate
    'GBP': 0.0525, # Bank of England Bank Rate
    'NOK': 0.0450, # Norges Bank Policy Rate
    'PLN': 0.0575, # National Bank of Poland Reference Rate
    'USD': 0.0550, # US Fed Funds Rate (Upper Bound)
    'CHF': 0.0150, # Swiss National Bank Policy Rate
    'CZK': 0.0525, # Czech National Bank 2-week Repo Rate
    'HKD': 0.0575, # Hong Kong Monetary Authority Base Rate
    'HUF': 0.0700, # Hungarian National Bank Base Rate
    'SEK': 0.0375, # Sveriges Riksbank Policy Rate
    'ZAR': 0.0825  # South African Reserve Bank Repo Rate
}

FX_PAIRS_TO_FETCH = {
    'AUD': 'EURAUD=X', 'DKK': 'EURDKK=X', 'GBP': 'EURGBP=X', 'NOK': 'EURNOK=X',
    'PLN': 'EURPLN=X', 'USD': 'EURUSD=X', 'CHF': 'EURCHF=X', 'CZK': 'EURCZK=X',
    'HKD': 'EURHKD=X', 'HUF': 'EURHUF=X', 'SEK': 'EURSEK=X', 'ZAR': 'EURZAR=X'
}

# --- Data Fetching ---
@st.cache_data(ttl=600)
def fetch_spot_rates():
    """Fetches the latest spot rates from Yahoo Finance."""
    tickers = list(FX_PAIRS_TO_FETCH.values())
    data = yf.download(tickers, period='1d', progress=False)
    if data.empty:
        st.error("Could not fetch live FX data. Please check your connection or try again later.")
        return None

    latest_prices = data['Close'].iloc[-1]

    spot_rates = {}
    for base_curr, ticker in FX_PAIRS_TO_FETCH.items():
        if ticker in latest_prices and latest_prices[ticker] > 0:
            spot_rates[base_curr] = 1 / latest_prices[ticker]
        else:
            # FIX: Changed 0 to 0.0 to ensure the function always returns floats.
            # This prevents the StreamlitMixedNumericTypesError in st.number_input.
            spot_rates[base_curr] = 0.0

    return spot_rates

# --- Core Logic ---
def calculate_cost_of_carry(spot_rate, domestic_rate, foreign_rate, days, day_count):
    """Calculates the Cost of Carry for a currency pair."""
    if day_count == 0 or spot_rate == 0:
        return 0.0
    return spot_rate * (domestic_rate - foreign_rate) * (days / day_count)

# --- Main App ---
if 'spot_rates' not in st.session_state:
    st.session_state.spot_rates = fetch_spot_rates()

st.title("Live Currency Cost of Carry Calculator")
st.markdown(f"This tool fetches **live exchange rates** to calculate the net cost or gain of holding a currency position. Interest rates are recent central bank policy rates.")
st.caption(f"Data last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Parameters")

    if st.button("Refresh Live Spot Rates"):
        st.session_state.spot_rates = fetch_spot_rates()
        st.rerun()

    base_currency_list = list(FX_PAIRS_TO_FETCH.keys())
    selected_base_curr = st.selectbox(
        "Select Base Currency (to go long/buy)",
        options=base_currency_list,
        index=base_currency_list.index('USD')
    )

    quote_currency = "EUR"

    st.write(f"**Selected Pair:** `{selected_base_curr}/{quote_currency}`")
    st.info(f"A 'long' position means you are buying **{selected_base_curr}** and selling **{quote_currency}**.")

    live_spot_rate = st.session_state.spot_rates.get(selected_base_curr, 0.0) if st.session_state.spot_rates else 0.0

    st.subheader("Adjust Values")

    spot_rate = st.number_input(
        f"Spot Rate ({selected_base_curr} per {quote_currency})",
        min_value=0.0,
        value=live_spot_rate,
        step=0.0001,
        format="%.4f",
        help="This value is fetched live. You can override it here."
    )

    foreign_rate = st.number_input(
        f"Interest Rate for **{selected_base_curr}** (Base Currency)",
        min_value=-0.05, max_value=1.0, value=INTEREST_RATES.get(selected_base_curr, 0.0), step=0.0001, format="%.4f"
    )

    domestic_rate = st.number_input(
        f"Interest Rate for **{quote_currency}** (Quote Currency)",
        min_value=-0.05, max_value=1.0, value=INTEREST_RATES.get(quote_currency, 0.0), step=0.0001, format="%.4f"
    )

    days = st.slider("Holding Period (Days)", min_value=1, max_value=365, value=90)
    day_count = st.selectbox("Day Count Convention", options=[360, 365], index=0)


# --- Results Display ---
st.markdown("---")
st.header("Results")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Calculation Inputs")
    st.metric(label="Pair", value=f"{selected_base_curr}/{quote_currency}")
    st.metric(label="Live Spot Rate", value=f"{spot_rate:.4f} {quote_currency}")
    st.metric(label="Holding Period", value=f"{days} days")
    st.metric(label="Interest Rate Differential (Quote - Base)", value=f"{(domestic_rate - foreign_rate) * 100:.2f} %")

with col2:
    st.subheader("Cost of Carry Analysis")
    carry_cost = calculate_cost_of_carry(spot_rate, domestic_rate, foreign_rate, days, day_count)

    if carry_cost >= 0:
        st.metric(
            label=f"Net COST for {days} days",
            value=f"€ {carry_cost:,.5f}",
            help="A positive result means it costs you money to hold this long position."
        )
        st.error(f"**Negative Carry:** Holding a long position in {selected_base_curr} results in a **net cost** of **€{carry_cost:.5f}** over {days} days. You are paying more on the currency you sold (EUR) than you are earning on the currency you bought ({selected_base_curr}).")
    else:
        carry_gain = -carry_cost
        st.metric(
            label=f"Net GAIN for {days} days",
            value=f"€ {carry_gain:,.5f}",
            help="A negative cost means you have a net gain from holding the position."
        )
        st.success(f"**Positive Carry:** Holding a long position in {selected_base_curr} results in a **net gain** of **€{carry_gain:.5f}** over {days} days. This is a **positive carry trade**, as the interest earned on {selected_base_curr} is higher than the interest paid on EUR.")
