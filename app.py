# Complete Streamlit App for Nifty 50 Stock Market Analyzer (Final Structure with Sections)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from datetime import timedelta

# Set page config
st.set_page_config(layout="wide")

# Title and Inputs
st.markdown("# 📈 Nifty 50 Stock Market Analyzer")

nifty50_stocks = {
    "Adani Enterprises": "ADANIENT.NS", "Adani Ports": "ADANIPORTS.NS", "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS", "Axis Bank": "AXISBANK.NS", "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Bajaj Finserv": "BAJAJFINSV.NS", "Bharti Airtel": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS", "Britannia": "BRITANNIA.NS", "Cipla": "CIPLA.NS", "Coal India": "COALINDIA.NS",
    "Divi's Labs": "DIVISLAB.NS", "Dr. Reddy's Labs": "DRREDDY.NS", "Eicher Motors": "EICHERMOT.NS",
    "Grasim": "GRASIM.NS", "HCL Technologies": "HCLTECH.NS", "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS", "Hero MotoCorp": "HEROMOTOCO.NS", "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "ICICI Bank": "ICICIBANK.NS", "ITC": "ITC.NS",
    "IndusInd Bank": "INDUSINDBK.NS", "Infosys": "INFY.NS", "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS", "LTIMindtree": "LTIM.NS", "L&T": "LT.NS",
    "M&M": "M&M.NS", "Maruti Suzuki": "MARUTI.NS", "NTPC": "NTPC.NS", "Nestle India": "NESTLEIND.NS",
    "Oil & Natural Gas Corp": "ONGC.NS", "Power Grid": "POWERGRID.NS", "Reliance Industries": "RELIANCE.NS",
    "SBI": "SBIN.NS", "SBI Life": "SBILIFE.NS", "Sun Pharma": "SUNPHARMA.NS", "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS", "Tata Consumer": "TATACONSUM.NS", "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS", "Titan": "TITAN.NS", "UPL": "UPL.NS", "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS"
}

col1, col2 = st.columns(2)
with col1:
    selected_stock = st.selectbox("Select a NIFTY 50 Stock:", sorted(nifty50_stocks.keys()))
with col2:
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

ticker = nifty50_stocks[selected_stock]

@st.cache_data
def fetch_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end)
    except:
        return pd.DataFrame()

with st.spinner("📡 Fetching data..."):
    data = fetch_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("⚠️ No historical data found for this ticker.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.write("📊 Sample Data", data.tail())

# ---------------------- Main Candlestick Chart ---------------------- #
st.subheader(f"📈 {ticker} - Candlestick Chart")
candlestick_data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
if candlestick_data.empty:
    st.warning("⚠️ No valid candlestick data to display.")
else:
    fig = go.Figure(data=[
        go.Candlestick(x=candlestick_data.index,
                       open=candlestick_data['Open'],
                       high=candlestick_data['High'],
                       low=candlestick_data['Low'],
                       close=candlestick_data['Close'])
    ])
    fig.update_layout(xaxis_rangeslider_visible=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- 📊 Fundamental Analysis ---------------------- #
st.header("📊 Fundamental Analysis")
ticker_info = yf.Ticker(ticker).info

try:
    roe = round(100 * (ticker_info['netIncome'] / ticker_info['totalStockholderEquity']), 2)
    de_ratio = round(ticker_info['totalDebt'] / ticker_info['totalStockholderEquity'], 2)
    eps = ticker_info.get('trailingEps', None)
    pe_ratio = round(ticker_info['currentPrice'] / eps, 2) if eps else None
except:
    roe = de_ratio = eps = pe_ratio = None

st.markdown(f"**Return on Equity (ROE):** {roe}%")
st.markdown(f"**Debt-to-Equity Ratio (D/E):** {de_ratio}")
st.markdown(f"**Earnings Per Share (EPS):** {eps}")
st.markdown(f"**Price-to-Earnings Ratio (P/E):** {pe_ratio}")

# Risk Comment with None check
if de_ratio is not None:
    if de_ratio < 1:
        risk_comment = "Low Risk"
    elif de_ratio < 2:
        risk_comment = "Medium Risk"
    else:
        risk_comment = "High Risk"
    st.info(f"💡 Risk Comment Based on D/E Ratio: **{risk_comment}**")
else:
    st.warning("⚠️ D/E Ratio not available. Risk analysis cannot be performed.")

# ---------------------- 📉 Technical Analysis ---------------------- #
st.header("📉 Technical Analysis")
# Moving Averages
data['SMA20'] = data['Close'].rolling(window=20).mean()
data['SMA50'] = data['Close'].rolling(window=50).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price", line=dict(color="white")))
fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name="SMA 20", line=dict(color="blue")))
fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name="SMA 50", line=dict(color="orange")))
fig_ma.update_layout(title="Moving Averages (SMA 20 & 50)", height=400)
st.plotly_chart(fig_ma, use_container_width=True)

# RSI
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=70, y1=70,
                  line=dict(color="red", dash="dash"))
fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=30, y1=30,
                  line=dict(color="green", dash="dash"))
fig_rsi.update_layout(title="RSI (Relative Strength Index)", height=300)
st.plotly_chart(fig_rsi, use_container_width=True)

st.info("📌 RSI above 70 = overbought, below 30 = oversold.")

# ---------------------- 🧠 Pattern & Trend Detection ---------------------- #
st.header("🧠 Pattern & Trend Detection")
recent_data = data[-30:].copy()
recent_data['SMA20'] = recent_data['Close'].rolling(window=20).mean()
recent_data['SMA50'] = recent_data['Close'].rolling(window=50).mean()

if recent_data['SMA20'].iloc[-1] > recent_data['SMA50'].iloc[-1]:
    st.success("📌 Trend Based on SMA Crossover: Uptrend")
else:
    st.error("📌 Trend Based on SMA Crossover: Downtrend")

# Pattern detection example: Double Top
pattern_fig = go.Figure()
pattern_fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], name="Price"))

def detect_double_top(prices, window=5):
    local_max = (prices == prices.rolling(window, center=True).max())
    return prices[local_max].dropna()

double_tops = detect_double_top(recent_data['Close'])
if not double_tops.empty:
    st.error("🔺 Possible Double Top Pattern Detected — Bearish Reversal")
    pattern_fig.add_trace(go.Scatter(x=double_tops.index, y=double_tops.values,
                                     name="Double Top", mode='markers', marker=dict(color='red', size=10)))

pattern_fig.update_layout(title="Detected Patterns (Last 30 Days)", height=400)
st.plotly_chart(pattern_fig, use_container_width=True)
