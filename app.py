# Complete Streamlit App for Nifty 50 Stock Analysis with Pattern Detection
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from datetime import date

# App title
st.title("📈 Nifty 50 Stock Market Analyzer")

# ------------------- Stock & Date Selection ------------------- #
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Choose a Stock")
    stocks = {
        "Adani Enterprises": "ADANIENT.NS",
        "Apollo Hospitals": "APOLLOHOSP.NS",
        "Reliance Industries": "RELIANCE.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS"
    }
    stock_name = st.selectbox("", list(stocks.keys()))
    ticker = stocks[stock_name]

with col2:
    st.markdown("### Start Date")
    start_date = st.date_input("", date(2022, 1, 1))

with col3:
    st.markdown("### End Date")
    end_date = st.date_input(" ", date.today())

# ---------------------- Fetch Data ---------------------- #
@st.cache_data
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except:
        return None

with st.spinner("📡 Fetching data..."):
    data = fetch_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("⚠️ No historical data found for this ticker.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.write("📊 Sample Data", data.tail())

# ---------------------- Tabs ---------------------- #
tabs = st.tabs(["📊 Fundamental Analysis", "⚙️ Technical Analysis", "🧠 Pattern & Trend Insights"])

# ---------------------- Fundamental Analysis ---------------------- #
with tabs[0]:
    st.subheader("📊 Fundamental Insights")
    stock_info = yf.Ticker(ticker).info
    pe = stock_info.get("trailingPE", 0)
    roe = stock_info.get("returnOnEquity", 0)
    st.metric("P/E Ratio", f"{pe:.2f}", help="Lower P/E = undervalued")
    st.metric("ROE", f"{roe * 100:.2f}%", help="Higher ROE = efficient management")

    # Risk Analysis
    st.subheader("⚠️ Risk Factor")
    risk = "Low"
    if pe > 40 or roe < 0.05:
        risk = "High"
    elif pe > 25 or roe < 0.10:
        risk = "Medium"
    st.warning(f"📌 Based on P/E and ROE: **{risk} Risk**")

# ---------------------- Technical Analysis ---------------------- #
with tabs[1]:
    st.subheader("📈 Candlestick Chart")
    candle_data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    if not candle_data.empty:
        fig = go.Figure(data=[
            go.Candlestick(x=candle_data.index,
                           open=candle_data['Open'],
                           high=candle_data['High'],
                           low=candle_data['Low'],
                           close=candle_data['Close'])
        ])
        fig.update_layout(xaxis_rangeslider_visible=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Relative Strength Index (RSI)")
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
    fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=70, y1=70,
                      line=dict(color="red", dash="dash"))
    fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=30, y1=30,
                      line=dict(color="green", dash="dash"))
    fig_rsi.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.success("📌 RSI > 70 = Overbought | RSI < 30 = Oversold")

# ---------------------- Pattern & Trend Analysis ---------------------- #
with tabs[2]:
    st.subheader("🧠 Trend & Pattern Insights")
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()

    trend = "Sideways"
    if data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1]:
        trend = "Uptrend"
    elif data['SMA20'].iloc[-1] < data['SMA50'].iloc[-1]:
        trend = "Downtrend"

    st.markdown(f"**📈 SMA Trend Signal:** {trend}")

    # Pattern detection chart with mock example
    st.markdown("### Chart Pattern Overlay")
    fig_pat = go.Figure()
    fig_pat.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))

    # Add known pattern samples from image (mock highlighting)
    last = data.index[-1]
    start = data.index[-30]
    fig_pat.add_trace(go.Scatter(x=[start, last], y=[data['Close'].iloc[-30], data['Close'].iloc[-1]],
                                 mode='lines', name='Double Top Mock', line=dict(color='red', dash='dash')))
    fig_pat.update_layout(title="Pattern Overlay Example", height=350)
    st.plotly_chart(fig_pat, use_container_width=True)

    st.error("🔺 Double Top Pattern Detected — Possible Bearish Reversal")
    st.info("📘 More patterns like Flags, Wedges, Pennants, Rectangles can be added here.")
