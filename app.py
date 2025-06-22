# Final Clean Working Version - Stock Analysis App with NIFTY 50
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from scipy.signal import argrelextrema

# Streamlit page config
st.set_page_config(page_title="Nifty 50 Stock Analyzer", layout="wide")

# ---- NIFTY 50 List ----
nifty50 = {
    "Adani Enterprises": "ADANIENT.NS", "Asian Paints": "ASIANPAINT.NS", "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Bharti Airtel": "BHARTIARTL.NS", "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS", "Divi's Labs": "DIVISLAB.NS", "Dr. Reddy's": "DRREDDY.NS",
    "HCL Tech": "HCLTECH.NS", "HDFC Bank": "HDFCBANK.NS", "HUL": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "ITC": "ITC.NS",
    "Kotak Bank": "KOTAKBANK.NS", "L&T": "LT.NS", "Maruti Suzuki": "MARUTI.NS",
    "Nestle India": "NESTLEIND.NS", "NTPC": "NTPC.NS", "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS", "Reliance": "RELIANCE.NS", "SBI": "SBIN.NS",
    "Sun Pharma": "SUNPHARMA.NS", "TCS": "TCS.NS", "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS", "Tech Mahindra": "TECHM.NS", "Wipro": "WIPRO.NS"
}

# ---- MAIN INPUTS ----
st.title("📊 Nifty 50 Stock Market Analyzer")
col1, col2 = st.columns(2)
with col1:
    selected_stock = st.selectbox("📌 Select a Nifty 50 Stock", sorted(nifty50.keys()))
with col2:
    start = st.date_input("📅 Start Date", pd.to_datetime("2022-01-01"))
    end = st.date_input("📅 End Date", pd.to_datetime("today"))

ticker = nifty50[selected_stock]
data = yf.download(ticker, start=start, end=end)
if data.empty:
    st.error("No data found. Try a different date range.")
    st.stop()

data.dropna(inplace=True)
info = yf.Ticker(ticker).info

# ---- CANDLESTICK CHART ----
st.subheader(f"📈 {selected_stock} Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])
fig.update_layout(xaxis_rangeslider_visible=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# ---- FUNDAMENTAL ANALYSIS ----
st.header("📘 Fundamental Analysis")
try:
    roe = (info['netIncome'] / info['totalStockholderEquity']) * 100
    de = info['totalDebt'] / info['totalStockholderEquity']
    eps = info['trailingEps']
    pe = info['currentPrice'] / eps

    st.write(f"**Return on Equity (ROE):** {roe:.2f}%")
    st.write(f"**Debt-to-Equity (D/E):** {de:.2f}")
    st.write(f"**Earnings Per Share (EPS):** ₹{eps:.2f}")
    st.write(f"**Price-to-Earnings (P/E):** {pe:.2f}")

    risk = "Low Risk" if de < 1 else "Medium Risk" if de < 2 else "High Risk"
    st.success(f"📌 Risk Level: {risk}")
except Exception as e:
    st.warning(f"Missing info: {e}")

# ---- TECHNICAL ANALYSIS ----
st.header("📕 Technical Analysis")

# Moving Averages
data['SMA20'] = data['Close'].rolling(20).mean()
data['SMA50'] = data['Close'].rolling(50).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA20', line=dict(color='blue')))
fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA50', line=dict(color='orange')))
fig_ma.update_layout(title="Moving Averages", height=400)
st.plotly_chart(fig_ma, use_container_width=True)

# RSI
st.subheader("📉 RSI (Relative Strength Index)")
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='magenta')))
fig_rsi.add_shape(type='line', x0=data.index.min(), x1=data.index.max(), y0=70, y1=70, line=dict(color='red', dash='dash'))
fig_rsi.add_shape(type='line', x0=data.index.min(), x1=data.index.max(), y0=30, y1=30, line=dict(color='green', dash='dash'))
fig_rsi.update_layout(height=300)
st.plotly_chart(fig_rsi, use_container_width=True)

rsi_comment = "Overbought (>70)" if data['RSI'].iloc[-1] > 70 else "Oversold (<30)" if data['RSI'].iloc[-1] < 30 else "Neutral"
st.success(f"📌 RSI Status: {rsi_comment}")

# ---- PATTERN DETECTION (Double Top Example) ----
st.header("📙 Pattern Detection (Last 30 Days)")
recent = data['Close'].tail(30)
peaks = argrelextrema(recent.values, np.greater, order=3)[0]
fig_pattern = go.Figure()
fig_pattern.add_trace(go.Scatter(x=recent.index, y=recent, name='Close Price'))
fig_pattern.add_trace(go.Scatter(x=recent.index[peaks], y=recent.iloc[peaks], mode='markers', marker=dict(color='red', size=10), name='Peaks'))
fig_pattern.update_layout(title="Double Top Detection (30 Days)", height=400)
st.plotly_chart(fig_pattern, use_container_width=True)

if len(peaks) >= 2:
    st.warning("⚠️ Potential Double Top pattern detected - Possible trend reversal.")

