# Nifty 50 Stock Analyzer - Streamlit App
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from scipy.signal import argrelextrema

# Set Streamlit layout
st.set_page_config(layout="wide")

# Stock list (NIFTY 50)
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
    "ONGC": "ONGC.NS", "Power Grid": "POWERGRID.NS", "Reliance Industries": "RELIANCE.NS",
    "SBI": "SBIN.NS", "SBI Life": "SBILIFE.NS", "Sun Pharma": "SUNPHARMA.NS", "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS", "Tata Consumer": "TATACONSUM.NS", "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS", "Titan": "TITAN.NS", "UPL": "UPL.NS", "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS"
}

# -------- UI --------
st.title("📈 Nifty 50 Stock Market Analyzer")
col1, col2 = st.columns(2)
with col1:
    selected_stock = st.selectbox("Select Stock", sorted(nifty50_stocks.keys()))
with col2:
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

ticker = nifty50_stocks[selected_stock]
data = yf.download(ticker, start=start_date, end=end_date)
info = yf.Ticker(ticker).info

if data.empty:
    st.error("No data found for selected stock and date range.")
    st.stop()

data.dropna(inplace=True)

# -------- Candlestick Chart (Main) --------
st.subheader(f"📈 Candlestick Chart for {selected_stock}")
fig = go.Figure(data=[go.Candlestick(
    x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']
)])
fig.update_layout(xaxis_rangeslider_visible=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# -------- Fundamental Analysis --------
st.header("📊 Fundamental Analysis")
try:
    roe = (info['netIncome'] / info['totalStockholderEquity']) * 100
    de_ratio = info['totalDebt'] / info['totalStockholderEquity']
    eps = info['trailingEps']
    pe = info['currentPrice'] / eps
    st.write(f"**Return on Equity (ROE):** {roe:.2f}%")
    st.write(f"**Debt-to-Equity Ratio (D/E):** {de_ratio:.2f}")
    st.write(f"**Earnings Per Share (EPS):** ₹{eps:.2f}")
    st.write(f"**Price-to-Earnings Ratio (P/E):** {pe:.2f}")

    risk = "Low Risk" if de_ratio < 1 else "Medium Risk" if de_ratio < 2 else "High Risk"
    st.success(f"📌 Risk Level: {risk}")
except:
    st.warning("Not enough data for fundamental metrics.")

# -------- Technical Analysis --------
st.header("📉 Technical Analysis")
data['SMA20'] = data['Close'].rolling(20).mean()
data['SMA50'] = data['Close'].rolling(50).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', line=dict(color='blue')))
fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='orange')))
fig_ma.update_layout(title="Moving Averages", height=400)
st.plotly_chart(fig_ma, use_container_width=True)

# RSI
st.subheader("📉 Relative Strength Index (RSI)")
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=70, y1=70, line=dict(color="red", dash="dash"))
fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=30, y1=30, line=dict(color="green", dash="dash"))
fig_rsi.update_layout(title="RSI", height=300)
st.plotly_chart(fig_rsi, use_container_width=True)
st.success("📌 Comment: RSI > 70 = Overbought | RSI < 30 = Oversold")

# -------- Pattern Detection (Double Top) --------
st.header("📈 Pattern Detection")
def detect_double_top(series, window=5):
    peaks = argrelextrema(series.values, np.greater, order=window)[0]
    return series.iloc[peaks]

last_30 = data['Close'].iloc[-30:]
tops = detect_double_top(last_30)

fig_pattern = go.Figure()
fig_pattern.add_trace(go.Scatter(x=last_30.index, y=last_30, name="Close Price"))
fig_pattern.add_trace(go.Scatter(x=tops.index, y=tops.values, name="Double Top", mode='markers', marker=dict(color='red')))
fig_pattern.update_layout(title="Double Top Pattern (Last 30 Days)", height=400)
st.plotly_chart(fig_pattern, use_container_width=True)

if not tops.empty:
    st.warning("⚠️ Possible Double Top pattern detected — may indicate bearish reversal.")
