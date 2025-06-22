# Complete Streamlit App for Nifty 50 Stock Market Analyzer (Styled)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Nifty 50 Stock Analyzer", page_icon="📊")

# Apply Custom Theme and Styling
st.markdown("""
    <style>
    .main {background-color: #0f1117; color: white;}
    .css-18e3th9 {background-color: #0f1117;}
    .stApp {background: linear-gradient(to bottom right, #1f222c, #111118); color: white; font-family: 'Segoe UI', sans-serif;}
    .card {padding: 1rem; border-radius: 12px; background-color: #1e2130; box-shadow: 0 0 10px rgba(0,0,0,0.5); margin-bottom: 1.5rem;}
    h1, h2, h3, h4 {color: #f5f5f5;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center;'>
        <img src='https://media.giphy.com/media/3o7btQcyH6G7U9QfDi/giphy.gif' width='200'/>
        <h1>📈 Nifty 50 Stock Market Analyzer</h1>
        <p>Analyze historical performance, patterns, and trends of NIFTY 50 stocks</p>
    </div>
""", unsafe_allow_html=True)

# Company list
ticker_map = {
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
    stock_name = st.selectbox("Select a NIFTY 50 Stock:", sorted(ticker_map.keys()))
with col2:
    start = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end = st.date_input("End Date", pd.to_datetime("today"))

symbol = ticker_map[stock_name]

def get_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end)
    except:
        return pd.DataFrame()

st.info("📡 Fetching stock data...")
data = get_data(symbol, start, end)
if data.empty:
    st.error("No data found.")
    st.stop()

st.markdown("<div class='card'><h3>📊 Sample Data</h3>", unsafe_allow_html=True)
st.dataframe(data.tail(), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----- 📈 Candlestick Chart ----- #
st.markdown("<div class='card'><h3>📈 Candlestick Chart</h3>", unsafe_allow_html=True)
fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----- 📊 Fundamental Analysis ----- #
st.markdown("<div class='card'><h3>📊 Fundamental Analysis</h3>", unsafe_allow_html=True)
info = yf.Ticker(symbol).info
try:
    roe = round(100 * (info['netIncome'] / info['totalStockholderEquity']), 2)
    de = round(info['totalDebt'] / info['totalStockholderEquity'], 2)
    eps = info.get('trailingEps', None)
    pe = round(info['currentPrice'] / eps, 2) if eps else None
except:
    roe = de = eps = pe = None
st.write(f"**Return on Equity (ROE):** {roe}%")
st.write(f"**Debt-to-Equity Ratio (D/E):** {de}")
st.write(f"**Earnings Per Share (EPS):** {eps}")
st.write(f"**Price-to-Earnings Ratio (P/E):** {pe}")
if de is not None:
    risk = "Low Risk" if de < 1 else "Medium Risk" if de < 2 else "High Risk"
    st.success(f"📌 Risk Assessment: {risk}")
else:
    st.warning("D/E Ratio not available.")
st.markdown("</div>", unsafe_allow_html=True)

# ----- 📉 Technical Analysis ----- #
st.markdown("<div class='card'><h3>📉 Technical Analysis</h3>", unsafe_allow_html=True)
data['SMA20'] = data['Close'].rolling(20).mean()
data['SMA50'] = data['Close'].rolling(50).mean()
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color="white")))
fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name="SMA20", line=dict(color="blue")))
fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name="SMA50", line=dict(color="orange")))
fig_ma.update_layout(title="Simple Moving Averages", height=400, template="plotly_dark")
st.plotly_chart(fig_ma, use_container_width=True)

data['RSI'] = ta.momentum.RSIIndicator(data['Close'], 14).rsi()
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=70, y1=70, line=dict(color="red", dash="dash"))
fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=30, y1=30, line=dict(color="green", dash="dash"))
fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=300, template="plotly_dark")
st.plotly_chart(fig_rsi, use_container_width=True)

st.info("RSI above 70 is overbought, below 30 is oversold.")
st.markdown("</div>", unsafe_allow_html=True)

# ----- 🧠 Pattern & Trend Detection ----- #
st.markdown("<div class='card'><h3>🧠 Pattern & Trend Detection</h3>", unsafe_allow_html=True)
recent = data[-30:]
recent['SMA20'] = recent['Close'].rolling(20).mean()
recent['SMA50'] = recent['Close'].rolling(50).mean()
trend = "Uptrend" if recent['SMA20'].iloc[-1] > recent['SMA50'].iloc[-1] else "Downtrend"
st.success(f"Detected Trend (Last 30 days): {trend}")

def detect_double_top(prices, w=5):
    maxima = (prices == prices.rolling(w, center=True).max())
    return prices[maxima].dropna()
tops = detect_double_top(recent['Close'])
fig_pat = go.Figure()
fig_pat.add_trace(go.Scatter(x=recent.index, y=recent['Close'], name="Close"))
if not tops.empty:
    fig_pat.add_trace(go.Scatter(x=tops.index, y=tops.values, name="Double Top", mode='markers', marker=dict(color='red')))
    st.error("Double Top Pattern Detected")
fig_pat.update_layout(title="Pattern Detection", height=400, template="plotly_dark")
st.plotly_chart(fig_pat, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
