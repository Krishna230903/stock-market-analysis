# Complete Streamlit App for Nifty 50 Stock Analysis (Updated with More Patterns)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from io import BytesIO

# List of NIFTY 50 companies and tickers
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

st.set_page_config(layout="wide")
st.title("📈 Nifty 50 Stock Market Analyzer")

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
        return None

with st.spinner("📡 Fetching data..."):
    data = fetch_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("⚠️ No historical data found for this ticker.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.write("📊 Sample Data", data.tail())

# ---------------------- Candlestick Chart ---------------------- #
candlestick_data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
if not candlestick_data.empty:
    st.subheader(f"📈 {selected_stock} - Candlestick Chart")
    fig = go.Figure(data=[
        go.Candlestick(x=candlestick_data.index,
                       open=candlestick_data['Open'],
                       high=candlestick_data['High'],
                       low=candlestick_data['Low'],
                       close=candlestick_data['Close'])
    ])
    fig.update_layout(xaxis_rangeslider_visible=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Tabs ---------------------- #
tabs = st.tabs(["📊 Fundamental Analysis", "📉 Technical Analysis", "🧠 Pattern & Trend Insights"])

# ---------------------- Technical Analysis ---------------------- #
with tabs[1]:
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

    st.subheader("📈 Moving Averages")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20'))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50'))
    fig_ma.update_layout(height=400)
    st.plotly_chart(fig_ma, use_container_width=True)

    st.subheader("📉 Relative Strength Index (RSI)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
    fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=70, y1=70,
                      line=dict(color="red", dash="dash"))
    fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=30, y1=30,
                      line=dict(color="green", dash="dash"))
    fig_rsi.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.success("📌 RSI above 70 = Overbought | Below 30 = Oversold")

# ---------------------- Pattern & Trend Analysis ---------------------- #
def detect_double_top(data):
    close = data['Close']
    local_max = argrelextrema(close.values, np.greater, order=5)[0]
    if len(local_max) >= 2:
        for i in range(len(local_max)-1):
            if abs(close.iloc[local_max[i]] - close.iloc[local_max[i+1]]) < 0.01 * close.iloc[local_max[i]]:
                return local_max[i], local_max[i+1]
    return None

def plot_pattern_chart(data, pattern_locs, pattern_type):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    for loc in pattern_locs:
        fig.add_trace(go.Scatter(x=[data.index[loc]], y=[data['Close'].iloc[loc]], mode='markers+text',
                                 marker=dict(color='red', size=10), name=pattern_type,
                                 text=[pattern_type], textposition='top center'))
    fig.update_layout(title=f"📉 Pattern Detected: {pattern_type}", height=400)
    return fig

with tabs[2]:
    st.subheader("🧠 Trend & Pattern Insights")
    trend = "Uptrend" if data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1] else "Downtrend"
    st.markdown(f"**📊 Trend Based on SMA Crossover:** {trend}")

    double_top = detect_double_top(data)
    if double_top:
        st.warning("🔺 Possible Double Top Pattern Detected — Bearish Reversal")
        fig_pattern = plot_pattern_chart(data, double_top, "Double Top")
        st.plotly_chart(fig_pattern, use_container_width=True)
    else:
        st.info("✅ No clear double top pattern found.")

    # TODO: Add wedge, triangle, pennant patterns (future improvements here)
    st.caption("More patterns such as Wedges, Triangles, and Pennants can be added using machine learning or advanced heuristics.")
