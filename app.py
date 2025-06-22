# Complete Streamlit App for Nifty 50 Stock Analysis (Updated Layout & Pattern Chart)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from scipy.signal import argrelextrema
from io import BytesIO

# Helper functions
def fetch_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def identify_double_top(prices, order=5):
    maxima = argrelextrema(prices.values, np.greater_equal, order=order)[0]
    if len(maxima) >= 2:
        tops = prices.iloc[maxima[-2:]]
        if abs(tops.iloc[0] - tops.iloc[1]) / tops.mean() < 0.02:
            return True, tops.index
    return False, []

# Dictionary for Nifty 50 Stocks
nifty50_stocks = {
    "Adani Enterprises": "ADANIENT.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Cipla": "CIPLA.NS",
    "Dr. Reddy's Labs": "DRREDDY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "L&T": "LT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI": "SBIN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "TCS": "TCS.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Wipro": "WIPRO.NS"
}

# --------------------------- UI Section --------------------------- #
st.set_page_config(page_title="Nifty 50 Stock Market Analyzer", layout="wide")
st.title("📈 Nifty 50 Stock Market Analyzer")

# Main Area Select Inputs (not sidebar)
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    selected_stock = st.selectbox("Select a NIFTY 50 Stock:", list(nifty50_stocks.keys()))
with col2:
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
with col3:
    end_date = st.date_input("End Date", pd.to_datetime("today"))

ticker = nifty50_stocks[selected_stock]

# --------------------------- Data Fetch --------------------------- #
with st.spinner("📡 Fetching data..."):
    data = fetch_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("⚠️ No historical data found for this ticker.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.dropna(inplace=True)
st.subheader("📊 Sample Data")
st.write(data.tail())

# ---------------------- Candlestick Chart ---------------------- #
st.subheader(f"📈 {selected_stock} - Candlestick Chart")
fig = go.Figure(data=[
    go.Candlestick(x=data.index,
                   open=data['Open'],
                   high=data['High'],
                   low=data['Low'],
                   close=data['Close'])
])
fig.update_layout(xaxis_rangeslider_visible=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- RSI Chart ---------------------- #
data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=70, y1=70,
                  line=dict(color="red", dash="dash"))
fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=30, y1=30,
                  line=dict(color="green", dash="dash"))
fig_rsi.update_layout(title="📉 Relative Strength Index (RSI)", height=300)
st.plotly_chart(fig_rsi, use_container_width=True)

# ---------------------- Tabs for Analysis ---------------------- #
tab1, tab2, tab3 = st.tabs(["📊 Fundamental Analysis", "📉 Technical Analysis", "🧠 Pattern & Trend Insights"])

# ---------------------- Fundamental Analysis ---------------------- #
with tab1:
    st.subheader("📊 Fundamental Metrics")
    stock_info = yf.Ticker(ticker).info
    try:
        pe_ratio = stock_info.get('trailingPE', None)
        roe = stock_info.get('returnOnEquity', None)
        risk_level = "High" if pe_ratio and pe_ratio > 30 else "Medium" if pe_ratio and pe_ratio > 15 else "Low"
        st.metric("PE Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        st.metric("ROE", f"{roe:.2%}" if roe else "N/A")
        st.info(f"🛡️ Risk Assessment: {risk_level}")
    except Exception as e:
        st.error(f"Error fetching fundamentals: {e}")

# ---------------------- Technical Analysis ---------------------- #
with tab2:
    st.subheader("📉 Technical Analysis Insights")
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name="SMA 20", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name="SMA 50", line=dict(color='green')))
    fig.update_layout(title="SMA Crossover", height=300)
    st.plotly_chart(fig, use_container_width=True)

    sma_signal = "Uptrend" if data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1] else "Downtrend"
    st.success(f"📌 Trend: {sma_signal}")

# ---------------------- Pattern Recognition ---------------------- #
with tab3:
    st.subheader("🧠 Trend & Pattern Insights")
    close_series = data['Close']
    detected, top_dates = identify_double_top(close_series)
    trend = "Uptrend" if data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1] else "Downtrend"
    st.markdown(f"**📈 Trend Based on SMA Crossover:** {trend}")

    pattern_fig = go.Figure()
    pattern_fig.add_trace(go.Scatter(x=close_series.index, y=close_series, name="Close Price"))
    if detected:
        for date in top_dates:
            pattern_fig.add_trace(go.Scatter(x=[date], y=[close_series.loc[date]], mode='markers',
                                             marker=dict(color='red', size=12), name="Double Top"))
        st.error("🔺 Possible Double Top Pattern Detected — Bearish Reversal")
    else:
        st.success("✅ No significant double top pattern detected.")

    pattern_fig.update_layout(title="Close Price with Pattern Markers", height=300)
    st.plotly_chart(pattern_fig, use_container_width=True)

# ---------------------- Export Button ---------------------- #
buffer = BytesIO()
data.to_csv(buffer)
st.download_button("📥 Export Data to CSV", data=buffer.getvalue(), file_name=f"{ticker}_analysis.csv", mime="text/csv")
