# Complete Streamlit App for Nifty 50 Stock Analysis with Pattern Detection
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

# App Title
st.set_page_config(layout="wide")
st.title("📊 Nifty 50 Stock Market Analyzer")

# Sidebar for input
st.sidebar.header("Select a NIFTY 50 Stock:")
selected_stock = st.sidebar.selectbox("", sorted(nifty50_stocks.keys()))
ticker = nifty50_stocks[selected_stock]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch Data
@st.cache_data
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        return df.dropna()
    except:
        return None

with st.spinner("📡 Fetching data..."):
    data = fetch_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("⚠️ No historical data found for this ticker.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.subheader("📊 Sample Data")
st.dataframe(data.tail())

# Main Candlestick Chart
st.subheader(f"📉 {selected_stock} Candlestick Chart")
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

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Fundamental Analysis", "📉 Technical Analysis", "📈 Pattern Recognition"])

# ---------------------- Fundamental Analysis ---------------------- #
with tab1:
    st.header("📊 Fundamental Analysis")
    try:
        info = yf.Ticker(ticker).info
        roe = (info['netIncome'] / info['totalStockholderEquity']) * 100
        de_ratio = info['totalDebt'] / info['totalStockholderEquity']
        eps = info['trailingEps']
        pe_ratio = info['currentPrice'] / eps

        st.write(f"**Return on Equity (ROE):** {roe:.2f}%")
        st.write(f"**Debt-to-Equity Ratio (D/E):** {de_ratio:.2f}")
        st.write(f"**Earnings Per Share (EPS):** {eps:.2f}")
        st.write(f"**Price-to-Earnings Ratio (P/E):** {pe_ratio:.2f}")

        risk_comment = "Low Risk" if de_ratio < 1 else "Medium Risk" if de_ratio < 2 else "High Risk"
        st.success(f"📌 Risk Assessment: {risk_comment}")
    except:
        st.warning("⚠️ Not enough data for fundamental metrics.")
    st.text_area("📝 Your Comment on Fundamental Analysis:", placeholder="Write your analysis...")

# ---------------------- Technical Analysis ---------------------- #
with tab2:
    st.header("📉 Technical Analysis")
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', line=dict(color='blue')))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='orange')))
    fig_ma.update_layout(title='Moving Averages', height=400)
    st.plotly_chart(fig_ma, use_container_width=True)

    st.subheader("📉 Relative Strength Index (RSI)")
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
    fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=70, y1=70, line=dict(color="red", dash="dash"))
    fig_rsi.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=30, y1=30, line=dict(color="green", dash="dash"))
    fig_rsi.update_layout(height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.success("📌 RSI above 70 = Overbought, below 30 = Oversold")

# ---------------------- Pattern Recognition ---------------------- #
with tab3:
    st.header("📈 Pattern Recognition")
    st.subheader("🔁 Reversal Patterns")

    def detect_double_top(prices, window=5):
        max_idx = argrelextrema(prices.values, np.greater, order=window)[0]
        return prices.iloc[max_idx]

    recent = data[-30:]
    double_tops = detect_double_top(recent['Close'])
    fig_pattern = go.Figure()
    fig_pattern.add_trace(go.Scatter(x=recent.index, y=recent['Close'], name="Close Price"))
    fig_pattern.add_trace(go.Scatter(x=double_tops.index, y=double_tops.values, name="Double Top", mode='markers', marker=dict(color='red')))
    fig_pattern.update_layout(title="Double Top Detection (30 Days)", height=400)
    st.plotly_chart(fig_pattern, use_container_width=True)
    if not double_tops.empty:
        st.warning("⚠️ Double Top pattern may indicate a reversal trend.")

    st.subheader("🔄 Continuation Patterns")
    st.info("📌 More pattern detection like Head & Shoulders, Triangles, Flags, Wedges coming next.")
