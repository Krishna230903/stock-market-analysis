# Complete Streamlit App for Nifty 50 Stock Analysis (Fixed RSI Issue + Tabs)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from io import BytesIO

# ============================ STOCK LIST ============================
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

# ============================ UI ============================
st.set_page_config(layout="wide")
st.title("📈 Nifty 50 Stock Market Analyzer")
st.sidebar.header("Choose a Stock and Date Range")
selected_stock = st.sidebar.selectbox("Select a NIFTY 50 Stock:", sorted(nifty50_stocks.keys()))
ticker = nifty50_stocks[selected_stock]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# ============================ DATA ============================
data = yf.download(ticker, start=start_date, end=end_date)
info = yf.Ticker(ticker).info
if data.empty:
    st.error("No data found for this stock in selected range.")
    st.stop()

# ============================ TABS ============================
tab1, tab2, tab3 = st.tabs(["📊 Fundamental Analysis", "📈 Technical Analysis", "🧠 Pattern & Trend Insights"])

# ============================ FUNDAMENTAL ANALYSIS ============================
with tab1:
    st.subheader("📊 Fundamental Analysis")
    st.markdown(f"**Market Cap:** {info.get('marketCap', 'NA')}")
    roe = info.get("returnOnEquity", 0)*100
    eps = info.get("trailingEps", 0)
    pe = info.get("trailingPE", 0)
    de = info.get("debtToEquity", 0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Return on Equity (ROE) %", f"{roe:.2f}", delta=f"{'↑' if roe > 15 else '↓'}")
        st.metric("EPS", f"{eps:.2f}")
    with col2:
        st.metric("P/E Ratio", f"{pe:.2f}", delta_color="inverse")
        st.metric("Debt-to-Equity (D/E)", f"{de:.2f}")

    risk = "High" if de > 1.2 else "Medium" if de > 0.5 else "Low"
    st.warning(f"🔎 **Risk Factor:** {risk}")

    df_export = pd.DataFrame({
        "ROE (%)": [roe],
        "EPS": [eps],
        "P/E": [pe],
        "D/E": [de],
        "Risk": [risk]
    })
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Fundamentals", data=csv, file_name=f"{ticker}_fundamentals.csv")

# ============================ TECHNICAL ANALYSIS ============================
with tab2:
    st.subheader("📈 Technical Analysis")
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    # Fix for RSI calculation
    close_prices = pd.Series(data['Close'].values, index=data.index).dropna()
    rsi_series = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi()
    data['RSI'] = rsi_series

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], name='SMA 200'))
    st.plotly_chart(fig, use_container_width=True)

    # RSI Alerts
    latest_rsi = data['RSI'].dropna().iloc[-1]
    if latest_rsi > 70:
        st.error(f"⚠️ RSI is {latest_rsi:.2f} — Stock is Overbought (Consider Sell)")
    elif latest_rsi < 30:
        st.success(f"✅ RSI is {latest_rsi:.2f} — Stock is Oversold (Consider Buy)")
    else:
        st.info(f"ℹ️ RSI is {latest_rsi:.2f} — Neutral")

# ============================ PATTERN & TREND ============================
with tab3:
    st.subheader("🧠 Trend & Pattern Insights")

    trend = "Uptrend" if data['SMA50'].iloc[-1] > data['SMA200'].iloc[-1] else "Downtrend"
    st.markdown(f"**📊 Trend Based on MA Crossover:** {trend}")

    # Double Top / Bottom detection (simple approach)
    prices = data['Close']
    local_max = prices[(prices.shift(1) < prices) & (prices.shift(-1) < prices)]
    local_min = prices[(prices.shift(1) > prices) & (prices.shift(-1) > prices)]
    if len(local_max) >= 2 and abs(local_max.iloc[-1] - local_max.iloc[-2]) / local_max.iloc[-1] < 0.03:
        st.error("🔺 Possible Double Top Pattern Detected — Bearish Reversal")
    elif len(local_min) >= 2 and abs(local_min.iloc[-1] - local_min.iloc[-2]) / local_min.iloc[-1] < 0.03:
        st.success("🔻 Possible Double Bottom Pattern Detected — Bullish Reversal")
    else:
        st.info("No strong reversal pattern detected.")
