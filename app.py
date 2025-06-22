# 📈 Nifty 50 Stock Market Analyzer with Tabs
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from scipy.signal import argrelextrema

# 📌 NIFTY 50 stock ticker map
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

# UI layout
st.title("📈 Nifty 50 Stock Market Analyzer")

col1, col2 = st.columns(2)
with col1:
    selected_stock = st.selectbox("Select a NIFTY 50 Stock:", sorted(nifty50_stocks.keys()))
with col2:
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

ticker = nifty50_stocks[selected_stock]

# Fetch historical stock data
data = yf.download(ticker, start=start_date, end=end_date)
info = yf.Ticker(ticker).info

if data.empty:
    st.error("No data found for this stock in the selected range.")
    st.stop()

data = data.dropna()

# Show recent sample
st.write("📊 Sample Data", data.tail())

# Candlestick chart
st.subheader(f"📈 {ticker} - Candlestick Chart")
candlestick_data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
fig = go.Figure(data=[
    go.Candlestick(
        x=candlestick_data.index,
        open=candlestick_data['Open'],
        high=candlestick_data['High'],
        low=candlestick_data['Low'],
        close=candlestick_data['Close']
    )
])
fig.update_layout(xaxis_rangeslider_visible=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# 📌 TABS: Fundamental | Technical | Pattern
tab1, tab2, tab3 = st.tabs(["📊 Fundamental Analysis", "📉 Technical Analysis", "📈 Pattern Recognition"])

# TAB 1: Fundamental
with tab1:
    st.subheader("📊 Fundamental Analysis")
    try:
        net_income = info.get("netIncome", None)
        equity = info.get("totalStockholderEquity", None)
        total_debt = info.get("totalDebt", None)
        eps = info.get("trailingEps", None)
        price = info.get("currentPrice", None)

        roe = (net_income / equity * 100) if net_income and equity else None
        de_ratio = (total_debt / equity) if total_debt and equity else None
        pe_ratio = (price / eps) if price and eps else None

        if roe is not None:
            st.write(f"**Return on Equity (ROE):** {roe:.2f}%")
        else:
            st.write("**Return on Equity (ROE):** Data not available")

        if de_ratio is not None:
            st.write(f"**Debt-to-Equity Ratio (D/E):** {de_ratio:.2f}")
        else:
            st.write("**Debt-to-Equity Ratio (D/E):** Data not available")

        if eps is not None:
            st.write(f"**Earnings Per Share (EPS):** {eps:.2f}")
        else:
            st.write("**Earnings Per Share (EPS):** Data not available")

        if pe_ratio is not None:
            st.write(f"**Price-to-Earnings Ratio (P/E):** {pe_ratio:.2f}")
        else:
            st.write("**Price-to-Earnings Ratio (P/E):** Data not available")

        # Risk comment
        if de_ratio is not None:
            if de_ratio < 1:
                risk = "Low Risk"
            elif de_ratio < 2:
                risk = "Medium Risk"
            else:
                risk = "High Risk"
            st.success(f"📌 Risk Assessment: {risk}")
        else:
            st.info("Risk assessment not possible due to missing D/E data.")
    except Exception as e:
        st.warning(f"⚠️ Error loading fundamentals: {e}")

# TAB 2: Technical
with tab2:
    st.subheader("📉 Technical Analysis")

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
    fig_rsi.update_layout(title="📉 Relative Strength Index (RSI)", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.success("📌 Comment: RSI above 70 = Overbought, below 30 = Oversold.")

# TAB 3: Pattern Recognition
with tab3:
    st.subheader("📈 Pattern Recognition (Basic Double Top)")

    def detect_double_top(prices, window=5):
        max_idx = argrelextrema(prices.values, np.greater, order=window)[0]
        return prices.iloc[max_idx]

    recent = data[-30:]
    double_tops = detect_double_top(recent['Close'])

    fig_pattern = go.Figure()
    fig_pattern.add_trace(go.Scatter(x=recent.index, y=recent['Close'], name="Close Price"))
    fig_pattern.add_trace(go.Scatter(x=double_tops.index, y=double_tops.values,
                                     name="Double Top", mode='markers', marker=dict(color='red')))
    fig_pattern.update_layout(title="Double Top Pattern Detection (30 Days)", height=400)
    st.plotly_chart(fig_pattern, use_container_width=True)

    if not double_tops.empty:
        st.warning("⚠️ Double Top pattern may indicate a reversal trend.")
