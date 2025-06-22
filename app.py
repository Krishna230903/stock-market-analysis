# Complete Streamlit App for Nifty 50 Stock Analysis (Final Version with Pattern Recognition)
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
nifty50_stocks = { ... }  # [unchanged list of stocks]

# Streamlit UI
st.title("📈 Nifty 50 Stock Market Analyzer")
st.sidebar.header("Choose a Stock and Date Range")
selected_stock = st.sidebar.selectbox("Select a NIFTY 50 Stock:", sorted(nifty50_stocks.keys()))
ticker = nifty50_stocks[selected_stock]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)
info = yf.Ticker(ticker).info
if data.empty:
    st.error("No data found for this stock in selected range.")
    st.stop()

st.subheader(f"📊 Stock Summary: {selected_stock}")
st.write(f"**Sector:** {info.get('sector', 'N/A')}")
st.write(f"**Market Cap:** ₹{info.get('marketCap', 0) / 1e7:.2f} Cr")
st.write(f"**52 Week High:** ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
st.write(f"**52 Week Low:** ₹{info.get('fiftyTwoWeekLow', 'N/A')}")

# Trend Pattern Recognition
st.subheader("📉 Pattern Recognition Analysis")
data['Min'] = data['Close'][argrelextrema(data['Close'].values, np.less_equal, order=5)[0]]
data['Max'] = data['Close'][argrelextrema(data['Close'].values, np.greater_equal, order=5)[0]]

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['Min'], mode='markers', name='Local Min', marker=dict(color='blue', size=6)))
fig.add_trace(go.Scatter(x=data.index, y=data['Max'], mode='markers', name='Local Max', marker=dict(color='orange', size=6)))

pattern_conclusion = ""
if len(data['Max'].dropna()) >= 2 and len(data['Min'].dropna()) >= 2:
    last_max = data['Max'].dropna().iloc[-2:]
    last_min = data['Min'].dropna().iloc[-2:]
    if all(np.diff(last_max) < 0) and all(np.diff(last_min) < 0):
        pattern_conclusion = "📉 **Potential Downtrend**: Lower highs and lower lows observed."
    elif all(np.diff(last_max) > 0) and all(np.diff(last_min) > 0):
        pattern_conclusion = "📈 **Potential Uptrend**: Higher highs and higher lows observed."
    else:
        pattern_conclusion = "📊 **Sideways Trend**: No clear higher/lower high-low pattern."
else:
    pattern_conclusion = "⚠️ Not enough extrema points to determine trend."

fig.update_layout(title="Price Trend with Extrema", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
st.info(pattern_conclusion)

# [Continue with existing sections: Fundamentals, Technicals, Volume, RSI, Export etc.]

