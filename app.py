# Complete Streamlit App for Nifty 50 Stock Analysis (Final Version with Rate Limit Handling)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from io import BytesIO
import time

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

# Streamlit UI
st.title("📈 Nifty 50 Stock Market Analyzer")
st.sidebar.header("Choose a Stock and Date Range")
selected_stock = st.sidebar.selectbox("Select a NIFTY 50 Stock:", sorted(nifty50_stocks.keys()))
ticker = nifty50_stocks[selected_stock]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)
time.sleep(1.5)  # avoid rate limiting
try:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
except yf.YFRateLimitError:
    st.warning("⚠️ Rate limit reached. Using fallback data.")
    try:
        info = ticker_obj.fast_info
        info['sector'] = 'N/A'
        info['marketCap'] = info.get('market_cap', 0)
        info['fiftyTwoWeekHigh'] = info.get('year_high', 'N/A')
        info['fiftyTwoWeekLow'] = info.get('year_low', 'N/A')
    except Exception as e:
        st.error(f"Failed to fetch stock info: {e}")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error fetching stock info: {e}")
    st.stop()

if data.empty:
    st.error("No data found for this stock in selected range.")
    st.stop()

# Candlestick chart
st.markdown("### 📊 Candlestick Chart")
candle = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'])])
candle.update_layout(title=f"Candlestick Chart for {selected_stock}", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(candle, use_container_width=True)

# Volume chart with Buy/Sell
st.markdown("### Volume Analysis")
data['Volume_Type'] = np.where(data['Close'] > data['Open'], 'Buy', 'Sell')
buy_volume = data[data['Volume_Type'] == 'Buy']['Volume']
sell_volume = data[data['Volume_Type'] == 'Sell']['Volume']
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=buy_volume.index, y=buy_volume, name='Buy Volume', marker_color='green'))
fig_vol.add_trace(go.Bar(x=sell_volume.index, y=sell_volume, name='Sell Volume', marker_color='red'))
fig_vol.update_layout(title='Buy vs Sell Volume', barmode='stack', xaxis_title='Date', yaxis_title='Volume')
st.plotly_chart(fig_vol, use_container_width=True)

# Stock summary
st.markdown("### ℹ️ Stock Summary")
st.write(f"**Sector:** {info.get('sector', 'N/A')}")
st.write(f"**Market Cap:** ₹{info.get('marketCap', 0) / 1e7:.2f} Cr")
st.write(f"**52 Week High:** ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
st.write(f"**52 Week Low:** ₹{info.get('fiftyTwoWeekLow', 'N/A')}")

# Trend pattern analysis
st.markdown("### 🔍 Trend Pattern Analysis")
close_prices = data['Close']
if close_prices.isnull().any() or close_prices.empty:
    st.warning("Insufficient data for pattern analysis.")
else:
    trend_comment = ""
    annotations = []
    pattern_messages = []

    if close_prices.iloc[-1] > close_prices.mean():
        trend_comment = "The stock is currently trading **above average**, indicating a potential bullish trend."
        annotations.append(dict(x=close_prices.index[-1], y=close_prices.iloc[-1], text='Above Avg → Bullish',
                                showarrow=True, arrowhead=1, ax=-40, ay=-40))
    elif close_prices.iloc[-1] < close_prices.mean():
        trend_comment = "The stock is trading **below average**, possibly indicating a bearish outlook."
        annotations.append(dict(x=close_prices.index[-1], y=close_prices.iloc[-1], text='Below Avg → Bearish',
                                showarrow=True, arrowhead=1, ax=-40, ay=40))
    else:
        trend_comment = "The stock is trading around its average. No strong trend observed."

    recent = close_prices[-50:]
    max_idx = argrelextrema(recent.values, np.greater, order=3)[0]
    min_idx = argrelextrema(recent.values, np.less, order=3)[0]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=close_prices.index, y=close_prices, mode='lines', name='Close Price'))
    fig_trend.add_trace(go.Scatter(x=close_prices.index, y=[close_prices.mean()]*len(close_prices),
                                   mode='lines', name='Average Price', line=dict(dash='dash')))

    # Head & Shoulders
    if len(max_idx) >= 3:
        peaks = recent.values[max_idx]
        if peaks[0] < peaks[1] > peaks[2] and abs(peaks[0] - peaks[2])/peaks[1] < 0.15:
            confidence = round(100 - abs(peaks[0] - peaks[2])/peaks[1] * 100, 1)
            pattern_messages.append(f"🔺 Head & Shoulders detected. Confidence: {confidence}%")
            for i in max_idx[:3]:
                fig_trend.add_trace(go.Scatter(x=[recent.index[i]], y=[recent.values[i]], mode='markers+text',
                                              name='H&S', marker=dict(size=10, color='red'),
                                              text=['H', 'S1', 'S2'][i % 3], textposition='top center'))

    # Cup & Handle
    if len(min_idx) > 0 and recent.idxmin() < recent.index[-5]:
        trough = recent.min()
        right_side = recent[recent.index > recent.idxmin()]
        if right_side.mean() > trough * 1.05:
            confidence = round((right_side.mean() / trough - 1) * 100, 1)
            pattern_messages.append(f"☕ Cup & Handle detected. Confidence: {confidence}%")
            fig_trend.add_trace(go.Scatter(x=[recent.idxmin()], y=[trough], mode='markers+text',
                                          name='Cup', marker=dict(size=10, color='blue'),
                                          text=['Cup'], textposition='bottom center'))

    # Flag
    if len(recent) > 20:
        spike = recent[-20:-10].pct_change().sum()
        if spike > 0.1:
            X = np.arange(10).reshape(-1, 1)
            y = recent[-10:].values.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0][0]
            if abs(slope) < 0.01:
                confidence = round((spike / 0.1) * 100, 1)
                pattern_messages.append(f"🚩 Bull Flag detected. Confidence: {confidence}%")
                fig_trend.add_trace(go.Scatter(x=recent[-10:].index, y=recent[-10:], mode='lines',
                                              name='Flag', line=dict(color='orange', dash='dot')))

    fig_trend.update_layout(title="Trend Visualization", annotations=annotations)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.info(trend_comment)
    for msg in pattern_messages:
        st.warning(msg)
