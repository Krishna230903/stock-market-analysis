# Complete Streamlit App for Nifty 50 Stock Analysis (Updated)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from io import BytesIO

# List of NIFTY 50 companies and tickers
nifty50_stocks = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Divi's Labs": "DIVISLAB.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCL Technologies": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "Infosys": "INFY.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "L&T": "LT.NS",
    "LTIMindtree": "LTIM.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI": "SBIN.NS",
    "SBI Life": "SBILIFE.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan Company": "TITAN.NS",
    "UPL": "UPL.NS",
    "Ultratech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS"
}

st.set_page_config(layout="wide")
st.title("📊 NIFTY 50 Stock Analysis Dashboard")

selected_stock = st.selectbox("Choose a Stock:", sorted(nifty50_stocks.keys()))
ticker = nifty50_stocks[selected_stock]

# Date range input
start_date = st.date_input("Select start date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("Select end date", pd.to_datetime("today"))

# Fetch data safely
with st.spinner("📡 Fetching data..."):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        st.stop()

if data.empty:
    st.error("⚠️ No historical data found for this ticker.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.write("📊 Sample Data", data.tail())

stock = yf.Ticker(ticker)
info = stock.info

candlestick_data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
if candlestick_data.empty:
    st.warning("⚠️ No valid candlestick data to display.")
else:
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

# Tabs
fundamental_tab, technical_tab = st.tabs(["📊 Fundamental Analysis", "📈 Technical Analysis"])

with fundamental_tab:
    st.markdown("### 🧮 Key Fundamental Ratios")
    roe = info.get('returnOnEquity', None)
    de = info.get('debtToEquity', None)
    eps = info.get('trailingEps', None)
    pe = info.get('trailingPE', None)

    roe_display = f"{roe*100:.2f}%" if roe else "N/A"
    de_display = f"{de:.2f}" if de else "N/A"
    pe_display = f"{pe:.2f}" if pe else "N/A"

    st.metric("Return on Equity (ROE)", roe_display)
    st.metric("Debt-to-Equity Ratio (D/E)", de_display)
    st.metric("Earnings Per Share (EPS)", f"₹{eps:.2f}" if eps else "N/A")
    st.metric("Price-to-Earnings Ratio (P/E)", pe_display)

    st.markdown("### ⚠️ Risk Assessment")
    if roe and roe < 0.10 or (de and de > 2.0):
        risk = "High"
    elif roe and roe > 0.20 and (de and de < 0.5):
        risk = "Low"
    else:
        risk = "Medium"
    st.metric("Risk Factor", risk)

    st.markdown("### 💡 Investment Suggestion")
    if risk == "Low" and pe and pe < 25:
        suggestion = "Buy Call"
    elif risk == "High" or (pe and pe > 40):
        suggestion = "Sell Call"
    else:
        suggestion = "Hold"
    st.metric("Overall Recommendation", suggestion)

    st.download_button("⬇️ Export Data to CSV", data.to_csv().encode(), file_name=f"{ticker}_data.csv", mime="text/csv")

with technical_tab:
    st.markdown("### Moving Averages")
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    st.line_chart(data[['Close', 'SMA50', 'SMA200']])

    st.markdown("### Relative Strength Index (RSI)")
    rsi = ta.momentum.RSIIndicator(data['Close'], window=14)
    data['RSI'] = rsi.rsi()
    st.line_chart(data[['RSI']])
    rsi_value = data['RSI'].iloc[-1]
    trend = "Overbought" if rsi_value > 70 else ("Oversold" if rsi_value < 30 else "Neutral")
    st.metric("Current RSI Trend", trend)

    if rsi_value > 70:
        st.warning("⚠️ RSI Alert: Stock is Overbought!")
    elif rsi_value < 30:
        st.success("✅ RSI Alert: Stock is Oversold!")

    st.markdown("### Support and Resistance Levels")
    data['Support'] = data['Low'].rolling(window=20).min()
    data['Resistance'] = data['High'].rolling(window=20).max()
    st.line_chart(data[['Close', 'Support', 'Resistance']])

    st.markdown("### Volume Analysis")
    data['Volume_Type'] = np.where(data['Close'] > data['Open'], 'Buy', 'Sell')
    buy_volume = data[data['Volume_Type'] == 'Buy']['Volume']
    sell_volume = data[data['Volume_Type'] == 'Sell']['Volume']

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=buy_volume.index, y=buy_volume, name='Buy Volume', marker_color='green'))
    fig_vol.add_trace(go.Bar(x=sell_volume.index, y=sell_volume, name='Sell Volume', marker_color='red'))

    fig_vol.update_layout(title='Buy vs Sell Volume', barmode='stack', xaxis_title='Date', yaxis_title='Volume')
    st.plotly_chart(fig_vol, use_container_width=True)

