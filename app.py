# Complete Streamlit App for Nifty 50 Stock Analysis
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta

# List of NIFTY 50 companies and tickers
nifty50_stocks = {
    'Adani Enterprises': 'ADANIENT.NS',
    'Adani Ports': 'ADANIPORTS.NS',
    'Apollo Hospitals': 'APOLLOHOSP.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Bajaj Finserv': 'BAJAJFINSV.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'BPCL': 'BPCL.NS',
    'Britannia Industries': 'BRITANNIA.NS',
    'Cipla': 'CIPLA.NS',
    'Coal India': 'COALINDIA.NS',
    "Divi's Labs": 'DIVISLAB.NS',
    'Dr. Reddy\'s': 'DRREDDY.NS',
    'Eicher Motors': 'EICHERMOT.NS',
    'Grasim Industries': 'GRASIM.NS',
    'HCL Technologies': 'HCLTECH.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'HDFC Life': 'HDFCLIFE.NS',
    'Hero MotoCorp': 'HEROMOTOCO.NS',
    'Hindalco': 'HINDALCO.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'ITC': 'ITC.NS',
    'Infosys': 'INFY.NS',
    'JSW Steel': 'JSWSTEEL.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'L&T': 'LT.NS',
    'LTIMindtree': 'LTIM.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Nestle India': 'NESTLEIND.NS',
    'NTPC': 'NTPC.NS',
    'ONGC': 'ONGC.NS',
    'Power Grid': 'POWERGRID.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'SBI': 'SBIN.NS',
    'SBI Life': 'SBILIFE.NS',
    'Sun Pharma': 'SUNPHARMA.NS',
    'Tata Consumer': 'TATACONSUM.NS',
    'Tata Motors': 'TATAMOTORS.NS',
    'Tata Steel': 'TATASTEEL.NS',
    'TCS': 'TCS.NS',
    'Tech Mahindra': 'TECHM.NS',
    'Titan Company': 'TITAN.NS',
    'UPL': 'UPL.NS',
    'Ultratech Cement': 'ULTRACEMCO.NS',
    'Wipro': 'WIPRO.NS'

}
st.set_page_config(layout="wide")
st.title("📊 NIFTY 50 Stock Analysis Dashboard")

# Dropdown for NIFTY 50 Stocks
selected_stock = st.selectbox("Choose a Stock:", sorted(nifty50_stocks.keys()))
ticker = nifty50_stocks[selected_stock]

# Fetch data
data = yf.download(ticker, period="1y")
stock = yf.Ticker(ticker)
info = stock.info

st.subheader(f"📈 {selected_stock} - Candlestick Chart")
fig = go.Figure(data=[
    go.Candlestick(x=data.index,
                   open=data['Open'], high=data['High'],
                   low=data['Low'], close=data['Close'])
])
fig.update_layout(xaxis_rangeslider_visible=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# Tabs for analysis
fundamental_tab, technical_tab, prediction_tab = st.tabs(["📊 Fundamental Analysis", "📈 Technical Analysis", "🔮 Prediction"])

with fundamental_tab:
    st.markdown("### 🧮 Key Fundamental Ratios")
    roe = info.get('returnOnEquity', None)
    de = info.get('debtToEquity', None)
    eps = info.get('trailingEps', None)
    pe = info.get('trailingPE', None)

    st.metric("Return on Equity (ROE)", f"{roe*100:.2f}%" if roe else "N/A")
    st.markdown("ROE = [(Net Income – Preference Dividend) / Avg Shareholders’ Equity] × 100")
    st.markdown("A high, consistent, and rising ROE indicates strong operational efficiency.")

    st.metric("Debt-to-Equity Ratio (D/E)", f"{de:.2f}" if de else "N/A")
    st.markdown("D/E = Total Debt / Total Equity")
    st.markdown("A lower and steadily declining D/E ratio suggests financial stability.")

    st.metric("Earnings Per Share (EPS)", f"₹{eps:.2f}" if eps else "N/A")
    st.markdown("EPS = (Net Income – Preference Dividend) / Avg Shares")

    st.metric("Price-to-Earnings Ratio (P/E)", f"{pe:.2f}" if pe else "N/A")
    st.markdown("P/E = Current Price / EPS")
    st.markdown("Lower P/E may indicate undervaluation, but sector context matters.")

with technical_tab:
    st.markdown("### 📉 Technical Indicators")

    # Moving Averages
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    st.line_chart(data[['Close', 'SMA50', 'SMA200']])

    # RSI
    rsi = ta.momentum.RSIIndicator(data['Close'], window=14)
    data['RSI'] = rsi.rsi()
    st.line_chart(data[['RSI']])
    st.markdown("RSI above 70 = Overbought, below 30 = Oversold")

    # Support and Resistance (basic version using rolling min/max)
    data['Support'] = data['Low'].rolling(window=20).min()
    data['Resistance'] = data['High'].rolling(window=20).max()
    st.line_chart(data[['Close', 'Support', 'Resistance']])
    
    # Volume
    st.bar_chart(data['Volume'])

with prediction_tab:
    st.markdown("### 🔮 Simple Price Prediction (Linear Trend)")
    from sklearn.linear_model import LinearRegression
    import numpy as np

    data = data.dropna()
    data['Date'] = data.index
    data['Date_ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)

    X = data['Date_ordinal'].values.reshape(-1,1)
    y = data['Close'].values.reshape(-1,1)

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    st.line_chart(pd.DataFrame({'Actual': y.flatten(), 'Predicted': y_pred.flatten()}, index=data.index))
    st.markdown("This is a basic linear prediction. For more accuracy, consider ARIMA, LSTM, or Prophet.")

