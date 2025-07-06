import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from scipy.signal import argrelextrema

# --- Page Configuration ---
st.set_page_config(
    page_title="Nifty 50 Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- NIFTY 50 Ticker List ---
# Using a dictionary for easy mapping of company names to tickers
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
    "Tata Consumer": "TATACONSUM.NS", "Tata Motors": "TATAMOTORS.NS", "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS", "Titan": "TITAN.NS", "UPL": "UPL.NS", "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS"
}

# --- Data Fetching Functions ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_data(ticker, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, f"No data found for ticker {ticker} in the selected date range."
        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)
        return data, None
    except Exception as e:
        return None, f"An error occurred while fetching data: {e}"

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    """Fetches company information from Yahoo Finance."""
    try:
        return yf.Ticker(ticker).info
    except Exception as e:
        st.warning(f"Could not fetch company info: {e}")
        return {}

# --- Pattern Detection Functions ---
def find_patterns(data, order=5, K=2):
    """
    Detects Double Top, Double Bottom, and Head and Shoulders patterns.
    `order`: How many points on each side to use for local extrema detection.
    `K`: The percentage difference allowed between peaks/troughs.
    """
    highs = data['High']
    lows = data['Low']
    
    # Find local maxima and minima
    peak_indices = argrelextrema(highs.values, np.greater, order=order)[0]
    valley_indices = argrelextrema(lows.values, np.less, order=order)[0]
    
    peaks = highs.iloc[peak_indices]
    valleys = lows.iloc[valley_indices]
    
    patterns = {'double_top': [], 'double_bottom': [], 'head_shoulders': []}
    
    # Double Top Detection
    for i in range(len(peaks) - 1):
        p1 = peaks.iloc[i]
        p2 = peaks.iloc[i+1]
        
        # Check if peaks are close in price
        if abs(p1 - p2) / p2 <= K/100:
            # Find the intervening valley
            intervening_valleys = valleys[(valleys.index > peaks.index[i]) & (valleys.index < peaks.index[i+1])]
            if not intervening_valleys.empty:
                patterns['double_top'].append((peaks.index[i], peaks.index[i+1]))

    # Double Bottom Detection
    for i in range(len(valleys) - 1):
        v1 = valleys.iloc[i]
        v2 = valleys.iloc[i+1]
        
        if abs(v1 - v2) / v2 <= K/100:
            intervening_peaks = peaks[(peaks.index > valleys.index[i]) & (peaks.index < valleys.index[i+1])]
            if not intervening_peaks.empty:
                patterns['double_bottom'].append((valleys.index[i], valleys.index[i+1]))

    # Head and Shoulders Detection
    for i in range(len(peaks) - 2):
        s1_idx, h_idx, s2_idx = peaks.index[i], peaks.index[i+1], peaks.index[i+2]
        s1, h, s2 = peaks.iloc[i], peaks.iloc[i+1], peaks.iloc[i+2]

        # Head must be higher than both shoulders
        if h > s1 and h > s2:
            # Shoulders should be roughly symmetrical
            if abs(s1 - s2) / s2 <= (K+5)/100: # Looser condition for shoulders
                patterns['head_shoulders'].append((s1_idx, h_idx, s2_idx))
                
    return patterns


# --- UI Layout ---
st.title("📈 Nifty 50 Stock Market Analyzer")
st.markdown("An advanced tool for fundamental and technical analysis of NIFTY 50 stocks.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Controls")
    selected_stock_name = st.selectbox("Select a Stock:", sorted(nifty50_stocks.keys()))
    ticker = nifty50_stocks[selected_stock_name]
    
    date_range = st.date_input(
        "Select Date Range",
        [pd.to_datetime("2022-01-01"), pd.to_datetime("today")],
        min_value=pd.to_datetime("2010-01-01"),
        max_value=pd.to_datetime("today")
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        st.error("Please select a valid date range.")
        st.stop()
    
    st.info(f"**Selected Ticker:** `{ticker}`")

# --- Main Application Logic ---
# Fetch data
data, error_msg = fetch_data(ticker, start_date, end_date)
info = get_company_info(ticker)

if error_msg:
    st.error(error_msg)
    st.stop()

if data is None or data.empty:
    st.error(f"⚠️ No historical data found for **{selected_stock_name} ({ticker})** for the selected period. Please try a different date range.")
    st.stop()

# --- Main Content Area ---
st.header(f"{info.get('longName', selected_stock_name)} ({ticker})")

# --- Candlestick Chart ---
st.subheader("📊 Candlestick Chart")
fig_candlestick = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name=ticker
)])
fig_candlestick.update_layout(
    xaxis_rangeslider_visible=False,
    height=450,
    title=f"{selected_stock_name} Price Action",
    yaxis_title="Price (INR)"
)
st.plotly_chart(fig_candlestick, use_container_width=True)


# --- TABS for Analysis ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 **Fundamental Analysis**", 
    "📉 **Technical Analysis**", 
    "📈 **Pattern Recognition**",
    "🌐 **NIFTY 50 Overview**"
])


# --- Tab 1: Fundamental Analysis ---
with tab1:
    st.header("Company Fundamentals")
    if not info:
        st.warning("Could not retrieve fundamental data.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏢 Company Profile")
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
            st.markdown(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', 'N/A')})")
            
            with st.expander("Business Summary"):
                st.write(info.get('longBusinessSummary', 'No summary available.'))

        with col2:
            st.subheader("💰 Key Financial Metrics")
            market_cap = info.get("marketCap")
            roe = info.get("returnOnEquity")
            de_ratio = info.get("debtToEquity")
            eps = info.get("trailingEps")
            pe_ratio = info.get("trailingPE")
            pb_ratio = info.get("priceToBook")
            div_yield = info.get("dividendYield")

            st.metric("Market Cap (Cr)", f"{market_cap / 1e7:.2f}" if market_cap else "N/A")
            st.metric("Return on Equity (ROE)", f"{roe * 100:.2f}%" if roe else "N/A")
            st.metric("Debt-to-Equity Ratio", f"{de_ratio:.2f}" if de_ratio else "N/A")
            st.metric("Earnings Per Share (EPS)", f"{eps:.2f}" if eps else "N/A")
            st.metric("Price-to-Earnings (P/E)", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
            st.metric("Price-to-Book (P/B)", f"{pb_ratio:.2f}" if pb_ratio else "N/A")
            st.metric("Dividend Yield", f"{div_yield * 100:.2f}%" if div_yield else "N/A")

        if de_ratio is not None:
            if de_ratio < 1:
                st.success("✅ **Low Risk:** Debt-to-Equity ratio is below 1, suggesting a healthy balance sheet.")
            elif de_ratio < 2:
                st.warning("⚠️ **Medium Risk:** Debt-to-Equity ratio is between 1 and 2. Caution is advised.")
            else:
                st.error("🚨 **High Risk:** Debt-to-Equity ratio is above 2, indicating high leverage.")

# --- Tab 2: Technical Analysis ---
with tab2:
    st.header("Technical Indicators")
    
    # Moving Averages
    st.subheader("Moving Averages (SMA)")
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='lightblue', width=1)))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', line=dict(color='yellow')))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='orange')))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA200'], name='SMA 200', line=dict(color='red')))
    fig_ma.update_layout(title='Simple Moving Averages', height=400, yaxis_title="Price (INR)")
    st.plotly_chart(fig_ma, use_container_width=True)
    st.info("**Golden Cross:** SMA50 crosses above SMA200 (Bullish). **Death Cross:** SMA50 crosses below SMA200 (Bearish).")

    # RSI
    st.subheader("Relative Strength Index (RSI)")
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=300, yaxis_title="RSI Value")
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.info("RSI > 70 suggests the stock may be overbought. RSI < 30 suggests it may be oversold.")

    # Bollinger Bands
    st.subheader("Bollinger Bands")
    indicator_bb = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['bb_h'] = indicator_bb.bollinger_hband()
    data['bb_l'] = indicator_bb.bollinger_lband()
    data['bb_m'] = indicator_bb.bollinger_mavg()
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='lightblue', width=1)))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['bb_h'], name='Upper Band', line=dict(color='red', dash='dash')))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['bb_l'], name='Lower Band', line=dict(color='green', dash='dash')))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['bb_m'], name='Middle Band (SMA20)', line=dict(color='orange', dash='dash')))
    fig_bb.update_layout(title="Bollinger Bands", height=400, yaxis_title="Price (INR)")
    st.plotly_chart(fig_bb, use_container_width=True)
    st.info("Prices moving outside the bands can signal overbought/oversold conditions. A 'squeeze' (bands coming together) can signal upcoming volatility.")

    # MACD
    st.subheader("Moving Average Convergence Divergence (MACD)")
    macd = ta.trend.MACD(close=data['Close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['macd'], name='MACD Line', line_color='blue'))
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name='Signal Line', line_color='orange'))
    fig_macd.add_trace(go.Bar(x=data.index, y=data['macd_diff'], name='Histogram', marker_color='grey'))
    fig_macd.update_layout(title="MACD", height=300, yaxis_title="Value")
    st.plotly_chart(fig_macd, use_container_width=True)
    st.info("A bullish signal occurs when the MACD line crosses above the Signal line. A bearish signal is the opposite.")

# --- Tab 3: Pattern Recognition ---
with tab3:
    st.header("Chart Pattern Recognition")
    st.info("This tool automatically detects potential reversal patterns. These are suggestions and not financial advice.")
    
    patterns = find_patterns(data, order=10, K=5) # Use a wider window (order) for better detection
    
    fig_patterns = go.Figure()
    fig_patterns.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'
    ))

    # Plot Double Tops
    for dt in patterns['double_top']:
        fig_patterns.add_trace(go.Scatter(
            x=dt, y=data['High'].loc[list(dt)], mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name='Double Top'
        ))
    
    # Plot Double Bottoms
    for db in patterns['double_bottom']:
        fig_patterns.add_trace(go.Scatter(
            x=db, y=data['Low'].loc[list(db)], mode='markers', marker=dict(symbol='triangle-up', color='green', size=12), name='Double Bottom'
        ))

    # Plot Head and Shoulders
    for hs in patterns['head_shoulders']:
        fig_patterns.add_trace(go.Scatter(
            x=hs, y=data['High'].loc[list(hs)], mode='markers', marker=dict(symbol='diamond', color='purple', size=12), name='Head & Shoulders'
        ))

    fig_patterns.update_layout(title="Detected Chart Patterns", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_patterns, use_container_width=True)

    if patterns['double_top']:
        st.warning(f"**Double Top Detected:** Found {len(patterns['double_top'])} potential Double Top pattern(s). This is often a bearish reversal signal.")
    if patterns['double_bottom']:
        st.success(f"**Double Bottom Detected:** Found {len(patterns['double_bottom'])} potential Double Bottom pattern(s). This is often a bullish reversal signal.")
    if patterns['head_shoulders']:
        st.warning(f"**Head and Shoulders Detected:** Found {len(patterns['head_shoulders'])} potential Head and Shoulders pattern(s). This is a bearish reversal pattern.")
    
    if not any(patterns.values()):
        st.info("No significant chart patterns were detected in the selected time frame.")

# --- Tab 4: NIFTY 50 Overview ---
with tab4:
    st.header("NIFTY 50 Index Performance")
    st.info("Comparing the stock's performance against the broader market index (^NSEI).")

    nifty_data, nifty_error = fetch_data('^NSEI', start_date, end_date)

    if nifty_error:
        st.error(nifty_error)
    else:
        # Normalize data to compare performance
        normalized_stock = (data['Close'] / data['Close'].iloc[0]) * 100
        normalized_nifty = (nifty_data['Close'] / nifty_data['Close'].iloc[0]) * 100

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=normalized_stock.index, y=normalized_stock, name=selected_stock_name, line=dict(color='cyan')))
        fig_compare.add_trace(go.Scatter(x=normalized_nifty.index, y=normalized_nifty, name='NIFTY 50 Index', line=dict(color='orange')))
        
        fig_compare.update_layout(
            title=f"Performance Comparison: {selected_stock_name} vs. NIFTY 50",
            yaxis_title="Normalized Performance (Base 100)",
            height=450
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Performance Metrics
        stock_return = (normalized_stock.iloc[-1] - 100)
        nifty_return = (normalized_nifty.iloc[-1] - 100)

        st.subheader("Performance in Selected Period")
        col1, col2 = st.columns(2)
        col1.metric(f"{selected_stock_name} Return", f"{stock_return:.2f}%")
        col2.metric("NIFTY 50 Return", f"{nifty_return:.2f}%")

        if stock_return > nifty_return:
            st.success(f"**Outperformance:** {selected_stock_name} has outperformed the NIFTY 50 index in this period.")
        else:
            st.warning(f"**Underperformance:** {selected_stock_name} has underperformed the NIFTY 50 index in this period.")
