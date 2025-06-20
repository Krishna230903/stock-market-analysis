import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Analysis App", layout="wide")

st.title("📊 Interactive Stock Analysis App")
st.markdown("Enter an Indian stock ticker (e.g., `TCS.NS`, `INFY.NS`, `RELIANCE.NS`) to view historical analysis and compare with NIFTY 50.")

# User Inputs
ticker = st.text_input("Enter Stock Ticker:", "TCS.NS")
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if ticker:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            st.warning("No data found. Please check the ticker symbol.")
        else:
            st.subheader(f"📈 Price Chart: {ticker}")
            st.line_chart(df[['Open', 'High', 'Low', 'Close']])

            # Moving Averages
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()

            st.subheader("📉 Moving Averages")
            st.line_chart(df[['Close', 'SMA50', 'SMA200']])

            # Fundamental Ratios
            st.subheader("📊 Fundamental Ratios")
            info = stock.info
            roe = info.get('returnOnEquity', None)
            de = info.get('debtToEquity', None)
            eps = info.get('trailingEps', None)
            pe = info.get('trailingPE', None)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Return on Equity (ROE)", f"{roe:.2%}" if roe else "N/A")
                st.metric("Debt-to-Equity Ratio", f"{de:.2f}" if de else "N/A")
            with col2:
                st.metric("Earnings Per Share (EPS)", f"{eps:.2f}" if eps else "N/A")
                st.metric("Price-to-Earnings Ratio (P/E)", f"{pe:.2f}" if pe else "N/A")

            # Compare with Nifty 50
            st.subheader("📊 Stock vs NIFTY 50 Cumulative Returns")
            index = yf.Ticker("^NSEI")
            df_index = index.history(start=start_date, end=end_date)

            df['Returns'] = df['Close'].pct_change()
            df_index['Returns'] = df_index['Close'].pct_change()

            df['Cumulative'] = (1 + df['Returns']).cumprod()
            df_index['Cumulative'] = (1 + df_index['Returns']).cumprod()

            comp_df = pd.DataFrame({
                ticker: df['Cumulative'],
                'NIFTY 50': df_index['Cumulative']
            }).dropna()

            st.line_chart(comp_df)

    except Exception as e:
        st.error(f"Error fetching data: {e}")

