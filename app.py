pip install yfinance streamlit pandas matplotlib
import yfinance as yf

ticker = "TCS.NS"
stock = yf.Ticker(ticker)
df = stock.history(period="1y")
df.head()
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()
df[['Close', 'SMA50', 'SMA200']].plot(figsize=(12,6))
info = stock.info
roe = info.get('returnOnEquity')
de = info.get('debtToEquity')
eps = info.get('trailingEps')
pe = info.get('trailingPE')

print(f"ROE: {roe}, D/E: {de}, EPS: {eps}, P/E: {pe}")
index = yf.Ticker("^NSEI")
df_index = index.history(period="1y")

# Calculate Cumulative Returns
df['Returns'] = df['Close'].pct_change()
df_index['Returns'] = df_index['Close'].pct_change()

df['Cumulative'] = (1 + df['Returns']).cumprod()
df_index['Cumulative'] = (1 + df_index['Returns']).cumprod()

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(df['Cumulative'], label='TCS')
plt.plot(df_index['Cumulative'], label='NIFTY 50')
plt.legend()
plt.title("Stock vs Market Return")
plt.show()
touch app
