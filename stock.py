import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date

# Streamlit App Configuration
st.set_page_config(page_title="Stock Trend Analysis", layout="wide")

# Sidebar for User Input
st.sidebar.title("📊 Stock Analysis Tool")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date:", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date:", date.today())

st.title(f"📈 {stock_symbol} Stock Price Trend Analysis")

# Function to Calculate RSI
def calculate_rsi(data, period=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Fetch Data Button
if st.sidebar.button("🔍 Analyze Stock Data"):
    if start_date >= end_date:
        st.error("❌ Invalid date range. The start date must be before the end date.")
    else:
        try:
            # Fetch Stock Data
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

            if stock_data.empty:
                st.error("⚠️ No data found. Please check the stock symbol or date range.")
            else:
                # Moving Averages
                stock_data["50_day_MA"] = stock_data["Close"].rolling(window=50, min_periods=1).mean()
                stock_data["200_day_MA"] = stock_data["Close"].rolling(window=200, min_periods=1).mean()

                # RSI Calculation
                stock_data["RSI"] = calculate_rsi(stock_data)

                # Buy/Sell Signals (Example: Simple Moving Average Crossover)
                stock_data["Buy_Signal"] = (stock_data["50_day_MA"] > stock_data["200_day_MA"]) & (
                    stock_data["50_day_MA"].shift(1) <= stock_data["200_day_MA"].shift(1))
                stock_data["Sell_Signal"] = (stock_data["50_day_MA"] < stock_data["200_day_MA"]) & (
                    stock_data["50_day_MA"].shift(1) >= stock_data["200_day_MA"].shift(1))

                # 📌 Candlestick Chart
                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data["Open"],
                    high=stock_data["High"],
                    low=stock_data["Low"],
                    close=stock_data["Close"],
                    name="Candlestick"
                ))

                # Moving Averages
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["50_day_MA"],
                                         mode="lines", name="50-Day MA", line=dict(color='red')))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["200_day_MA"],
                                         mode="lines", name="200-Day MA", line=dict(color='green')))

                # Buy/Sell Signal Markers
                buy_points = stock_data[stock_data["Buy_Signal"]]
                sell_points = stock_data[stock_data["Sell_Signal"]]

                fig.add_trace(go.Scatter(
                    x=buy_points.index, y=buy_points["Close"],
                    mode="markers", name="🟢 Buy Signal",
                    marker=dict(color="lime", size=10)
                ))

                fig.add_trace(go.Scatter(
                    x=sell_points.index, y=sell_points["Close"],
                    mode="markers", name="🔴 Sell Signal",
                    marker=dict(color="red", size=10)
                ))

                fig.update_layout(title=f"{stock_symbol} Stock Price",
                                  xaxis_title="Date", yaxis_title="Price (USD)",
                                  xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # 📉 Volume Chart
                st.subheader("📊 Trading Volume")
                volume_fig = go.Figure()
                volume_fig.add_trace(go.Bar(
                    x=stock_data.index, y=stock_data["Volume"], name="Volume",
                    marker=dict(color="blue")
                ))
                volume_fig.update_layout(title="Trading Volume Over Time", xaxis_title="Date", yaxis_title="Volume")
                st.plotly_chart(volume_fig, use_container_width=True)

                # 📊 RSI Chart
                st.subheader("📈 Relative Strength Index (RSI)")
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(
                    x=stock_data.index, y=stock_data["RSI"], mode="lines", name="RSI", line=dict(color='purple')
                ))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                rsi_fig.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI Level")
                st.plotly_chart(rsi_fig, use_container_width=True)

                # 📌 Trend Analysis
                if not stock_data.empty:
                    latest_price = stock_data["Close"].dropna().iloc[-1] if not stock_data["Close"].dropna().empty else None
                    latest_50_MA = stock_data["50_day_MA"].dropna().iloc[-1] if not stock_data["50_day_MA"].dropna().empty else None
                    latest_200_MA = stock_data["200_day_MA"].dropna().iloc[-1] if not stock_data["200_day_MA"].dropna().empty else None
                else:
                    latest_price = latest_50_MA = latest_200_MA = None

                # Convert to float before formatting
                if latest_price is not None:
                    latest_price = float(latest_price)

                if latest_50_MA is not None:
                    latest_50_MA = float(latest_50_MA)

                if latest_200_MA is not None:
                    latest_200_MA = float(latest_200_MA)

                # ✅ Use only the column version to avoid duplication
                col1, col2, col3 = st.columns(3)
                col1.metric("Latest Price", f"${latest_price:.2f}")
                col2.metric("50-Day Moving Avg", f"${latest_50_MA:.2f}")
                col3.metric("200-Day Moving Avg", f"${latest_200_MA:.2f}")


                if latest_price > latest_50_MA > latest_200_MA:
                    st.success("📊 Strong Uptrend: The stock is consistently rising.")
                elif latest_price < latest_50_MA < latest_200_MA:
                    st.warning("📉 Downtrend: The stock is consistently falling.")
                else:
                    st.info("⚖️ Mixed Trend: Watch for breakout signals.")

                # 📥 Data Download Option
                csv = stock_data.to_csv().encode("utf-8")
                st.download_button(label="📥 Download Stock Data",
                                   data=csv, file_name=f"{stock_symbol}_data.csv",
                                   mime="text/csv")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
