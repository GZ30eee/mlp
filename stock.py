import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit App
st.title("Stock Price Trend Analysis")

# User Input for Stock Symbol and Date Range
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL")
start_date = st.date_input("Select Start Date:", pd.to_datetime("2020-01-01"))
end_date = st.date_input("Select End Date:", pd.to_datetime("2024-01-01"))

if st.button("Fetch and Analyze Stock Data"):
    try:
        # Fetch Stock Data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        if stock_data.empty:
            st.error("No data fetched. Please check the stock symbol and date range.")
        else:
            st.write("### Stock Data (First 5 Rows)")
            st.write(stock_data.head())

            # Check if 'Close' column exists
            if 'Close' not in stock_data.columns:
                st.error("The 'Close' column is missing in the fetched data. Unable to calculate moving averages.")
            else:
                # Calculate Moving Averages
                stock_data['50_day_MA'] = stock_data['Close'].rolling(window=50, min_periods=1).mean()
                stock_data['200_day_MA'] = stock_data['Close'].rolling(window=200, min_periods=1).mean()

                # Ensure Moving Averages were calculated
                if '50_day_MA' not in stock_data.columns or '200_day_MA' not in stock_data.columns:
                    st.error("Moving averages could not be calculated. Ensure the dataset has enough data points.")
                else:
                    # Plot Stock Prices and Moving Averages
                    fig, ax = plt.subplots(figsize=(14, 8))
                    ax.plot(stock_data['Close'], label="Stock Price", color='blue', alpha=0.6)
                    ax.plot(stock_data['50_day_MA'], label="50-Day Moving Average", color='red', alpha=0.8)
                    ax.plot(stock_data['200_day_MA'], label="200-Day Moving Average", color='green', alpha=0.8)
                    ax.set_title(f"Stock Price Trend Analysis for {stock_symbol}", fontsize=16)
                    ax.set_xlabel("Date", fontsize=14)
                    ax.set_ylabel("Price (USD)", fontsize=14)
                    ax.legend(loc='upper left')
                    ax.grid(True)
                    st.pyplot(fig)

                    # # Drop NaN values to prepare for latest value extraction
                    # stock_data = stock_data.dropna(subset=['50_day_MA', '200_day_MA'])

                    # if stock_data.empty:
                    #     st.error("Not enough data to calculate moving averages. Please adjust the date range.")
                    # else:
                        # Extract the latest values
                        # latest_price = stock_data['Close'].iloc[-1]
                        # latest_50_day_MA = stock_data['50_day_MA'].iloc[-1]
                        # latest_200_day_MA = stock_data['200_day_MA'].iloc[-1]

                        # # Display results
                        # st.write(f"Latest Stock Price: ${latest_price:.2f}")
                        # st.write(f"Latest 50-Day Moving Average: ${latest_50_day_MA:.2f}")
                        # st.write(f"Latest 200-Day Moving Average: ${latest_200_day_MA:.2f}")

                        # # Basic trend indication
                        # if latest_price > latest_50_day_MA:
                        #     st.success("The stock is in an upward trend relative to the 50-day moving average.")
                        # else:
                        #     st.warning("The stock is in a downward trend relative to the 50-day moving average.")

                        # if latest_50_day_MA > latest_200_day_MA:
                        #     st.success("The stock is in a short-term uptrend compared to the long-term trend.")
                        # else:
                        #     st.warning("The stock is in a short-term downtrend compared to the long-term trend.")

    except Exception as e:
        st.error(f"An error occurred: {e}")