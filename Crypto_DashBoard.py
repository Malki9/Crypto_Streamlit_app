import pandas as pd
import streamlit as st
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import feedparser
from datetime import timedelta


# Assuming the rest of your imports are correct and you've installed all necessary packages

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2018-11-28")
    end_date = st.sidebar.text_input("End Date", "2023-11-28")
    crypto_symbol = st.sidebar.text_input("Crypto Symbol", "BTC")
    return start_date, end_date, crypto_symbol


def get_crypto_name(symbol):
    symbol = symbol.upper()
    if symbol == "BTC":
        return "Bitcoin"
    elif symbol == "ETH":
        return "Ethereum"
    elif symbol == "DOGE":
        return "Dogecoin"
    elif symbol == "AVAX":
        return "Avalanche"
    elif symbol == "BCH":
        return "Bitcoin Cash "
    elif symbol == "BNB":
        return "BNB"
    elif symbol == "DAI":
        return "Dai"
    elif symbol == "DOT":
        return "Polkadot"
    elif symbol == "LEO":
        return "UNUS SED LEO "
    elif symbol == "LINK":
        return "Chainlink"
    elif symbol == "LTC":
        return "Litecoin"
    elif symbol == "MATIC":
        return "Polygon"
    elif symbol == "SOL":
        return "Solana"
    elif symbol == "STETH":
        return "Lido Staked ETH"
    elif symbol == "TON11419":
        return "Toncoin"
    elif symbol == "TRX":
        return "TRON"
    elif symbol == "UNI7083":
        return "Uniswap"
    elif symbol == "USDC":
        return "USD Coin"
    elif symbol == "USDT":
        return "Tether"
    elif symbol == "WBTC":
        return "Wrapped Bitcoin"
    elif symbol == "WEOS":
        return "Wrapped EOS"
    elif symbol == "WTRX":
        return "Wrapped TRON"
    elif symbol == "XRP":
        return "XRP"
    else:
        return "None"


def get_data(symbol, start, end):
    symbol = symbol.upper()
    file_path = f"H:\\Binali Crypto\\crypto csv\\{symbol}-USD.csv"  # Adjust this path
    df = pd.read_csv(file_path)

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    return df.loc[start:end]


def calculate_moving_average(df, window_size=10):
    df['Moving Average'] = df['Close'].rolling(window=window_size).mean()
    return df


def load_all_crypto_data(base_path, start, end):
    crypto_files = os.listdir(base_path)
    all_data = pd.DataFrame()

    for file in crypto_files:
        symbol = file.split('-')[0]
        file_path = os.path.join(base_path, file)
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        df = df.loc[start:end]
        all_data[symbol] = df['Close']

    return all_data


def display_correlated_cryptos(df, symbol):
    symbol = symbol.upper()
    if symbol in df.columns:
        correlations = df.corr()[symbol].sort_values()
        st.subheader(f"Top 10 Positively Correlated Cryptocurrencies with {symbol}")
        st.write(correlations.tail(11)[1:])  # Exclude itself
        st.subheader(f"Top 10 Negatively Correlated Cryptocurrencies with {symbol}")
        st.write(correlations.head(10))
    else:
        st.write(f"No data available for {symbol}")


# Function to display RSS feed news
def display_crypto_news(feed_url):
    st.subheader("Top Cryptocurrency News")
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:5]:  # Display top 5 news stories
            st.write(f"[{entry.title}]({entry.link}) - {entry.published}")
    except Exception as e:
        st.error(f"Error fetching news: {e}")


# LSTM Prediction
def forecast_future(model, recent_data, start_date, n_future_steps=30):
    # Convert the 'Close' column to a NumPy array and normalize it
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = recent_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_prices)

    # Use the most recent data for prediction
    recent_scaled_data = scaled_data[-60:]
    current_batch = recent_scaled_data.reshape(1, 60, 1)

    future_predictions = []
    for i in range(n_future_steps):
        future_pred = model.predict(current_batch)[0, 0]
        future_predictions.append(future_pred)
        # Update the current batch to include the new prediction
        current_batch = np.append(current_batch[:, 1:, :], [[[future_pred]]], axis=1)

    future_dates = [start_date + timedelta(days=i) for i in range(n_future_steps)]
    predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    return pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})


# Identify High and Low Points
def identify_high_low_points(predicted_df):
    high_point = predicted_df.loc[predicted_df['Predicted Price'].idxmax()]
    low_point = predicted_df.loc[predicted_df['Predicted Price'].idxmin()]
    return high_point, low_point

# Function to perform 'What-If' analysis
def what_if_analysis(df, symbol, sell_price, quantity):
    current_price = df['Close'].iloc[-1]  # Get the latest closing price
    potential_revenue = sell_price * quantity
    cost = current_price * quantity
    profit = potential_revenue - cost
    return current_price, potential_revenue, cost, profit


crypto_rss_feeds = {
    "BTC": "https://cointelegraph.com/rss/tag/bitcoin",
    "ETH": "https://cointelegraph.com/rss/tag/ethereum",
    "DOGE": "https://cointelegraph.com/rss/tag/dogecoin",
    "AVAX": "",  # Add the RSS feed URL for Avalanche
    "BCH": "",  # Add the RSS feed URL for Bitcoin Cash
    "BNB": "https://cointelegraph.com/rss/tag/binance-coin",
    # Continue adding more mappings for each cryptocurrency
    # ...
    "XRP": "https://cointelegraph.com/rss/tag/ripple"
}

# Load LSTM model
lstm_model = load_model('H:\Binali Crypto\crypto_prediction_model.h5')



# Main application flow
st.write("""
# Cryptocurrency Dashboard Application
Visually show data on crypto from Nov 28, 2018 to Nov 28, 2023
""")

image_path = "H:\\Binali Crypto\\design.png"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

# User Input for Crypto Data
st.sidebar.header("User Input")
start, end, symbol = get_input()
df = get_data(symbol, start, end)
df_with_ma = calculate_moving_average(df)

# Display Crypto Data
crypto_name = get_crypto_name(symbol)
st.header(f"{crypto_name} Data")
st.write(df)
st.header(f"{crypto_name} Closing Price")
st.line_chart(df['Close'])
st.header(f"{crypto_name} Moving Average")
st.line_chart(df_with_ma[['Close', 'Moving Average']])

# User Inputs for 'What-If' Analysis
st.sidebar.header("What-If Analysis")
sell_price = st.sidebar.number_input(f"Enter the selling price for {symbol}", value=df['Close'].iloc[-1])
quantity = st.sidebar.number_input("Enter the quantity of the coin", min_value=0.0, value=1.0)


# Load and Display Correlated Cryptos
base_path = "H:\\Binali Crypto\\crypto csv"
all_crypto_data = load_all_crypto_data(base_path, pd.to_datetime(start), pd.to_datetime(end))
display_correlated_cryptos(all_crypto_data, symbol)
# Perform and Display 'What-If' Analysis
if st.sidebar.button("Calculate What-If Scenario"):
    current_price, potential_revenue, cost, profit = what_if_analysis(df, symbol, sell_price, quantity)
    st.sidebar.write(f"Current Price of {symbol}: {current_price}")
    st.sidebar.write(f"Potential Revenue: {potential_revenue}")
    st.sidebar.write(f"Cost: {cost}")
    st.sidebar.write(f"Profit/Loss: {profit}")
    
# Predictive Analysis
st.sidebar.header("Predictive Analysis")
if st.sidebar.button("Predict Future Prices"):
    df = get_data(symbol, start, end)
    start_date = pd.to_datetime(end)
    predicted_df = forecast_future(lstm_model, df, start_date, 30)
    st.subheader("Predicted Future Prices")
    st.line_chart(predicted_df.set_index('Date')['Predicted Price'])
    high_point, low_point = identify_high_low_points(predicted_df)
    st.write(f"Predicted Highest Price: {high_point['Predicted Price']} on {high_point['Date'].date()}")
    st.write(f"Predicted Lowest Price: {low_point['Predicted Price']} on {low_point['Date'].date()}")
    # Prepare the plot
    fig, ax = plt.subplots()
    ax.plot(predicted_df['Date'], predicted_df['Predicted Price'], label='Predicted Price')
    ax.scatter(high_point['Date'], high_point['Predicted Price'], color='red', marker='s',
               label='Sell Point (Highest Price)')
    ax.scatter(low_point['Date'], low_point['Predicted Price'], color='green', marker='s',
               label='Buy Point (Lowest Price)')

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f"Predicted Prices with Buy/Sell Points for {get_crypto_name(symbol)}")
    plt.legend()

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Show info about the best buy and sell times
    st.write(f"Best Time to Buy: {low_point['Date'].date()} at price {low_point['Predicted Price']}")
    st.write(f"Best Time to Sell: {high_point['Date'].date()} at price {high_point['Predicted Price']}")

# RSS Feed Section for Crypto News
st.sidebar.header("Cryptocurrency News")
selected_crypto = st.sidebar.selectbox("Choose a cryptocurrency for news", list(crypto_rss_feeds.keys()))
rss_feed_url = crypto_rss_feeds.get(selected_crypto, "default_rss_feed_url")  # Replace with a default RSS feed URL
display_crypto_news(rss_feed_url)
