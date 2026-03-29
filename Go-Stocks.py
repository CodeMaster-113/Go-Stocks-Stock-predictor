import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import r2_score
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------- Load NSE stock list ---------------- #
url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

try:
    df = pd.read_csv(url)
except Exception as e:
    raise SystemExit(f"Failed to load NSE stock list: {e}")


# ---------------- Stock search ---------------- #
def search_stock(stock_name):

    stock_list = df[
        (df['NAME OF COMPANY'].str.strip().str.upper() == stock_name.strip().upper()) |
        (df['NAME OF COMPANY'].str.contains(stock_name, case=False, na=False))
    ]

    if not stock_list.empty:

        stockarr = []

        for i, (name, symbol) in enumerate(
            stock_list[['NAME OF COMPANY', 'SYMBOL']].values, 1
        ):
            print(f"{i} : {name} ({symbol})")
            stockarr.append(symbol)

        return stockarr

    else:
        print("No stock found")
        return None


# ---------------- Training Data ---------------- #
# ---------------- Training Data ---------------- #
def get_training_data(stock):

    try:
        data = yf.download(stock, period="60d", interval="5m", progress=False)
    except Exception as e:
        print(f"Error fetching training data: {e}")
        return pd.DataFrame()

    if not data.empty and data.index.tz is not None:
        data.index = data.index.tz_convert("Asia/Kolkata").tz_localize(None)

    return data

# ---------------- Live Data ---------------- #
def get_live_data(stock):

    try:
        data = yf.download(stock, period="1d", interval="1m", progress=False)
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return pd.DataFrame()

    if not data.empty and data.index.tz is not None:
        data.index = data.index.tz_convert("Asia/Kolkata").tz_localize(None)

    return data


# ---------------- Sentiment ---------------- #
def get_sentiment_score(stock):

    newsapi = NewsApiClient(api_key='eb5f2077be32495095642e5db3953c4a')

    articles = newsapi.get_everything(
        q=stock,
        language='en',
        sort_by='relevancy',
        page=1
    )

    headlines = [article['title'] for article in articles['articles']]

    analyzer = SentimentIntensityAnalyzer()

    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]

    return sum(scores) / len(scores) if scores else 0.0


# ---------------- ML Training ---------------- #
def train_ml_model(data, news_sentiment):

    data['Return'] = data['Close'].pct_change()

    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA20'] = data['Close'].rolling(20).mean()

    data['Volatility'] = data['Close'].rolling(10).std()
    data['Momentum'] = data['Close'] - data['Close'].shift(5)

    data['Sentiment'] = news_sentiment

    # MACD
    data["EMA5"] = data['Close'].ewm(span=5, adjust=False).mean()
    data["EMA10"] = data['Close'].ewm(span=10, adjust=False).mean()

    data["MACD"] = data["EMA5"] - data["EMA10"]
    data["signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["histogram"] = data["MACD"] - data["signal"]

    data = data.dropna()

    X = data[
        [
            'Close','Return','MA5','MA10','MA20',
            'Volatility','Momentum','Volume','Sentiment',
            'EMA5','EMA10','MACD','signal','histogram'
        ]
    ]

    y = data['Close'].shift(-1)

    X = X.iloc[:-1]
    y = y.iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model trained")

    print(
        f"Train R2: {model.score(X_train,y_train):.4f} | "
        f"Test R2: {model.score(X_test,y_test):.4f}"
    )

    return model, X_test, y_test


# ---------------- User input ---------------- #

choice = input("Enter stock name: ").strip()

if not choice:
    raise SystemExit("Please enter a valid stock name.")

result = search_stock(choice)

if result:

    try:
        choice2 = int(input("Select stock number: "))
    except:
        raise SystemExit("Invalid selection")

    stock = result[choice2 - 1] + ".NS"

    news_sentiment = get_sentiment_score(stock)

    data = get_training_data(stock)

    if data.empty:
        print("No data available")
        exit()

    model, X_test, y_test = train_ml_model(data, news_sentiment)

    fig, ax = plt.subplots(figsize=(12,5))


    # ---------------- Live Prediction ---------------- #
    def update(frame):

        live_data = get_live_data(stock)

        if live_data.empty:
            return

        live_data['Return'] = live_data['Close'].pct_change()

        live_data['MA5'] = live_data['Close'].rolling(5).mean()
        live_data['MA10'] = live_data['Close'].rolling(10).mean()
        live_data['MA20'] = live_data['Close'].rolling(20).mean()

        live_data['Volatility'] = live_data['Close'].rolling(10).std()
        live_data['Momentum'] = live_data['Close'] - live_data['Close'].shift(5)

        live_data['Sentiment'] = news_sentiment

        live_data["EMA5"] = live_data['Close'].ewm(span=5, adjust=False).mean()
        live_data["EMA10"] = live_data['Close'].ewm(span=10, adjust=False).mean()

        live_data["MACD"] = live_data["EMA5"] - live_data["EMA10"]
        live_data["signal"] = live_data["MACD"].ewm(span=9, adjust=False).mean()
        live_data["histogram"] = live_data["MACD"] - live_data["signal"]

        live_data = live_data.dropna()

        if live_data.empty:
            return

        close_prices = live_data['Close'].rolling(2).mean()

        ax.clear()

        ax.plot(
            close_prices.index,
            close_prices,
            color="royalblue",
            linewidth=1.5,
            label="Close Price"
        )

        latest_X = live_data[
            [
                'Close','Return','MA5','MA10','MA20',
                'Volatility','Momentum','Volume','Sentiment',
                'EMA5','EMA10','MACD','signal','histogram'
            ]
        ].iloc[-1:]

        next_price = float(model.predict(latest_X)[0])
        last_price = float(close_prices.iloc[-1])

        threshold = 0.002

        if next_price > last_price * (1 + threshold):
            signal = "BUY"
            color = "green"

        elif next_price < last_price * (1 - threshold):
            signal = "SELL"
            color = "red"

        else:
            signal = "HOLD"
            color = "orange"

        preds = model.predict(X_test)

        r2_live = r2_score(y_test, preds)

        if r2_live > 0.7:
            accuracy = "High"
        elif r2_live > 0.4:
            accuracy = "Moderate"
        else:
            accuracy = "Low"

        ax.scatter(close_prices.index[-1], last_price, color=color)

        ax.annotate(
            f'₹{last_price:.2f} — {signal}\nPredicted: ₹{next_price:.2f}\n{accuracy} Chances',
            (close_prices.index[-1], last_price),
            fontsize=10,
            color=color
        )

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()

        ax.set_title(f"{stock} — Live Prediction")
        ax.set_xlabel("Time (IST)")
        ax.set_ylabel("Price (INR)")
        ax.legend()
        ax.grid(True)

        print(
            f"Frame {frame} | Last ₹{last_price:.2f} | Predicted ₹{next_price:.2f} | {signal}"
        )

    ani = FuncAnimation(fig, update, interval=15000)

    plt.show(block=True)