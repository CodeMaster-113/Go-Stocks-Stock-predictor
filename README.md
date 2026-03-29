# 📈 Live Stock Prediction with Sentiment Analysis

A Python-based stock prediction tool that combines **historical price data**, **technical indicators**, and **news sentiment analysis** to provide **live trading signals** (BUY/HOLD/SELL) and visualize price trends. The project fetches data from Yahoo Finance and NSE, trains an XGBoost regression model, and updates live predictions with interactive graphs.

---

## 🔹 Features

- **NSE Stock Search** – Find stocks by company name.  
- **Historical Data Analysis** – 60-day stock data with 5-minute intervals.  
- **Technical Indicators** – Computes Moving Averages (MA), Exponential Moving Averages (EMA), MACD, Momentum, and Volatility.  
- **News Sentiment Analysis** – Fetches latest stock news and calculates sentiment scores using VADER.  
- **ML Predictions** – XGBoost model predicts next-minute closing prices.  
- **Live Prediction Visualization** – Updates price chart and signals every 15 seconds.  
- **Trading Signals** – BUY, SELL, or HOLD recommendation based on predicted price and threshold.  
- **Accuracy Estimation** – Shows live model R² score as confidence measure.  

---

## 🔹 Tech Stack

- **Python 3.x**  
- **pandas, yfinance, numpy** – Data handling and financial calculations  
- **matplotlib** – Interactive live plots  
- **scikit-learn** – Train/test split and metrics  
- **xgboost** – Regression model for prediction  
- **newsapi-python** – Fetch latest stock news  
- **vaderSentiment** – Sentiment analysis  

---

## 🔹 How It Works

1. Enter the stock name.  
2. The app searches NSE for matching stocks.  
3. Fetches **historical stock prices** (last 60 days).  
4. Fetches **live stock data** (1-day, 1-minute interval).  
5. Fetches **latest news** and computes sentiment scores.  
6. Trains an **XGBoost regression model**.  
7. Computes **technical indicators** for live prediction.  
8. Generates a **real-time Matplotlib graph** showing prices, prediction, and signal.  
9. Updates every **15 seconds** for live trading insights.  

---

## 🔹 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/stock-live-prediction.git
cd stock-live-prediction
