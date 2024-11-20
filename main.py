import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import asyncio
import websockets
import json
from threading import Thread, Lock
import time

# Global variables and lock for thread-safe updates
predictions = {}
lock = Lock()

# Step 1: Fetch Historical Data
def fetch_data(ticker, interval="1m", period="5d"):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data fetched. Check the ticker or the interval/period combination.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Step 2: Feature Engineering (Add Technical Indicators)
def add_indicators(data):
    """
    Add technical indicators like SMA, RSI, etc., to the data.
    """
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data = data.dropna()  # Drop rows with NaN values
    return data

# Step 3: Prepare Data for LSTM
def prepare_data(data, feature_col, lookback):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[feature_col]])
    X = [scaled_data[i-lookback:i, 0] for i in range(lookback, len(scaled_data))]
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

# Step 4: Build LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Train Model
def train_model(data, lookback):
    X, scaler = prepare_data(data, 'Close', lookback)
    model = build_model((lookback, 1))
    model.fit(X[:-1], X[1:, 0], epochs=10, batch_size=32, verbose=1)
    return model, scaler

# Step 6: Predict Future Price
def predict_future(model, recent_data, scaler):
    recent_data = recent_data.reshape(1, -1, 1)
    predicted_price = model.predict(recent_data)
    return scaler.inverse_transform(predicted_price)

# WebSocket Server to Broadcast Predictions
async def socket_server(websocket):
    global predictions
    while True:
        with lock:
            if predictions:
                await websocket.send(json.dumps(predictions))
        await asyncio.sleep(1)  # Broadcast every second

# Real-Time Prediction Task
def prediction_task(ticker, lookback=60):
    global predictions

    while True:
        # Fetch the latest data
        data = fetch_data(ticker, interval="1m", period="5d")
        if data.empty:
            print("No data fetched. Retrying in 1 minute.")
            time.sleep(60)
            continue

        data = add_indicators(data)
        if len(data) < lookback:
            print("Not enough data for prediction. Waiting for more data...")
            time.sleep(60)
            continue

        # Prepare data for prediction
        X, scaler = prepare_data(data, 'Close', lookback)
        recent_data = X[-1]

        # Train a fresh model with the latest data
        model, scaler = train_model(data, lookback)

        # Predict next price dynamically
        predicted_price = predict_future(model, recent_data, scaler)

        # Update predictions with a lock for thread safety
        with lock:
            predictions = {
                "time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "predicted_price": float(predicted_price[0][0])
            }
        print(f"Updated Prediction: {predictions}")
        time.sleep(60)  # Update prediction every minute

# Main Execution
async def main():
    ticker = "^NSEI"  # Nifty 50 Index

    # Start Prediction Task in a Separate Thread
    prediction_thread = Thread(target=prediction_task, args=(ticker,))
    prediction_thread.daemon = True
    prediction_thread.start()

    # Start WebSocket Server
    print("Starting WebSocket server on ws://localhost:8765...")
    async with websockets.serve(socket_server, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
