import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Helper Functions ---

def fetch_data_for_advanced_model(ticker, start, end):
    """Fetches and returns the stock data."""
    print("Fetching data...")
    try:
        # FIX: Use the Ticker object's history method to avoid MultiIndex column issues.
        data = yf.Ticker(ticker).history(start=start, end=end)
        
        if data.empty:
            print(f"No data fetched for {ticker}. Ticker might be wrong or no data in range.")
            return None
        print("Data fetched successfully.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def create_features(df):
    """Create technical analysis features from the stock data."""
    print("Creating features...")
    if df is None or df.empty:
        print("Input DataFrame is empty. Cannot create features.")
        return None
        
    df_feat = df.copy()
    try:
        # Using pandas_ta to append features. Default names (e.g., 'SMA_50') are used.
        df_feat.ta.sma(length=50, append=True)
        df_feat.ta.sma(length=200, append=True)
        df_feat.ta.rsi(length=14, append=True)
        df_feat.ta.bbands(length=20, append=True)
        df_feat.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # The 'Prediction' column is our target, created before dropping NaNs
        df_feat['Prediction'] = df_feat['Close'].shift(-1)
        
        # Drop rows with NaN values created by the indicators (especially the 200-day SMA)
        df_feat.dropna(inplace=True)
        return df_feat
    except Exception as e:
        print(f"Error creating features: {e}")
        return None

# --- Main Execution Function ---

def run_advanced_prediction():
    """
    Main function to run the advanced prediction model.
    """
    try:
        # 1. Parameters
        # EDIT: Ask the user for the ticker symbol instead of hardcoding it.
        ticker_symbol = input("Please enter the stock ticker symbol (e.g., AAPL, GOOGL): ").upper()
        start_date = '2015-01-01'
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        days_to_predict = 180 # Approx 6 months

        # 2. Fetch and Prepare Data
        raw_data = fetch_data_for_advanced_model(ticker_symbol, start_date, end_date)
        if raw_data is None:
            return

        data_with_features = create_features(raw_data)
        if data_with_features is None or data_with_features.empty:
            print("Could not create features or not enough data. Please use a larger date range.")
            return

        # 3. Split the data using a time-based split
        X = data_with_features.drop('Prediction', axis=1)
        y = data_with_features['Prediction']
        
        split_point = int(len(X) * 0.9)
        X_train, y_train = X[:split_point], y[:split_point]
        
        if len(X_train) == 0:
            print("Not enough data to perform train/test split. Try a larger date range.")
            return

        # 4. Train the Random Forest Model
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
        model.fit(X_train, y_train)
        print(f"Model trained. OOB Score: {model.oob_score_:.4f}")

        # 5. Corrected Iterative Forecasting
        print(f"Forecasting future {days_to_predict} prices...")
        
        # We need a history to calculate indicators correctly. Start with all available feature data.
        history = X.copy()
        future_predictions = []

        for i in tqdm(range(days_to_predict), desc="Predicting"):
            # Get the last row of the history to make a prediction
            last_features = history.iloc[-1:].copy()
            
            # FIX: Ensure the feature order is the same as the training data
            last_features_ordered = last_features[X_train.columns]
            
            # Predict the next day's price
            next_prediction = model.predict(last_features_ordered)[0]
            future_predictions.append(next_prediction)

            # Create a new row for the next day
            next_date = history.index[-1] + pd.Timedelta(days=1)
            new_row_data = {
                'Open': history['Close'].iloc[-1],
                'High': next_prediction * 1.01, # Approximate high
                'Low': next_prediction * 0.99,   # Approximate low
                'Close': next_prediction,
                # FIX: Removed 'Adj Close' as it's not in the training features
                'Volume': history['Volume'].iloc[-1], # Assume volume stays constant (simplification)
                'Dividends': 0, # Assume no dividends
                'Stock Splits': 0 # Assume no stock splits
            }
            new_row_df = pd.DataFrame([new_row_data], index=[next_date])
            
            # Append this new row (with only base features) to our history
            history = pd.concat([history, new_row_df])
            
            # Now, recalculate technical indicators on the ENTIRE updated history
            # The library will overwrite the existing indicator columns with new values.
            history.ta.sma(length=50, append=True)
            history.ta.sma(length=200, append=True)
            history.ta.rsi(length=14, append=True)
            history.ta.bbands(length=20, append=True)
            history.ta.macd(fast=12, slow=26, signal=9, append=True)
            
            # FIX: Updated to use the recommended ffill() method to fix the FutureWarning
            history.ffill(inplace=True)

        # 6. Plotting
        print("Plotting results...")
        future_dates = pd.date_range(start=X.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
        future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Forecast'])

        plt.figure(figsize=(15, 8))
        plt.plot(X.index, X['Close'], label='Historical Actual Prices')
        plt.plot(future_df.index, future_df['Forecast'], label='Future Forecast', linestyle='--')
        plt.title(f'{ticker_symbol} Price Forecast', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"An error occurred in the advanced predictor: {e}")

