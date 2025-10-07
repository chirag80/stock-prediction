# main.py

# ===============================================================
# IMPORTS
# ===============================================================
# Import for Option 2
from advanced_predictor import run_advanced_prediction

# Imports for Option 1 and 3
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# ===============================================================
# OPTION 1: ORIGINAL SIMPLE MODEL
# ===============================================================
def run_simple_prediction():
    """
    This function contains the first version of the simple linear regression model.
    It's a self-contained function for quick execution.
    """
    print("\n--- Running Simple Linear Regression Model (Version 1) ---")
    
    ticker_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    try:
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data for {ticker_symbol}")
            return
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    stock_data['Prediction'] = stock_data['Close'].shift(-1)
    stock_data.dropna(inplace=True)
    X = stock_data[['Close']]
    y = stock_data['Prediction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Simple model trained.")

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Simple Model RMSE: ${rmse:.2f}")

    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}).sort_index()
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['Actual'], label='Actual Prices')
    plt.plot(results_df['Predicted'], label='Predicted Prices', linestyle='--')
    plt.title('Simple Model (V1): Actual vs. Predicted', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("--- Simple Model (Version 1) Finished ---")


# ===============================================================
# OPTION 3: USER'S PROVIDED CODE, RESTRUCTURED
# ===============================================================

# --- All helper functions from the user's provided code ---

def fetch_stock_data(ticker_symbol, start_date, end_date):
    """Fetches stock data from Yahoo Finance."""
    print(f"\n--- Fetching data for {ticker_symbol} ---")
    try:
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data found for {ticker_symbol}. It might be delisted or the ticker is incorrect.")
            return None
        else:
            print(f"Successfully downloaded data for {ticker_symbol}")
            return stock_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_closing_price(stock_data, ticker_symbol):
    """Plot the closing price history."""
    if stock_data is None or stock_data.empty:
        print("No data to plot.")
        return
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'])
    plt.title(f'{ticker_symbol} Stock Closing Price History', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)
    plt.grid(True)
    plt.show()

def prepare_prediction_data(stock_data):
    """Create features/target for predicting next day's close price."""
    if stock_data is None or stock_data.empty:
        print("No data available to prepare prediction dataset.")
        return None
    stock_data = stock_data.copy()
    stock_data['Prediction'] = stock_data['Close'].shift(-1)
    stock_data.dropna(inplace=True)
    X = stock_data[['Close']]
    y = stock_data['Prediction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """Train a LinearRegression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model

def evaluate_and_plot(results_df, X_test):
    """Evaluate predictions and plot Actual vs Predicted in feature space."""
    y_test = results_df['Actual']
    predictions = results_df['Predicted'].values
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"Model Performance on Test Data:")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print("\nSample of Predictions:")
    print(results_df.head())
    plt.figure(figsize=(14, 7))
    plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
    try:
        X_plot = X_test.squeeze()
        sorted_idx = np.argsort(X_plot.values)
        plt.plot(X_plot.iloc[sorted_idx], predictions[sorted_idx], color='red', linewidth=2, label='Predicted Prices (Regression Line)')
    except Exception:
        plt.plot(X_test, predictions, color='red', linewidth=2, label='Predicted Prices (Regression Line)')
    plt.title('Model Prediction vs. Actual Prices', fontsize=16)
    plt.xlabel('Current Day Price (USD)', fontsize=12)
    plt.ylabel('Next Day Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions_over_time(results_df):
    """Plot actual vs predicted prices over time."""
    results_df = results_df.sort_index()
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['Actual'], label='Actual Prices')
    plt.plot(results_df['Predicted'], label='Predicted Prices', linestyle='--')
    plt.title('Stock Price Prediction: Actual vs. Predicted', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Wrapper function to run the user's code as a single menu option ---
def run_user_code_as_option_3():
    """
    This function executes the user's provided script logic.
    """
    print("\n--- Running User's Original main.py Code ---")
    # Define parameters from the user's script
    ticker_symbol = 'AAPL'
    start_date = '2021-01-01'
    # NOTE: The original end_date was in the future. It has been changed to today's date.
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

    # The main logic from the user's `if __name__ == '__main__':` block
    stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)
    if stock_data is not None:
        prep = prepare_prediction_data(stock_data)
        if prep is None:
            print("Preparation of prediction data was skipped or failed.")
        else:
            X_train, X_test, y_train, y_test = prep
            model = train_linear_regression(X_train, y_train)
            if model is None:
                print("Model training could not be completed.")
            else:
                predictions = model.predict(X_test)
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=y_test.index)
                evaluate_and_plot(results_df, X_test)
                plot_predictions_over_time(results_df)
    
    print("--- User's Original Code Finished ---")


# ===============================================================
# MAIN APP CONTROLLER
# ===============================================================
if __name__ == '__main__':
    while True:
        print("\n--- Stock Prediction App Menu ---")
        print("1. Run Simple Linear Regression Model (Version 1)")
        print("2. Run Advanced Random Forest Forecast")
        print("3. Run User's Original Script (Refactored Simple Model)")
        print("4. Exit")
        
        choice = input("Enter your choice (1, 2, 3, or 4): ")
        
        if choice == '1':
            run_simple_prediction()
        elif choice == '2':
            run_advanced_prediction()
        elif choice == '3':
            run_user_code_as_option_3()
        elif choice == '4':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
