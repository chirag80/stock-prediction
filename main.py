import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("yFinance version:", yf.__version__)

ticker_symbol = 'AAPL'
start_date = '2021-01-01'
end_date = '2025-10-04'

def fetch_stock_data(ticker_symbol, start_date, end_date):
    try:
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

        # Check if data was downloaded
        if stock_data.empty:
            print(f"No data found for {ticker_symbol}. It might be delisted or the ticker is incorrect.")
            return None
        else:
            print(f"Successfully downloaded data for {ticker_symbol}")
            # Display the first 5 rows of the downloaded data
            # print(stock_data.head())
            return stock_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def plot_closing_price(stock_data, ticker_symbol):
    """Plot the closing price history from the provided DataFrame.

    Args:
        stock_data (pd.DataFrame): DataFrame containing at least a 'Close' column and a DatetimeIndex.
        ticker_symbol (str): Ticker symbol used for the plot title.
    """
    if stock_data is None or stock_data.empty:
        print("No data to plot.")
        return

    # Create a plot of the closing price
    plt.figure(figsize=(14, 7))  # Set the figure size for better readability
    plt.plot(stock_data['Close'])

    # Add titles and labels for clarity
    plt.title(f'{ticker_symbol} Stock Closing Price History', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)
    plt.grid(True)  # Add a grid for easier reading

    # Show the plot
    plt.show()


def prepare_prediction_data(stock_data):
    """Create features/target for predicting next day's close price.

    Returns: X_train, X_test, y_train, y_test (or None on failure)
    """
    if stock_data is None or stock_data.empty:
        print("No data available to prepare prediction dataset.")
        return None

    # 1. Create the target variable (next day's Close)
    stock_data = stock_data.copy()
    stock_data['Prediction'] = stock_data['Close'].shift(-1)

    # 2. Remove the last row which will have a NaN for 'Prediction'
    stock_data.dropna(inplace=True)

    # 3. Define features and target
    X = stock_data[['Close']]
    y = stock_data['Prediction']

    # 4. Split the data into training and testing sets
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("scikit-learn is not installed. Install it with: pip install scikit-learn")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    """Train a LinearRegression model on the provided training data.

    Returns the trained model, or None if sklearn is not available.
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("scikit-learn is not installed. Install it with: pip install scikit-learn")
        return None

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model training complete!")
    return model


def evaluate_and_plot(results_df, X_test):
    """Evaluate the predictions contained in results_df and plot Actual vs Predicted in feature space.

    Args:
        results_df (pd.DataFrame): must contain columns 'Actual' and 'Predicted' and be aligned with X_test by position/index.
        X_test (pd.DataFrame): feature DataFrame (single column 'Close') used for scatter x-axis.
    Returns:
        results_df (pd.DataFrame): the same DataFrame (convenience return for reuse).
    """
    if results_df is None or results_df.empty:
        print("No results to evaluate.")
        return None

    try:
        from sklearn.metrics import mean_squared_error
    except ImportError:
        print("scikit-learn is not installed. Install it with: pip install scikit-learn")
        return None

    # Extract actuals and predictions
    y_test = results_df['Actual']
    predictions = results_df['Predicted'].values

    # Evaluate
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"Model Performance on Test Data:")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

    print("\nSample of Predictions:")
    print(results_df.head())

    # Plot actual vs predicted in feature space
    plt.figure(figsize=(14, 7))
    plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
    # For plotting a regression line we need X_test sorted
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

    return results_df


def plot_predictions_over_time(results_df):
    """Plot actual vs predicted prices over time from a results DataFrame.

    Args:
        results_df (pd.DataFrame): must contain 'Actual' and 'Predicted' and have a DatetimeIndex.
    """
    if results_df is None or results_df.empty:
        print("No results to plot over time.")
        return

    # Ensure the index is sorted in chronological order for the line plot
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


if __name__ == '__main__':
    stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)
    if stock_data is not None:
        # plot_closing_price(stock_data, ticker_symbol)
        
        # Prepare data for a simple next-day prediction task
        prep = prepare_prediction_data(stock_data)
        if prep is None:
            print("Preparation of prediction data was skipped or failed.")
        else:
            X_train, X_test, y_train, y_test = prep
            model = train_linear_regression(X_train, y_train)
            if model is None:
                print("Model training could not be completed.")
            else:
                # Compute predictions once and build a results DataFrame aligned with y_test's index
                predictions = model.predict(X_test)
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=y_test.index)

                # Evaluate and plot using the results DataFrame
                evaluate_and_plot(results_df, X_test)
                plot_predictions_over_time(results_df)

