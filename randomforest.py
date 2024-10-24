import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def predict_rf(df_tick):
    df_stockprice = df_tick['StockPrice'].copy().iloc[1:]
    # Calculate percentage change and fill missing values
    df_tick = df_tick.pct_change(fill_method=None).iloc[1:]
    df_tick.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
    df_tick.fillna(df_tick.mean(), inplace=True)  # Fill NaN values with mean

    # Ensure 'StockPrice' is in the DataFrame
    if 'StockPrice' not in df_tick.columns:
        raise ValueError("The 'StockPrice' column is missing from the DataFrame.")

    # Define features (excluding 'StockPrice')
    features = df_tick.columns.drop('StockPrice')

    # --- Include Rolling Statistics ---
    window_size = 3  # Adjust the window size as needed

    # Calculate rolling mean and std for features
    rolling_means = df_tick[features].rolling(window=window_size, min_periods=1).mean()
    rolling_means.columns = [f'{col}_rolling_mean' for col in features]

    rolling_stds = df_tick[features].rolling(window=window_size, min_periods=1).std()
    rolling_stds.columns = [f'{col}_rolling_std' for col in features]

    # Shift rolling features by 1 period to avoid data leakage
    rolling_means = rolling_means.shift(1)
    rolling_stds = rolling_stds.shift(1)

    # Concatenate the new rolling features to df_tick
    df_tick = pd.concat([df_tick, rolling_means, rolling_stds], axis=1)

    # Replace infinite values with NaN after rolling calculations
    df_tick.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with the mean of each column
    df_tick.fillna(df_tick.mean(), inplace=True)

    # Ensure no infinite or NaN values are left
    assert not np.isinf(df_tick.values).any(), "Data contains infinite values."
    assert not np.isnan(df_tick.values).any(), "Data contains NaN values."

    # Update the list of features to include original features and rolling features
    all_features = list(features) + list(rolling_means.columns) + list(rolling_stds.columns)
    X = df_tick[all_features]
    y = df_tick['StockPrice']

    print('Number of columns in X:', len(X.columns))

    # Recalculate the number of samples
    n_samples = len(df_tick)
    train_size = int(0.6 * n_samples)
    valid_size = int(0.2 * n_samples)

    # Re-split indices
    train_indices = df_tick.index[:train_size]
    valid_indices = df_tick.index[train_size:train_size + valid_size]
    test_indices = df_tick.index[train_size + valid_size:]

    # Re-create datasets
    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    X_valid = X.loc[valid_indices]
    y_valid = y.loc[valid_indices]
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    # Check for infinite or NaN values in the datasets
    for dataset_name, dataset in [('X_train', X_train), ('X_valid', X_valid), ('X_test', X_test)]:
        inf_values = np.isinf(dataset.values).sum()
        nan_values = np.isnan(dataset.values).sum()
        print(f'Infinite values in {dataset_name}:', inf_values)
        print(f'NaN values in {dataset_name}:', nan_values)
        if inf_values > 0 or nan_values > 0:
            raise ValueError(f"{dataset_name} contains infinite or NaN values.")

    # Initialize and train the Random Forest Regressor
    regressor = RandomForestRegressor(random_state=84, n_estimators=len(X.columns)*20)
    regressor.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_valid_pred = regressor.predict(X_valid)
    mse_valid = mean_squared_error(y_valid, y_valid_pred)
    print('Validation Mean Squared Error:', mse_valid)

    # Evaluate the model on the test set
    y_test_pred = regressor.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print('Test Mean Squared Error:', mse_test)


    # # --- Improved Visualization ---
    # plt.figure(figsize=(14, 7))

    # # Plot actual stock prices
    # plt.plot(y_train.index, y_train, label='Actual Stock Price - Training', marker='o', color='blue')
    # plt.plot(y_valid.index, y_valid, label='Actual Stock Price - Validation', marker='o', color='green')
    # plt.plot(y_test.index, y_test, label='Actual Stock Price - Test', marker='o', color='red')

    # # Plot predicted stock prices on validation and test sets
    # plt.plot(y_valid.index, y_valid_pred, label='Predicted Stock Price - Validation', marker='x', linestyle='--', color='lime')
    # plt.plot(y_test.index, y_test_pred, label='Predicted Stock Price - Test', marker='x', linestyle='--', color='orange')

    # plt.title('Random Forest Regression - Actual vs Predicted Stock Price')
    # plt.xlabel('Date')
    # plt.ylabel('Stock Price')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # Print actual and predicted values for the test set
    print("Actual Stock Prices (Test Set):")
    print(y_test)
    print("\nPredicted Stock Prices (Test Set):")
    print(pd.DataFrame(y_test_pred, index=y_test.index, columns=['Predicted StockPrice']))

    return (mse_valid, mse_test), (y_train, y_valid, y_test), (X_train, X_valid, X_test)




# Define the ticker symbol
tick = 'XOM'

# Load your existing data
df_tick = pd.read_csv(tick + '_with_price.csv', sep=';')
df_tick.set_index('end', inplace=True)


(mse_valid, mse_test), (y_train, y_valid, y_test), (X_train, X_valid, X_test) = predict_rf(df_tick=df_tick)


print(y_test)