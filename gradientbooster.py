import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define the ticker symbol
tick = 'XOM'

# Load your existing data
df_tick = pd.read_csv(tick + '_with_price.csv', sep=';')
df_tick.set_index('end', inplace=True)

# Calculate percentage change and fill missing values
df_tick = df_tick.pct_change(fill_method=None).iloc[1:]  # Exclude the first NaN value
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

# Feature Scaling (Optional for tree-based models)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_valid_scaled = scaler_X.transform(X_valid)
X_test_scaled = scaler_X.transform(X_test)

# Initialize and train the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
    n_estimators=len(X.columns)*20,
    learning_rate=0.1,
    max_depth=int(np.around(len(X.columns)/6, decimals=0)),
    subsample=0.8,
    random_state=84
)
gbr.fit(X_train_scaled, y_train)

# Evaluate the model on the validation set
y_valid_pred = gbr.predict(X_valid_scaled)
mse_valid = mean_squared_error(y_valid, y_valid_pred)
mae_valid = mean_absolute_error(y_valid, y_valid_pred)
print('Validation Mean Squared Error (% Change):', mse_valid)
print('Validation Mean Absolute Error (% Change):', mae_valid)

# Evaluate the model on the test set
y_test_pred = gbr.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print('Test Mean Squared Error (% Change):', mse_test)
print('Test Mean Absolute Error (% Change):', mae_test)

# --- Visualization ---
plt.figure(figsize=(14, 7))

# Plot actual percentage changes
plt.plot(y_train.index, y_train, label='Actual Stock Price % Change - Training', marker='o', color='blue')
plt.plot(y_valid.index, y_valid, label='Actual Stock Price % Change - Validation', marker='o', color='green')
plt.plot(y_test.index, y_test, label='Actual Stock Price % Change - Test', marker='o', color='red')

# Plot predicted percentage changes on validation and test sets
plt.plot(y_valid.index, y_valid_pred, label='Predicted Stock Price % Change - Validation', marker='x', linestyle='--', color='lime')
plt.plot(y_test.index, y_test_pred, label='Predicted Stock Price % Change - Test', marker='x', linestyle='--', color='orange')

plt.title('Gradient Boosting Regression - Actual vs Predicted Stock Price % Change')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.legend()
plt.grid(True)
plt.show()

# Print actual and predicted values for the test set
print("Actual Stock Price % Changes (Test Set):")
print(y_test)
print("\nPredicted Stock Price % Changes (Test Set):")
print(pd.DataFrame(y_test_pred, index=y_test.index, columns=['Predicted Stock Price % Change']))
