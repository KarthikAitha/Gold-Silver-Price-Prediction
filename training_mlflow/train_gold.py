import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import mlflow
import mlflow.keras

# Load and preprocess data
conn = sqlite3.connect('metal_prices.db')

# Query the data from the silver_prices table
query = "SELECT * FROM gold_prices"
data = pd.read_sql(query, conn)
conn.close()

# Convert the 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select the relevant column (e.g., 'Close' price)
prices = data[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Prepare the dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60  # Use last 60 days to predict the next day
X, Y = create_dataset(scaled_data, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:]
Y_train, Y_test = Y[0:train_size], Y[train_size:]

# Start MLflow experiment
mlflow.set_experiment("Gold Price Prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("look_back", look_back)
    mlflow.log_param("epochs", 1)
    mlflow.log_param("batch_size", 1)
    
    # Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, batch_size=1, epochs=1)
    
    # Log model
    mlflow.keras.log_model(model, "model")

    # Evaluate the model
    train_loss = model.evaluate(X_train, Y_train, verbose=0)
    test_loss = model.evaluate(X_test, Y_test, verbose=0)
    
    # Log metrics
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("test_loss", test_loss)

print("Training loss: ", train_loss)
print("Test loss: ", test_loss)


