import numpy as np
import pandas as pd
from flask import Flask, render_template
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import plotly.io as pio
import sqlite3

app = Flask(__name__)

# Load the model
silver_model = load_model('models\silver_price_lstm_model.h5')
gold_model = load_model('models\silver_price_lstm_model.h5')

# Load data from the database
conn = sqlite3.connect('metal_prices.db')
query_silver = "SELECT * FROM silver_prices"
query_gold = "SELECT * FROM gold_prices"
data_silver = pd.read_sql(query_silver, conn)
data_gold = pd.read_sql(query_gold, conn)
conn.close()

# Convert the 'Date' column to datetime and set it as index
data_silver['Date'] = pd.to_datetime(data_silver['Date'])
data_silver.set_index('Date', inplace=True)

data_gold['Date'] = pd.to_datetime(data_gold['Date'])
data_gold.set_index('Date', inplace=True)

# Select the relevant column (e.g., 'Close' price)
silver_prices = data_silver[['Close']]
gold_prices = data_gold[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(silver_prices)
scaled_data = scaler.fit_transform(gold_prices)

# Prepare the dataset for LSTM
look_back = 60  # Use last 60 days to predict the next day

# Predict the next five valid days' prices (skip weekends)
next_days = 5
last_days = scaled_data[-look_back:]
silver_predictions = []
gold_predictions = []
future_dates = []

current_date = silver_prices.index[-1] + pd.Timedelta(days=1)
while len(silver_predictions) < next_days:
    if current_date.weekday() < 5:  # Monday to Friday are 0-4
        last_days = np.reshape(last_days, (1, look_back, 1))
        next_day_prediction = silver_model.predict(last_days)
        silver_predictions.append(next_day_prediction[0][0])
        future_dates.append(current_date)
        next_day_prediction = np.reshape(next_day_prediction, (1, 1, 1))
        last_days = np.append(last_days[:, 1:, :], next_day_prediction, axis=1)
    current_date += pd.Timedelta(days=1)

# Invert the predictions
silver_predictions = scaler.inverse_transform(np.array(silver_predictions).reshape(-1, 1))

current_date = gold_prices.index[-1] + pd.Timedelta(days=1)
while len(gold_predictions) < next_days:
    if current_date.weekday() < 5:  # Monday to Friday are 0-4
        last_days = np.reshape(last_days, (1, look_back, 1))
        next_day_prediction = silver_model.predict(last_days)
        gold_predictions.append(next_day_prediction[0][0])
        future_dates.append(current_date)
        next_day_prediction = np.reshape(next_day_prediction, (1, 1, 1))
        last_days = np.append(last_days[:, 1:, :], next_day_prediction, axis=1)
    current_date += pd.Timedelta(days=1)

# Invert the predictions
gold_predictions = scaler.inverse_transform(np.array(gold_predictions).reshape(-1, 1))

@app.route('/')
def index():
    # Get the last 60 days prices for silver and gold
    last_60_days_silver = scaler.inverse_transform(scaled_data[-look_back:])
    last_60_days_silver_dates = silver_prices.index[-look_back:]
    last_60_days_gold = scaler.inverse_transform(scaled_data[-look_back:])
    last_60_days_gold_dates = gold_prices.index[-look_back:]

    # Combine future dates with their respective predictions for silver and gold
    silver_predictions_with_dates = list(zip([date.strftime('%Y-%m-%d') for date in future_dates], silver_predictions.flatten()))
    gold_predictions_with_dates = list(zip([date.strftime('%Y-%m-%d') for date in future_dates], gold_predictions.flatten()))

    # Plot the chart for silver
    fig_silver = go.Figure()
    fig_silver.add_trace(go.Scatter(x=last_60_days_silver_dates, y=last_60_days_silver.flatten(), mode='lines', name='Last 60 Days Prices'))
    fig_silver.add_trace(go.Scatter(x=future_dates, y=silver_predictions.flatten(), mode='lines', name='Next 5 Days Predictions'))
    fig_silver.update_layout(
        title='Silver Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    plot_html_silver = pio.to_html(fig_silver, full_html=False)

    # Plot the chart for gold
    fig_gold = go.Figure()
    fig_gold.add_trace(go.Scatter(x=last_60_days_gold_dates, y=last_60_days_gold.flatten(), mode='lines', name='Last 60 Days Prices'))
    fig_gold.add_trace(go.Scatter(x=future_dates, y=gold_predictions.flatten(), mode='lines', name='Next 5 Days Predictions'))
    fig_gold.update_layout(
        title='Gold Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    plot_html_gold = pio.to_html(fig_gold, full_html=False)

    return render_template('index.html', plot_html_silver=plot_html_silver, plot_html_gold=plot_html_gold, silver_predictions=silver_predictions_with_dates, gold_predictions=gold_predictions_with_dates)


if __name__ == '__main__':
    app.run(debug=True)
