import requests
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

gold_api_key = os.getenv("GOLD_API_KEY")

def make_gapi_request(date_str):
    api_key = gold_api_key
    symbol = "XAG"
    curr = "USD"
    url = f"https://www.goldapi.io/api/{symbol}/{curr}{date_str}"
    
    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        result = response.json()
        print(result)
        return result
    except requests.exceptions.RequestException as e:
        print("Error:", str(e))

def save_to_sqlite(data, db_path="metal_prices.db"):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert data into the table
    cursor.execute('''
        INSERT OR REPLACE INTO silver_prices (Date, Close)
        VALUES (?, ?)
    ''', (data["Date"], data["Close"]))

    # Commit and close the connection
    conn.commit()
    conn.close()

yesterday = datetime.now() - timedelta(days=1)
date_str = yesterday.strftime("/%Y%m%d")

result = make_gapi_request(date_str)

if result:
    # Format the date as "DD-MM-YYYY"
    formatted_date = datetime.strptime(result["date"], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%d-%m-%Y")
    # Prepare data for saving
    data_to_save = {
        "Date": formatted_date,
        "Close": result["price"]
    }
    save_to_sqlite(data_to_save)
    print(f"Saved data for {formatted_date}")
else:
    print(f"Failed to retrieve data for {date_str}")


