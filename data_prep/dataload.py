import sqlite3
import pandas as pd

# Load CSV file into a DataFrame
df = pd.read_csv('training_data\gold-ten-years.csv')

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('metal_prices.db')

# Save the DataFrame to a SQLite table
df.to_sql('gold_prices', conn, if_exists='replace', index=False)

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database and table created successfully.")
