import requests
import pandas as pd
from datetime import datetime, timedelta

def make_gapi_request(start_date, end_date):
    api_key = "goldapi-vbiimslw57nfnu-io"
    symbol = "XAG"
    curr = "INR"

    # Convert start_date and end_date strings to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Create an empty list to store the data
    data_list = []

    # Iterate over the dates using a loop
    while start_date <= end_date:
        date_str = start_date.strftime("%Y-%m-%d")
        url = f"https://www.goldapi.io/api/{symbol}/{curr}/{date_str}"

        headers = {
            "x-access-token": api_key,
            "Content-Type": "application/json"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            result = response.json()

            # Check if the result is an error message
            if "error" in result and result["error"] == "No data available for this date or pair.":
                print(f"Ignoring date {date_str}")
            else:
                # Append the data to the list
                data_list.append(result)
        except requests.exceptions.RequestException as e:
            print("Error:", str(e))

        # Increment the date for the next iteration
        start_date += timedelta(days=1)

    # Convert the list of data into a DataFrame
    df = pd.DataFrame(data_list)

    return df

# Example usage
start_date = "2012-01-01"
end_date = "2024-05-10"
df = make_gapi_request(start_date, end_date)
print(df)
