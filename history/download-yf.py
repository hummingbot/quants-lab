import sys
import os
import pandas as pd
import yfinance as yf

def main():
    # Check if the correct number of arguments have been provided
    if len(sys.argv) != 4:
        print("Usage: py -m yf <ticker> <start_date> <end_date>")
        print("Example: py -m yf AAPL 2000-01-01 2025-01-01")
        sys.exit(1)

    # Parse command-line arguments
    ticker = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    # Download the data
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, multi_level_index=False)

    # If the columns are MultiIndex, keep the OHLCV part and flatten
    if isinstance(data.columns, pd.MultiIndex):
        if 'Price' in data.columns.get_level_values(0):        # v0.2+ layout
            data = data['Price']
        else:                                                  # v0.1 layout
            data = data.xs(ticker, level=0, axis=1)

    # At this point `data.columns` is like ["Close", "High", "Low", ...]
    # ------------------------------------------------------------------ #
    # 3. if the user previously *flattened* with "_" and got Close_AAPL   
    #    tidy them up (harmless if they’re already clean)                 
    data.columns = (
        data.columns
            .str.replace(f"_{ticker}$", "", regex=True)   # drop "_AAPL"
            .str.replace("^Adj Close$", "AdjClose", regex=True)  # optional
    )
    data.columns.name = None

    # Define the directory where you want to save the data
    data_directory = 'history/yf'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Save the OHLC data to a CSV file in the 'data' subfolder
    filename = os.path.join(data_directory, f"{ticker}.csv")
    data.to_csv(filename)

    print(f"Data for {ticker} downloaded and saved to {filename}")

if __name__ == "__main__":
    main()
