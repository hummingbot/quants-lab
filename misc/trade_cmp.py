import pandas as pd

def comare_trades(file1, file2):
    # Read the CSV files into pandas dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2, dtype={'Profit': str, 'Roll': str}) #nexus

    # Standardize the 'Type' column in df2 to match df1
    df2['Type'] = df2['Type'].map({'Long': 'BUY', 'Short': 'SELL'})

    # Clean up 'Profit' and 'Roll' columns in df2 by removing '+' signs and converting to float
    df2['Profit'] = df2['Profit'].str.replace('+', '', regex=False).astype(float)
    df2['Roll'] = df2['Roll'].str.replace('+', '', regex=False).astype(float)

    # Ensure 'Entry' and 'Exit' columns are of float type
    df1['Entry'] = df1['Entry'].astype(float)
    df1['Exit'] = df1['Exit'].astype(float)
    df2['Entry'] = df2['Entry'].astype(float)
    df2['Exit'] = df2['Exit'].astype(float)

    # Convert date columns to datetime objects
    df1['Open'] = pd.to_datetime(df1['Open'])
    df1['Close'] = pd.to_datetime(df1['Close'])
    df2['Open'] = pd.to_datetime(df2['Open'])
    df2['Close'] = pd.to_datetime(df2['Close'])

    # Compare the total number of rows
    print(f"Number of rows in df1: {len(df1)}")
    print(f"Number of rows in df2: {len(df2)}")

    if len(df1) != len(df2):
        print("Number of rows not equal!")   
        return

    # Define columns to compare
    columns_to_compare = ['Open', 'Close', 'Entry', 'Exit', 'Profit', 'Roll']

    # Compare individual columns
    for col in columns_to_compare:
        print(f"\nComparing column: {col}")
        if df1[col].dtype == 'datetime64[ns]':
            # For datetime columns, check for exact equality
            comparison = df1[col] == df2[col]
            if comparison.all():
                print(f"All values in column '{col}' are equal.")
            else:
                differences = df1.loc[~comparison, col]
                print(f"Differences found in column '{col}':")
                print("Values in df1:")
                print(differences)
                print("Corresponding values in df2:")
                print(df2.loc[~comparison, col])
        else:
            # For numerical columns, check for equality within a tolerance
            differences = abs(df1[col] - df2[col])
            max_diff = differences.max()
            print(f"Maximum difference in column '{col}': {max_diff}")
            significant_diff = differences > 1e-2  # Adjust tolerance as needed
            if significant_diff.any():
                print(f"Significant differences in column '{col}':")
                print("Values in df1:")
                print(df1.loc[significant_diff, col])
                print("Corresponding values in df2:")
                print(df2.loc[significant_diff, col])
                print("Differences:")
                print(differences[significant_diff])
            else:
                print(f"All values in column '{col}' are equal within tolerance.")

if __name__ == "__main__":
    print("Zorro vs Nexus:")
    file1 = 'C:/Users/maxpa/Zorro/Log/testtrades.csv'
    file2 = 'C:/Projects/nexus/log/Alice1a/trades_log.csv'
    comare_trades(file1, file2)