import pandas as pd
import re
import os

def parse_log(log_file):
    # Regular expression pattern to match lines starting with '<<'
    pattern = re.compile(r'^<<\s*(.*?),\s*(.*?),\s*(.*?)\s*>>')
    
    data = []
    malformed_lines = 0
    
    with open(log_file, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line.startswith('<<'):
                match = pattern.match(line)
                if match:
                    variable_id = match.group(1).strip()
                    timestamp_str = match.group(2).strip()
                    variable_value_str = match.group(3).strip()
                    
                    # Attempt to parse the timestamp
                    try:
                        timestamp = pd.to_datetime(timestamp_str, errors='raise')
                    except ValueError:
                        print(f"Line {line_number}: Invalid timestamp format '{timestamp_str}'. Skipping line.")
                        print(f"Line {line_number}: {line}")
                        malformed_lines += 1
                        continue
                    
                    # Attempt to convert variable_value to numeric, keep as string if it fails
                    try:
                        variable_value = float(variable_value_str)
                    except ValueError:
                        variable_value = variable_value_str  # Keep as string if not a float
                    
                    data.append({
                        'variable_id': variable_id,
                        'timestamp': timestamp,
                        'variable_value': variable_value
                    })
                else:
                    print(f"Line {line_number}: Line does not match the expected format. Content: {line}")
                    malformed_lines += 1
            else:
                # Ignore lines that do not start with '<<'
                continue
        
    if malformed_lines > 0:
        print(f"Total malformed lines in '{log_file}': {malformed_lines}")
    else:
        print(f"All lines parsed successfully in '{log_file}'.")
        
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure that the DataFrame is not empty
    if df.empty:
        print(f"No valid data found in '{log_file}'.")
    else:
        print(f"Parsed {len(df)} valid lines from '{log_file}'.")
        
    return df

def compare_logs(log_file1, log_file2, tolerance='1s', output_csv='differences.csv'):
    # Parse both logs
    print(f"Parsing log file 1: {log_file1}")
    df1 = parse_log(log_file1)
    print(f"Parsing log file 2: {log_file2}")
    df2 = parse_log(log_file2)
    
    # Check if both DataFrames have data
    if df1.empty or df2.empty:
        print("One or both log files contain no valid data. Exiting comparison.")
        return
    
    # Pivot the data so that variable_id becomes columns
    df1_pivot = df1.pivot_table(index='timestamp', columns='variable_id', values='variable_value')
    df2_pivot = df2.pivot_table(index='timestamp', columns='variable_id', values='variable_value')
    
    # Reset index to prepare for merge_asof
    df1_pivot = df1_pivot.sort_index().reset_index()
    df2_pivot = df2_pivot.sort_index().reset_index()
    
    # Merge the DataFrames on timestamp within a tolerance
    merged_df = pd.merge_asof(
        df1_pivot, df2_pivot, 
        on='timestamp', 
        direction='nearest', 
        tolerance=pd.Timedelta(tolerance), 
        suffixes=('_log1', '_log2')
    )
    
    # Proceed with comparison
    # Identify variables to compare
    variables = set(df1['variable_id']).union(set(df2['variable_id']))
    
    # Initialize a list to hold difference indicators
    difference_columns = []
    
    for var in variables:
        col_log1 = f"{var}_log1" if f"{var}_log1" in merged_df.columns else None
        col_log2 = f"{var}_log2" if f"{var}_log2" in merged_df.columns else None
        diff_col = f"{var}_diff"
        
        if col_log1 and col_log2:
            # Handle numeric and non-numeric comparisons
            if pd.api.types.is_numeric_dtype(merged_df[col_log1]) and pd.api.types.is_numeric_dtype(merged_df[col_log2]):
                # Define a tolerance for numeric comparison, e.g., absolute difference > 1e-5
                merged_df[diff_col] = (merged_df[col_log1] - merged_df[col_log2]).abs() > 1e-5
            else:
                # For non-numeric, use direct comparison
                merged_df[diff_col] = merged_df[col_log1] != merged_df[col_log2]
        else:
            # If the variable is only in one of the logs, mark as difference
            merged_df[diff_col] = True
        difference_columns.append(diff_col)
    
    # Filter rows where any variable differs
    differences = merged_df[merged_df[difference_columns].any(axis=1)]
    
    if not differences.empty:
        print("\nDifferences found:")
        # Display relevant columns
        display_columns = ['timestamp']
        for var in variables:
            if f"{var}_log1" in merged_df.columns:
                display_columns.append(f"{var}_log1")
            if f"{var}_log2" in merged_df.columns:
                display_columns.append(f"{var}_log2")
        display_columns += difference_columns
        print(differences[display_columns])
        # Save differences to CSV
        differences.to_csv(output_csv, index=False)
        print(f"Differences saved to {output_csv}")
    else:
        print("No differences found.")
    
    return differences

if __name__ == "__main__":
    # Correctly encode the full paths using raw strings or double backslashes
    log_zorro = r"C:\Users\maxpa\Zorro\Log\Alice1a-my_test.log" 
    log_nexus = r"C:\Projects\nexus\log\Alice1a\Alice1a.log"
    #log_zorro = r"C:\Users\maxpa\Zorro\Log\SpectralFiltersDemo_test.log"
    #log_nexus = r"C:\Projects\nexus\log\plot_filters\plot_filters.log.txt"

    # Check if files exist
    if not os.path.isfile(log_zorro):
        print(f"Log file not found: {log_zorro}")
    elif not os.path.isfile(log_nexus):
        print(f"Log file not found: {log_nexus}")
    else:
        differences = compare_logs(log_zorro, log_nexus, tolerance='1s')

 