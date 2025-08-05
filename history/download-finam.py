import os
import requests

# Function to get the instrument code (em) for a given symbol
def get_instrument_code(symbol):
    # Predefined mapping of symbols to instrument codes
    instrument_codes = {
        'RIH0': 17455,  # RTS Futures March 2020
        # Add more symbols and their 'em' codes if needed
    }
    return instrument_codes.get(symbol)

# Parameters
market = 14  # Futures market code in Finam
code = 'RIH0'  # Symbol for RTS futures (March 2020 contract)
em = get_instrument_code(code)
if em is None:
    print(f"Instrument code for {code} not found. Please update the 'get_instrument_code' function.")
    exit(1)

# Date parameters for January 3, 2020
df = 3       # Day from
mf = 0       # Month from (January is 0)
yf = 2020    # Year from
dt = 3       # Day to
mt = 0       # Month to
yt = 2020    # Year to

p = 1  # Timeframe (1 for tick data)

# File name format
file_name = f'{code}_{yf % 100:02d}{(mf + 1):02d}{df:02d}_{yt % 100:02d}{(mt + 1):02d}{dt:02d}'
file_extension = '.csv'

# Define the directory where you want to save the data
data_directory = 'history/finam'
    
# Create the directory if it doesn't exist
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

filename_full = os.path.join(data_directory, f"{file_name}{file_extension}")
# Additional parameters for the request
params = {
    'market': market,
    'em': em,
    'code': code,
    'apply': 0,
    'df': df,
    'mf': mf,
    'yf': yf,
    'from': f'{df:02d}.{mf + 1:02d}.{yf}',
    'dt': dt,
    'mt': mt,
    'yt': yt,
    'to': f'{dt:02d}.{mt + 1:02d}.{yt}',
    'p': p,
    'f': file_name,
    'e': file_extension,
    'cn': code,
    'dtf': 1,
    'tmf': 3,
    'MSOR': 0,
    'mstime': 'on',
    'mstimever': 1,
    'sep': 1,
    'sep2': 1,
    'datf': 6,
    'at': 1
}

# Construct the URL for data download
url = f'http://export.finam.ru/{file_name}{file_extension}'

# Send the HTTP GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:

    # Save the data to a CSV file
    with open(filename_full, 'wb') as file:
        file.write(response.content)
    print(f'Data successfully saved to {file_name}{file_extension}')
else:
    print(f'Error downloading data: HTTP {response.status_code}')
