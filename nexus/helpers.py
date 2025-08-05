def push(arr, value):
    """
    Shifts elements to the right in np array and insert new value at index [0]

    Parameters:
    - arr : A numpy array.
    - value : The new value to insert at index 0.
    """
    # avoid  extra memory allocation for the duration of the function call
    arr[1:] = arr[:-1]      # shift elements right
    arr[0] = value          # insert new value  # insert new value

def get_price(bar):
    """
    Get the appropriate price field
    """
    if 'Close' in bar:
        return bar['Close']
    elif 'Price' in bar:
        return bar['Price']
    else:
        return bar.get('LAST', None)

def get_ohlc(bar):
    """
    Get the Open, High, Low, Close values from the bar.

    Parameters:
    - bar (dict): A dictionary containing price fields.

    Returns:
    - tuple: A tuple containing (open, high, low, close) values.

    Raises:
    - KeyError: If any of 'Open', 'High', 'Low', or 'Close' fields are missing.
    """
    try:
        open = bar['Open']
        high = bar['High']
        low = bar['Low']
        close = bar['Close']
    except KeyError as e:
        raise KeyError(f"Missing required field in the bar data: {e}")
    
    return open, high, low, close

def get_twap(bar):
    """
    Get the Open, High, Low, Close values from the bar.

    Parameters:
    - bar (dict): A dictionary containing price fields.

    Returns:
    - tuple: A tuple containing (open, high, low, close) values.

    Raises:
    - KeyError: If any of 'Open', 'High', 'Low', or 'Close' fields are missing.
    """
    try:
        twap = bar['TWAP']
    except KeyError as e:
        raise KeyError(f"Missing required field in the bar data: {e}")
    
    return twap

def cross_over(time_series_1, time_series_2) -> bool:
    """
    Determines if series1 crosses over series2 between the previous and current data points.

    Parameters:
    - series1 (numpy array or list): The first timeseries.
    - series2 (numpy array or list): The second timeseries.

    Returns:
    - bool: True if a crossover occurred, False otherwise.
    """
    if len(time_series_1) < 2 or len(time_series_2) < 2:
        return False
    return time_series_1[1] < time_series_2[1] and time_series_1[0] > time_series_2[0]

def cross_under(time_series_1, time_series_2) -> bool:
    """
    Determines if series1 crosses under series2 between the previous and current data points.

    Parameters:
    - series1 (numpy array or list): The first timeseries.
    - series2 (numpy array or list): The second timeseries.

    Returns:
    - bool: True if a crossunder occurred, False otherwise.
    """
    if len(time_series_1) < 2 or len(time_series_2) < 2:
        return False
    return time_series_1[1] > time_series_2[1] and time_series_1[0] < time_series_2[0]

def peak(time_series) -> bool:
    """
    Determines if the data series had a maximum (peak) at the previous bar.

    Parameters:
    - time_series (numpy array or list): The timeseries to analyze.

    Returns:
    - bool: True if a peak is detected, False otherwise.
    """
    if len(time_series) < 3:
        return False  # Not enough data to determine a peak
    return (time_series[2] <= time_series[1]) and (time_series[1] > time_series[0]) 

def valley(time_series) -> bool:
    """
    Determines if the data series had a minimum (valley) at the previous bar.

    Parameters:
    - data (numpy array or list): The timeseries to analyze.

    Returns:
    - bool: True if a valley is detected, False otherwise.
    """
    if len(time_series) < 3:
        return False  # Not enough data to determine a valley
    return (time_series[2] >= time_series[1]) and (time_series[1] < time_series[0])
