import math

import numpy as np

from nexus.indicators.utils import push


def Normalize(data, period):
    """
    Normalize a value to the -1..+1 range.

    Parameters:
    - data (numpy array):): The data series with the most recent value at index 0.
    - Period (int): The number of periods to consider for normalization.

    Returns:
    - float: The normalized value of the most recent data point.
    """
    period = max(2, period)
    if len(data) < period:
        raise ValueError(f"Not enough data points; required at least {period}")
    vMin = min(data[:period])
    vMax = max(data[:period])
    if vMax > vMin:
        return 2.0 * (data[0] - vMin) / (vMax - vMin) - 1.0
    else:
        return 0.0


def Fisher(value):
    """
    Compute the Fisher Transform of a value.

    Parameters:
    - value (float): The value to transform.

    Returns:
    - float: The Fisher Transform of the value.
    """
    v = max(min(value, 0.998), -0.998)  # Clamp between -0.998 and +0.998
    return 0.5 * math.log((1.0 + v) / (1.0 - v))


def FisherInv(value):
    """
    Compute the Inverse Fisher Transform of a value.

    Parameters:
    - value (float): The value to invert.

    Returns:
    - float: The Inverse Fisher Transform of the value.
    """
    Exp = math.exp(2.0 * value)
    return (Exp - 1.0) / (Exp + 1.0)


class FisherN:
    """
    Class to compute the Normalized Fisher Transform.
    """

    def __init__(self, period):
        """
        Initialize the FisherN instance.
        """
        self.period = max(2, period)
        self.Value = np.zeros(2)
        self.FN = np.zeros(2)
        # Initialize with zeros to avoid IndexError on first update
        # self.Value.update(0.0)
        # self.Value.update(0.0)
        # self.FN.update(0.0)
        # self.FN.update(0.0)

    def update(self, data):
        """
        Update the FisherN with new data and compute the normalized Fisher Transform.

        Parameters:
        - data (numpy array):
        Returns:
        - float: The normalized Fisher Transform value.
        """
        if len(data) < self.period:
            # raise ValueError(f"Not enough data points; required at least {self.period}")
            return data[0]

        # Normalize the data
        normalized = Normalize(data, self.period)

        # Update Value[0] with weighted sum
        value = 0.33 * normalized + 0.67 * self.Value[1]
        # self.Value.update(value)
        push(self.Value, value)

        # Compute Fisher Transform of Value[0]
        fisher_value = Fisher(self.Value[0])

        # Update FN[0] with current Fisher Transform and previous FN[1]
        fn_value = fisher_value + 0.5 * self.FN[1]
        #self.FN.update(fn_value)
        push(self.FN, fn_value)

        return self.FN[0]
