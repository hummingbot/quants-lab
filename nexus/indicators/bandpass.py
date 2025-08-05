import numpy as np
import math

class BandPass:
    """
    BandPass filter implementation.

    This filter lets only the cycles close to the given time period pass through.
    The Delta value (0.05 .. 1) determines the filter width; at Delta = 0.1,
    cycles outside a 30% range centered at the time period are attenuated with > 3 dB.

    Attributes:
        a (float): Filter coefficient.
        b (float): Filter coefficient.
        output (list of float): Stores the last three output values.
        _isInit (bool): Indicates if the filter has been initialized.
    """

    def __init__(self, period=None, Delta=None):
        """
        Initializes the BandPass filter.

        Args:
            period (int, optional): The central period of the filter.
            Delta (float, optional): Determines the filter width.
        """
        self.a = 0.0
        self.b = 0.0
        self.output = np.zeros(3)
        self.initialized = False

        if period is not None and Delta is not None:
            self.Set(period, Delta)

    def Set(self, period, Delta):
        """
        Sets the filter parameters and calculates coefficients.

        Args:
            period (int): The central period of the filter.
            Delta (float): Determines the filter width.
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")
        if not (0.05 <= Delta <= 1):
            raise ValueError("Delta must be between 0.05 and 1.")

        K = math.cos(4 * math.pi * Delta / period)
        denominator = K * K
        if (1 / denominator - 1) < 0:
            raise ValueError("Invalid parameters leading to negative square root.")

        self.a = (1 / K) - math.sqrt((1 / (K * K)) - 1)
        self.b = math.cos(2 * math.pi / period)
        self.initialized = False  # Reset initialization flag upon setting new parameters

    def update(self, data) -> float:
        """
        Applies the bandpass filter to the incoming data points.

        Args:
            - data (numpy array): A list containing at least three data points.
            
        Returns:
            The filtered output.
        """
        if not self.initialized:
            if len(data) < 3:
                raise ValueError("At least three data points are required for initialization.")
            self.output[:3] = np.copy(data[:3])
            self.initialized = True

        self.output[2] = self.output[1]
        self.output[1] = self.output[0]
        self.output[0] = ((1 + self.a) / 2) * (data[0] - data[2]) + \
                         self.b * (1 + self.a) * self.output[1] - \
                         self.a * self.output[2]
        return self.output[0]
