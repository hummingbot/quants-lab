from __future__ import annotations
from numpy.typing import NDArray  
import numpy as np
import random
import math

def push(arr: NDArray[np.floating], value: float) -> None:
    """
    Shifts elements to the right in np array and insert new value at index [0]
    Avoid  extra memory allocation for the duration of the function call
    Parameters:
    - arr : A numpy array.
    - value : The new value to insert at index 0.
    """ 
    arr[1:] = arr[:-1]      # shift elements right
    arr[0] = value          # insert new value

def smoothF(period) -> float:
    """
    Converts a smoothing factor to an alpha value.

    Parameters:
    - period (int): The smoothing period.

    Returns:
    - float: The calculated alpha value.
    """
    return 2.0 / (period + 1)

def genSine(Period1: float, Period2: float, numBars: int, lookBack: int, nBar: int) -> float:
    """
    Sine wave generator with dynamically changing period, based on Zorro's genSine.

    Parameters:
    - Period1: Starting period of the wave.
    - Period2: Ending period of the wave. If Period2 == 0, uses Period1.
    - numBars: Total number of bars in the series.
    - lookBack: Look-back period.
    - nBar: Current bar number.

    Returns:
    - float: The sine wave value for the current bar.
    """
    if Period1 <= 1.0:
        Period1 = 1.0
    if Period2 == 0.0:
        Period2 = Period1

    k = (Period2 - Period1) / (numBars - lookBack)
    Phase = nBar - lookBack

    if k == 0.0 or Phase < 0:  # Constant frequency
        return 0.5 + 0.5 * math.sin((2.0 * math.pi * Phase) / Period1)
    else:  # Hyperbolic chirp
        return 0.5 + 0.5 * math.sin(2.0 * math.pi * math.log(1 + k * Phase / Period1) / k)

def genSquare(Period1, Period2, numBars, lookBack, nBar) -> float:
    """
    Square wave generator based on Zorro's genSquare.

    Parameters:
    - Period1: Starting period of the wave.
    - Period2: Ending period of the wave.
    - numBars: Total number of bars in the series.
    - lookBack: Look-back period.
    - nBar: Current bar number.

    Returns:
    - float: The square wave value for the current bar (0 or 1).
    """
    return 1.0 if genSine(Period1, Period2, numBars, lookBack, nBar) >= 0.5 else 0.0

def genNoise() -> float:
    """
    Noise generator, producing random noise between 0 and 1.

    Returns:
    - float: Random noise value.
    """
    return random.uniform(0.0, 1.0)
