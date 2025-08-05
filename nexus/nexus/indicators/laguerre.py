import numpy as np

from nexus.indicators.utils import smoothF


class LaguerreFilter:
    """
    Implements a Laguerre filter for smoothing price data

    Attributes
    ----------
    alpha : float
        The smoothing factor.
    alpha1 : float
        Complementary smoothing factor (1 - alpha).
    arr : np.ndarray
        Internal state of the Laguerre filter (length 8).
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """
        Parameters
        ----------
        alpha : float, optional
            The smoothing factor (default = 0.5). Values > 1.0 are passed
            through ``smoothF`` first.
        """
        self.alpha = smoothF(alpha) if alpha > 1.0 else alpha
        self.alpha1 = 1.0 - self.alpha
        self.arr = np.zeros(8)
        self.initialized = False

    def update(self, data) -> float:
        """
        Parameters
        ----------
        data: can be scalar or numpy array.
        """
        if isinstance(data, np.ndarray):
            val = float(data.flat[0])  # works for any array shape
        else:
            val = float(data)  # assumes plain scalar

        if not self.initialized:
            self.arr[:] = val
            self.initialized = True
        else:
            self.arr[1:] = self.arr[:-1]  # Shifts elements to the right
        self.arr[0] = self.alpha * val + self.alpha1 * self.arr[1]
        self.arr[2] = (
            -self.alpha1 * self.arr[0] + self.arr[1] + self.alpha1 * self.arr[3]
        )
        self.arr[4] = (
            -self.alpha1 * self.arr[2] + self.arr[3] + self.alpha1 * self.arr[5]
        )
        self.arr[6] = (
            -self.alpha1 * self.arr[4] + self.arr[5] + self.alpha1 * self.arr[7]
        )
        return (self.arr[0] + 2.0 * self.arr[2] + 2.0 * self.arr[4] + self.arr[6]) / 6.0
