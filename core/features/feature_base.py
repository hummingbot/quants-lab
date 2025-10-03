from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING, Optional
import plotly.graph_objects as go

from pydantic import BaseModel

if TYPE_CHECKING:
    from core.data_structures.candles import Candles
    from core.features.models import Feature, Signal

# Define a type variable that can be any subclass of FeatureConfig
T = TypeVar('T', bound='FeatureConfig')


class FeatureConfig(BaseModel):
    name: str


class FeatureBase(ABC, Generic[T]):
    """
    Base class for all feature calculators.

    Features can:
    1. Calculate indicators on candle data
    2. Create Feature objects for storage
    3. Create Signal objects based on conditions
    4. Visualize themselves on charts
    """

    def __init__(self, feature_config: T):
        self.config = feature_config
        self._calculated_data = None

    @abstractmethod
    def calculate(self, data):
        """
        Calculate the feature and add it to the dataframe.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added feature columns
        """
        ...

    @abstractmethod
    def create_feature(self, candles: "Candles") -> "Feature":
        """
        Create a Feature object from candles data.

        Args:
            candles: Candles object containing OHLCV data and metadata

        Returns:
            Feature object ready for storage
        """
        ...

    def create_signal(self, candles: "Candles", **kwargs) -> Optional["Signal"]:
        """
        Create a Signal object if conditions are met.

        Args:
            candles: Candles object containing OHLCV data and metadata
            **kwargs: Additional parameters for signal generation

        Returns:
            Signal object or None if conditions not met
        """
        return None

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        """
        Add feature visualization to an existing figure.

        Args:
            fig: Plotly figure to add traces to
            candles: Candles object with calculated features
            row: Subplot row number (if using subplots)
            **kwargs: Additional plotting parameters

        Returns:
            Modified figure
        """
        return fig

    def plot(self, candles: "Candles", height: int = 600, width: int = 1200, **kwargs):
        """
        Create a standalone plot with candles and this feature.

        Args:
            candles: Candles object
            height: Figure height
            width: Figure width
            **kwargs: Additional plotting parameters
        """
        # Create figure with candles
        fig = candles.candles_fig(height=height, width=width)

        # Add feature visualization
        fig = self.add_to_fig(fig, candles, **kwargs)

        fig.show()

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"
