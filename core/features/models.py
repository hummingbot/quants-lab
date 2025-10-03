"""
Data models for features and signals storage.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional, List, Union, Literal, Dict
from datetime import datetime, timezone


class Feature(BaseModel):
    """Base model for storing individual features"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_name: str  # e.g., "ema_trend", "rsi", "bollinger_bands"
    trading_pair: str
    connector_name: Optional[str] = None  # Optional for aggregated features
    value: Union[float, List[float], Dict[str, float]]  # Can be single value, array, or dict
    info: Optional[Dict[str, Any]] = None  # Additional metadata as dict

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_mongo(self) -> dict[str, Any]:
        """Convert to MongoDB document"""
        doc = self.model_dump()
        doc['timestamp'] = self.timestamp
        return doc

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "Feature":
        """Create from MongoDB document"""
        return cls(**doc)


class Signal(BaseModel):
    """Base model for trading signals"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signal_name: str
    trading_pair: str
    category: Literal['tf', 'mr', 'pt']  # trend_following, mean_reversion, pairs_trading
    value: float  # Between -1 (short) and 1 (long)

    @field_validator('value')
    @classmethod
    def validate_value(cls, v: float) -> float:
        """Ensure value is between -1 and 1"""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f'Signal value must be between -1 and 1, got {v}')
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_mongo(self) -> dict[str, Any]:
        """Convert to MongoDB document"""
        doc = self.model_dump()
        doc['timestamp'] = self.timestamp
        return doc

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "Signal":
        """Create from MongoDB document"""
        return cls(**doc)
