"""
Storage layer for features and signals using MongoDB.
"""
import logging
from typing import List, Optional, Literal
from datetime import datetime, timedelta, timezone

from core.database_manager import db_manager
from core.features.models import Feature, Signal

logger = logging.getLogger(__name__)


class FeatureStorage:
    """MongoDB storage manager for features and signals"""

    def __init__(self):
        """Initialize feature storage with MongoDB."""
        self.mongo_client = None
        self.features_collection = "features"
        self.signals_collection = "signals"

    async def connect(self):
        """Connect to MongoDB"""
        self.mongo_client = await db_manager.get_mongodb_client()
        if self.mongo_client:
            logger.info("Connected to MongoDB for feature storage")
        else:
            raise RuntimeError("Failed to connect to MongoDB. Ensure MONGO_URI is set in environment variables.")

    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.mongo_client:
            await self.mongo_client.disconnect()

    # Feature operations
    async def save_feature(self, feature: Feature):
        """Save a single feature"""
        await self.save_features([feature])

    async def save_features(self, features: List[Feature]):
        """Save multiple features to MongoDB"""
        if not features:
            return

        documents = [f.to_mongo() for f in features]
        await self.mongo_client.insert_documents(
            collection_name=self.features_collection,
            documents=documents,
            index=["feature_name", "trading_pair", "connector_name", "timestamp"]
        )
        logger.info(f"Saved {len(features)} features to MongoDB")

    async def get_features(
        self,
        feature_name: Optional[str] = None,
        trading_pair: Optional[str] = None,
        connector_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Feature]:
        """
        Retrieve features with optional filters.

        Args:
            feature_name: Filter by feature name
            trading_pair: Filter by trading pair
            connector_name: Filter by connector (None for aggregated features)
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum number of results

        Returns:
            List of Feature objects
        """
        query = {}
        if feature_name:
            query['feature_name'] = feature_name
        if trading_pair:
            query['trading_pair'] = trading_pair
        if connector_name is not None:
            query['connector_name'] = connector_name
        if start_time or end_time:
            query['timestamp'] = {}
            if start_time:
                query['timestamp']['$gte'] = start_time
            if end_time:
                query['timestamp']['$lte'] = end_time

        docs = await self.mongo_client.get_documents(
            collection_name=self.features_collection,
            query=query,
            limit=limit
        )
        return [Feature.from_mongo(doc) for doc in docs]

    # Signal operations
    async def save_signal(self, signal: Signal):
        """Save a single signal"""
        await self.save_signals([signal])

    async def save_signals(self, signals: List[Signal]):
        """Save multiple signals to MongoDB"""
        if not signals:
            return

        documents = [s.to_mongo() for s in signals]
        await self.mongo_client.insert_documents(
            collection_name=self.signals_collection,
            documents=documents,
            index=["signal_name", "trading_pair", "category", "timestamp"]
        )
        logger.info(f"Saved {len(signals)} signals to MongoDB")

    async def get_signals(
        self,
        signal_name: Optional[str] = None,
        trading_pair: Optional[str] = None,
        category: Optional[Literal['tf', 'mr', 'pt']] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Signal]:
        """
        Retrieve signals with optional filters.

        Args:
            signal_name: Filter by signal name
            trading_pair: Filter by trading pair
            category: Filter by category (tf, mr, pt)
            min_value: Minimum signal value (-1 to 1)
            max_value: Maximum signal value (-1 to 1)
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum number of results

        Returns:
            List of Signal objects
        """
        query = {}
        if signal_name:
            query['signal_name'] = signal_name
        if trading_pair:
            query['trading_pair'] = trading_pair
        if category:
            query['category'] = category
        if min_value is not None or max_value is not None:
            query['value'] = {}
            if min_value is not None:
                query['value']['$gte'] = min_value
            if max_value is not None:
                query['value']['$lte'] = max_value
        if start_time or end_time:
            query['timestamp'] = {}
            if start_time:
                query['timestamp']['$gte'] = start_time
            if end_time:
                query['timestamp']['$lte'] = end_time

        docs = await self.mongo_client.get_documents(
            collection_name=self.signals_collection,
            query=query,
            limit=limit
        )
        return [Signal.from_mongo(doc) for doc in docs]

    async def delete_old_features(self, days: int = 30):
        """Delete features older than specified days"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        await self.mongo_client.delete_documents(
            collection_name=self.features_collection,
            query={'timestamp': {'$lt': cutoff_date}}
        )
        logger.info(f"Deleted features older than {days} days")

    async def delete_old_signals(self, days: int = 30):
        """Delete signals older than specified days"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        await self.mongo_client.delete_documents(
            collection_name=self.signals_collection,
            query={'timestamp': {'$lt': cutoff_date}}
        )
        logger.info(f"Deleted signals older than {days} days")
