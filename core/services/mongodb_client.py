import math
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta


class MongoClient:
    def __init__(
            self,
            uri: str = None,
            database: str = "mongodb",
    ):
        self.client = None
        self.db = None

        # Connection parameters with env fallbacks
        self.uri = uri
        self.database = database

    async def connect(self):
        """Connect to MongoDB using provided or environment variables."""
        try:
            self.client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client[self.database]
            await self.db.command('ping')
            logging.info(f"Successfully connected to MongoDB")

        except Exception as e:
            print(f"Failed to connect to MongoDB: {str(e)}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB")

    async def create_database(self, db_name: str):
        """Create a new database."""
        self.client[db_name]  # MongoDB creates a database automatically when a collection is added
        logging.info(f"Database {db_name} is now available.")

    async def delete_database(self, db_name: str):
        """Delete a database."""
        self.client.drop_database(db_name)
        logging.info(f"Database {db_name} deleted.")

    async def create_collection(self, collection_name: str, db_name: Optional[str] = None):
        """Create a collection in a given database."""
        db = self.client[db_name] if db_name else self.db
        await db.create_collection(collection_name)
        logging.info(f"Collection {collection_name} created in {db_name or self.db.name}.")

    async def delete_collection(self, collection_name: str, db_name: Optional[str] = None):
        """Delete a collection from a given database."""
        db = self.client[db_name] if db_name else self.db
        await db[collection_name].drop()
        logging.info(f"Collection {collection_name} deleted from {db_name or self.db.name}.")

    async def insert_documents(self, collection_name: str, documents: Union[Dict[str, Any], List[Dict[str, Any]]],
                               db_name: Optional[str] = None, index: List[str] = []):
        """Insert one or multiple documents into a specified collection."""
        db = self.client[db_name] if db_name else self.db
        collection = db[collection_name]

        if isinstance(documents, dict):
            documents = [documents]

        try:
            if index:
                await collection.create_index(index)
            result = await collection.insert_many(documents)
            logging.info(f"Inserted {len(result.inserted_ids)} documents into {collection_name}.")
        except Exception as e:
            logging.error(f"Error inserting documents into {collection_name}: {str(e)}")
            raise

    async def get_documents(self, collection_name: str, query: Dict[str, Any] = None, db_name: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve documents from a collection with an optional query."""
        db = self.client[db_name] if db_name else self.db
        collection = db[collection_name]
        query = query or {}

        try:
            cursor = collection.find(query).sort("timestamp", -1)
            if limit:
                cursor = cursor.limit(limit)
            documents = await cursor.to_list(length=None)
            logging.info(f"Retrieved {len(documents)} documents from {collection_name}.")
            return documents
        except Exception as e:
            logging.error(f"Error retrieving documents from {collection_name}: {str(e)}")
            raise

    async def delete_documents(self, collection_name: str, query: Dict[str, Any], db_name: Optional[str] = None):
        """Delete documents matching a query from a collection."""
        db = self.client[db_name] if db_name else self.db
        collection = db[collection_name]
        try:
            result = await collection.delete_many(query)
            logging.info(f"Deleted {result.deleted_count} documents from {collection_name}.")
        except Exception as e:
            logging.error(f"Error deleting documents from {collection_name}: {str(e)}")
            raise
