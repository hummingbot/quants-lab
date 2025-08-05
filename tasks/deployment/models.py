from typing import Any

from pydantic import BaseModel


class ConfigCandidate(BaseModel):
    config: dict[str, Any]
    extra_info: dict[str, Any]
    id: str

    @classmethod
    def from_mongo(cls, data):
        """Convert MongoDB document to Pydantic model."""
        data["id"] = str(data["_id"])  # Convert ObjectId to string
        return cls(**data)
