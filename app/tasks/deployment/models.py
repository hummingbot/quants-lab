from typing import Any, Dict

from pydantic import BaseModel


class ConfigCandidate(BaseModel):
    config: Dict[str, Any]
    extra_info: Dict[str, Any]
    id: str

    @classmethod
    def from_mongo(cls, data):
        """Convert MongoDB document to Pydantic model."""
        data["id"] = str(data["_id"])  # Convert ObjectId to string
        return cls(**data)

