from typing import Dict, Any

from pydantic import BaseModel


class TradingSession(BaseModel):
    session_id: str
    controller_config: dict
    db_name: str
    performance_metrics: Dict[str, Any]
    start_timestamp: float = None
    end_timestamp: float = None
