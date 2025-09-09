"""
Base classes for notification services.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class NotificationMessage:
    """Structure for notification messages."""
    
    title: str
    message: str
    level: str = "info"  # info, warning, error, success
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseNotifier(ABC):
    """Base class for all notification services."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notifier with configuration.
        
        Args:
            config: Configuration dictionary for the notifier
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def is_enabled(self) -> bool:
        """Check if this notifier is enabled."""
        return self.enabled
    
    @abstractmethod
    async def send_notification(self, message: NotificationMessage) -> bool:
        """
        Send a notification message.
        
        Args:
            message: The notification message to send
            
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        pass
    
    def format_message(self, message: NotificationMessage) -> str:
        """
        Format the message for this notification service.
        Default implementation returns the message as-is.
        
        Args:
            message: The notification message to format
            
        Returns:
            str: Formatted message
        """
        if message.title and message.message:
            return f"**{message.title}**\n\n{message.message}"
        elif message.title:
            return message.title
        else:
            return message.message
    
    def _log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        self._logger.error(f"Failed to send notification{' ' + context if context else ''}: {str(error)}")
    
    def _log_success(self, context: str = ""):
        """Log successful notifications."""
        self._logger.info(f"Notification sent successfully{' ' + context if context else ''}")