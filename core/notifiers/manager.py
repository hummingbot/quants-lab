"""
Notification Manager for coordinating multiple notifiers.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional

from .base import BaseNotifier, NotificationMessage
from .telegram import TelegramNotifier
from .email import EmailNotifier
from .discord import DiscordNotifier
from .slack import SlackNotifier

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages multiple notification services."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the notification manager.
        
        Args:
            config_dict: Direct configuration dictionary (if None, loads from environment)
        """
        self.notifiers: Dict[str, BaseNotifier] = {}
        self._logger = logging.getLogger(__name__)
        
        # Load configuration from environment or provided dict
        if config_dict:
            config = config_dict
        else:
            config = self._load_env_config()
        
        # Initialize notifiers
        self._initialize_notifiers(config)
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Telegram configuration
        if os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
            config['telegram'] = {
                'enabled': os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true',
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
                'parse_mode': os.getenv('TELEGRAM_PARSE_MODE', 'HTML'),
                'disable_notification': os.getenv('TELEGRAM_DISABLE_NOTIFICATION', 'false').lower() == 'true'
            }
        
        # Email configuration
        if os.getenv('EMAIL_USERNAME') and os.getenv('EMAIL_PASSWORD'):
            config['email'] = {
                'enabled': os.getenv('EMAIL_ENABLED', 'false').lower() == 'true',
                'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '587')),
                'username': os.getenv('EMAIL_USERNAME'),
                'password': os.getenv('EMAIL_PASSWORD'),
                'from_address': os.getenv('EMAIL_FROM', os.getenv('EMAIL_USERNAME')),
                'to_addresses': [addr.strip() for addr in os.getenv('EMAIL_TO', '').split(',') if addr.strip()]
            }
        
        # Discord configuration
        if os.getenv('DISCORD_WEBHOOK_URL'):
            config['discord'] = {
                'enabled': os.getenv('DISCORD_ENABLED', 'false').lower() == 'true',
                'webhook_url': os.getenv('DISCORD_WEBHOOK_URL')
            }
        
        # Slack configuration
        if os.getenv('SLACK_WEBHOOK_URL'):
            config['slack'] = {
                'enabled': os.getenv('SLACK_ENABLED', 'false').lower() == 'true',
                'webhook_url': os.getenv('SLACK_WEBHOOK_URL'),
                'channel': os.getenv('SLACK_CHANNEL', '#general')
            }
        
        return config
    
    def _initialize_notifiers(self, config: Dict[str, Any]):
        """Initialize notifiers based on configuration."""
        notifier_classes = {
            'telegram': TelegramNotifier,
            'email': EmailNotifier,
            'discord': DiscordNotifier,
            'slack': SlackNotifier
        }
        
        for notifier_name, notifier_class in notifier_classes.items():
            notifier_config = config.get(notifier_name, {})
            if notifier_config and notifier_config.get('enabled', False):
                try:
                    notifier = notifier_class(notifier_config)
                    if notifier.is_enabled():
                        self.notifiers[notifier_name] = notifier
                        self._logger.info(f"Initialized {notifier_name} notifier")
                    else:
                        self._logger.warning(f"{notifier_name} notifier is disabled due to configuration issues")
                except Exception as e:
                    self._logger.error(f"Failed to initialize {notifier_name} notifier: {str(e)}")
    
    def get_enabled_notifiers(self) -> List[str]:
        """Get list of enabled notifier names."""
        return list(self.notifiers.keys())
    
    async def send_notification(self, message: NotificationMessage, 
                              notifiers: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Send notification through specified notifiers.
        
        Args:
            message: The notification message to send
            notifiers: Optional list of notifier names to use (defaults to all enabled)
            
        Returns:
            Dict mapping notifier names to success status
        """
        if notifiers is None:
            notifiers = list(self.notifiers.keys())
        
        # Filter to only enabled notifiers
        active_notifiers = {
            name: notifier for name, notifier in self.notifiers.items() 
            if name in notifiers and notifier.is_enabled()
        }
        
        if not active_notifiers:
            self._logger.warning("No active notifiers available")
            return {}
        
        # Send notifications concurrently
        tasks = [
            notifier.send_notification(message) 
            for notifier in active_notifiers.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results back to notifier names
        result_dict = {}
        for (name, notifier), result in zip(active_notifiers.items(), results):
            if isinstance(result, Exception):
                self._logger.error(f"Exception in {name} notifier: {str(result)}")
                result_dict[name] = False
            else:
                result_dict[name] = result
        
        return result_dict
    
    async def send_info(self, title: str, message: str, **kwargs) -> Dict[str, bool]:
        """Send an info notification."""
        notification = NotificationMessage(
            title=title,
            message=message,
            level="info",
            **kwargs
        )
        return await self.send_notification(notification)
    
    async def send_warning(self, title: str, message: str, **kwargs) -> Dict[str, bool]:
        """Send a warning notification."""
        notification = NotificationMessage(
            title=title,
            message=message,
            level="warning",
            **kwargs
        )
        return await self.send_notification(notification)
    
    async def send_error(self, title: str, message: str, **kwargs) -> Dict[str, bool]:
        """Send an error notification."""
        notification = NotificationMessage(
            title=title,
            message=message,
            level="error",
            **kwargs
        )
        return await self.send_notification(notification)
    
    async def send_success(self, title: str, message: str, **kwargs) -> Dict[str, bool]:
        """Send a success notification."""
        notification = NotificationMessage(
            title=title,
            message=message,
            level="success",
            **kwargs
        )
        return await self.send_notification(notification)
    
    def add_notifier(self, name: str, notifier: BaseNotifier):
        """Add a custom notifier."""
        if notifier.is_enabled():
            self.notifiers[name] = notifier
            self._logger.info(f"Added custom notifier: {name}")
    
    def remove_notifier(self, name: str):
        """Remove a notifier."""
        if name in self.notifiers:
            del self.notifiers[name]
            self._logger.info(f"Removed notifier: {name}")
    
    def get_notifier(self, name: str) -> Optional[BaseNotifier]:
        """Get a specific notifier by name."""
        return self.notifiers.get(name)


# Global notification manager instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager instance."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


def set_notification_manager(manager: NotificationManager):
    """Set a custom notification manager instance."""
    global _notification_manager
    _notification_manager = manager