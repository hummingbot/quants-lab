"""
Discord notification service.
"""

import aiohttp
from typing import Dict, Any, Optional
from .base import BaseNotifier, NotificationMessage


class DiscordNotifier(BaseNotifier):
    """Discord notification service using webhooks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Discord notifier.
        
        Args:
            config: Configuration dictionary containing:
                - webhook_url: Discord webhook URL
        """
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
        
        if not self.webhook_url:
            self._logger.error("Discord webhook_url is required")
            self.enabled = False
    
    def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """
        Format message for Discord using embeds.
        
        Args:
            message: The notification message to format
            
        Returns:
            Dict: Discord webhook payload
        """
        # Color based on message level
        level_colors = {
            "info": 0x2196F3,
            "warning": 0xFF9800,
            "error": 0xF44336, 
            "success": 0x4CAF50
        }
        
        color = level_colors.get(message.level, 0x666666)
        
        embed = {
            "title": message.title or "QuantsLab Notification",
            "description": message.message,
            "color": color,
            "timestamp": None  # Discord will use current time
        }
        
        # Add footer
        embed["footer"] = {
            "text": "QuantsLab",
            "icon_url": "https://raw.githubusercontent.com/hummingbot/hummingbot/master/hummingbot/assets/hummingbot_logo.png"
        }
        
        return {
            "embeds": [embed],
            "username": "QuantsLab Bot"
        }
    
    async def send_notification(self, message: NotificationMessage,
                               webhook_urls: Optional[list] = None) -> bool:
        """
        Send notification via Discord webhook.
        
        Args:
            message: The notification message to send
            webhook_urls: Optional list of webhook URLs to send to multiple servers/channels
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled():
            self._logger.debug("Discord notifier is disabled")
            return False
        
        # Use custom webhook URLs if provided, otherwise use configured default
        target_webhooks = webhook_urls if webhook_urls else [self.webhook_url]
        
        try:
            success_count = 0
            total_webhooks = len(target_webhooks)
            
            async with aiohttp.ClientSession() as session:
                for webhook_url in target_webhooks:
                    try:
                        payload = self.format_message(message)
                        
                        async with session.post(webhook_url, json=payload) as response:
                            if response.status in [200, 204]:
                                success_count += 1
                                # Extract server/channel info from webhook URL for logging
                                webhook_id = webhook_url.split('/')[-2] if '/' in webhook_url else "unknown"
                                self._logger.debug(f"Discord message sent successfully to webhook: {webhook_id[:8]}...")
                            else:
                                response_text = await response.text()
                                webhook_info = webhook_url.split('/')[-2][:8] if '/' in webhook_url else "unknown"
                                self._logger.error(f"Discord webhook error {response.status} for {webhook_info}...: {response_text}")
                    except Exception as e:
                        webhook_info = webhook_url.split('/')[-2][:8] if '/' in webhook_url else "unknown"
                        self._logger.error(f"Error sending to Discord webhook {webhook_info}...: {e}")
            
            # Consider success if at least one message was sent
            if success_count > 0:
                self._log_success(f"via Discord ({success_count}/{total_webhooks} webhooks)")
                return True
            else:
                self._logger.error("Failed to send Discord message to all webhooks")
                return False
                        
        except Exception as e:
            self._log_error(e, "via Discord")
            return False