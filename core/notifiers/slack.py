"""
Slack notification service.
"""

import aiohttp
from typing import Dict, Any, Optional
from .base import BaseNotifier, NotificationMessage


class SlackNotifier(BaseNotifier):
    """Slack notification service using webhooks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Slack notifier.
        
        Args:
            config: Configuration dictionary containing:
                - webhook_url: Slack webhook URL
                - channel: Optional channel to post to (overrides webhook default)
        """
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel")
        
        if not self.webhook_url:
            self._logger.error("Slack webhook_url is required")
            self.enabled = False
    
    def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """
        Format message for Slack.
        
        Args:
            message: The notification message to format
            
        Returns:
            Dict: Slack webhook payload
        """
        # Emoji based on message level
        level_emojis = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "success": ":white_check_mark:"
        }
        
        emoji = level_emojis.get(message.level, ":speech_balloon:")
        
        # Create formatted text
        if message.title and message.message:
            text = f"{emoji} *{message.title}*\n{message.message}"
        elif message.title:
            text = f"{emoji} *{message.title}*"
        else:
            text = f"{emoji} {message.message}"
        
        payload = {
            "text": text,
            "username": "QuantsLab Bot",
            "icon_emoji": ":robot_face:"
        }
        
        if self.channel:
            payload["channel"] = self.channel
            
        return payload
    
    async def send_notification(self, message: NotificationMessage,
                               channels: Optional[list] = None,
                               webhook_urls: Optional[list] = None) -> bool:
        """
        Send notification via Slack webhook.
        
        Args:
            message: The notification message to send
            channels: Optional list of channels to override default (e.g., ["#general", "#alerts"])
            webhook_urls: Optional list of webhook URLs to send to multiple workspaces
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled():
            self._logger.debug("Slack notifier is disabled")
            return False
        
        # Determine webhook URLs to use
        target_webhooks = webhook_urls if webhook_urls else [self.webhook_url]
        
        try:
            success_count = 0
            total_targets = 0
            
            async with aiohttp.ClientSession() as session:
                for webhook_url in target_webhooks:
                    # If channels are specified, send to each channel
                    target_channels = channels if channels else [self.channel] if self.channel else [None]
                    
                    for channel in target_channels:
                        total_targets += 1
                        payload = self.format_message(message)
                        
                        # Override channel in payload if specified
                        if channel:
                            payload["channel"] = channel
                            
                        try:
                            async with session.post(webhook_url, json=payload) as response:
                                if response.status == 200:
                                    response_text = await response.text()
                                    if response_text == "ok":
                                        success_count += 1
                                        channel_info = f" to {channel}" if channel else ""
                                        self._logger.debug(f"Slack message sent successfully{channel_info}")
                                    else:
                                        self._logger.error(f"Slack webhook error{' for ' + channel if channel else ''}: {response_text}")
                                else:
                                    response_text = await response.text()
                                    channel_info = f" for {channel}" if channel else ""
                                    self._logger.error(f"Slack webhook HTTP error {response.status}{channel_info}: {response_text}")
                        except Exception as e:
                            channel_info = f" to {channel}" if channel else ""
                            self._logger.error(f"Error sending Slack message{channel_info}: {e}")
            
            # Consider success if at least one message was sent
            if success_count > 0:
                self._log_success(f"via Slack ({success_count}/{total_targets} targets)")
                return True
            else:
                self._logger.error("Failed to send Slack message to all targets")
                return False
                        
        except Exception as e:
            self._log_error(e, "via Slack")
            return False