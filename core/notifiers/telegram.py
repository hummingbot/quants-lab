"""
Telegram notification service.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional
from .base import BaseNotifier, NotificationMessage


class TelegramNotifier(BaseNotifier):
    """Telegram notification service using Bot API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Telegram notifier.
        
        Args:
            config: Configuration dictionary containing:
                - bot_token: Telegram bot token
                - chat_id: Telegram chat ID to send messages to
                - parse_mode: Optional parse mode (HTML, Markdown, or None)
                - disable_notification: Optional flag to send silent notifications
        """
        super().__init__(config)
        self.bot_token = config.get("bot_token")
        self.chat_id = config.get("chat_id")
        self.parse_mode = config.get("parse_mode", "HTML")
        self.disable_notification = config.get("disable_notification", False)
        
        if not self.bot_token or not self.chat_id:
            self._logger.error("Telegram bot_token and chat_id are required")
            self.enabled = False
    
    def format_message(self, message: NotificationMessage) -> str:
        """
        Format message for Telegram using HTML or Markdown.
        
        Args:
            message: The notification message to format
            
        Returns:
            str: Formatted message for Telegram
        """
        # Choose emoji based on level
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌",
            "success": "✅"
        }
        
        emoji = emoji_map.get(message.level, "📢")
        
        if self.parse_mode == "HTML":
            formatted_title = f"<b>{message.title}</b>" if message.title else ""
            formatted_message = message.message
            
            if formatted_title and formatted_message:
                return f"{emoji} {formatted_title}\n\n{formatted_message}"
            elif formatted_title:
                return f"{emoji} {formatted_title}"
            else:
                return f"{emoji} {formatted_message}"
                
        elif self.parse_mode == "Markdown":
            formatted_title = f"*{message.title}*" if message.title else ""
            formatted_message = message.message
            
            if formatted_title and formatted_message:
                return f"{emoji} {formatted_title}\n\n{formatted_message}"
            elif formatted_title:
                return f"{emoji} {formatted_title}"
            else:
                return f"{emoji} {formatted_message}"
        else:
            # Plain text
            if message.title and message.message:
                return f"{emoji} {message.title}\n\n{message.message}"
            elif message.title:
                return f"{emoji} {message.title}"
            else:
                return f"{emoji} {message.message}"
    
    async def send_notification(self, message: NotificationMessage) -> bool:
        """
        Send notification via Telegram Bot API.
        
        Args:
            message: The notification message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled():
            self._logger.debug("Telegram notifier is disabled")
            return False
        
        try:
            formatted_message = self.format_message(message)
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_message,
                "disable_notification": self.disable_notification
            }
            
            if self.parse_mode:
                payload["parse_mode"] = self.parse_mode
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self._log_success("via Telegram")
                        return True
                    else:
                        response_text = await response.text()
                        self._logger.error(f"Telegram API error {response.status}: {response_text}")
                        return False
                        
        except Exception as e:
            self._log_error(e, "via Telegram")
            return False
    
    async def send_photo(self, photo_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a photo via Telegram.
        
        Args:
            photo_path: Path to the photo file
            caption: Optional caption for the photo
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled():
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            
            data = aiohttp.FormData()
            data.add_field('chat_id', str(self.chat_id))
            data.add_field('photo', open(photo_path, 'rb'))
            
            if caption:
                data.add_field('caption', caption)
                if self.parse_mode:
                    data.add_field('parse_mode', self.parse_mode)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        self._log_success("photo via Telegram")
                        return True
                    else:
                        response_text = await response.text()
                        self._logger.error(f"Telegram photo API error {response.status}: {response_text}")
                        return False
                        
        except Exception as e:
            self._log_error(e, "sending photo via Telegram")
            return False
    
    async def send_document(self, document_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a document via Telegram.
        
        Args:
            document_path: Path to the document file
            caption: Optional caption for the document
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled():
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"
            
            data = aiohttp.FormData()
            data.add_field('chat_id', str(self.chat_id))
            data.add_field('document', open(document_path, 'rb'))
            
            if caption:
                data.add_field('caption', caption)
                if self.parse_mode:
                    data.add_field('parse_mode', self.parse_mode)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        self._log_success("document via Telegram")
                        return True
                    else:
                        response_text = await response.text()
                        self._logger.error(f"Telegram document API error {response.status}: {response_text}")
                        return False
                        
        except Exception as e:
            self._log_error(e, "sending document via Telegram")
            return False