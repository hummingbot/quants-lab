"""
Email notification service.
"""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional
from .base import BaseNotifier, NotificationMessage


class EmailNotifier(BaseNotifier):
    """Email notification service using SMTP."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Email notifier.
        
        Args:
            config: Configuration dictionary containing:
                - smtp_server: SMTP server address
                - smtp_port: SMTP server port
                - username: SMTP username
                - password: SMTP password  
                - from_address: Sender email address
                - to_addresses: List of recipient email addresses
        """
        super().__init__(config)
        self.smtp_server = config.get("smtp_server")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.from_address = config.get("from_address")
        self.to_addresses = config.get("to_addresses", [])
        
        required_fields = ["smtp_server", "username", "password", "from_address", "to_addresses"]
        if not all(getattr(self, field) for field in required_fields):
            self._logger.error("Missing required email configuration fields")
            self.enabled = False
    
    def format_message(self, message: NotificationMessage) -> str:
        """
        Format message for email (HTML format).
        
        Args:
            message: The notification message to format
            
        Returns:
            str: HTML formatted message for email
        """
        # Create HTML email content
        level_colors = {
            "info": "#2196F3",
            "warning": "#FF9800", 
            "error": "#F44336",
            "success": "#4CAF50"
        }
        
        color = level_colors.get(message.level, "#666666")
        
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="border-left: 4px solid {color}; padding-left: 20px; margin: 20px 0;">
                <h2 style="color: {color}; margin-top: 0;">{message.title or 'Notification'}</h2>
                <p style="margin: 10px 0;">{message.message.replace('\n', '<br>')}</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    async def send_notification(self, message: NotificationMessage, 
                               to_addresses: Optional[List[str]] = None) -> bool:
        """
        Send notification via email.
        
        Args:
            message: The notification message to send
            to_addresses: Optional custom recipient list, defaults to configured recipients
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled():
            self._logger.debug("Email notifier is disabled")
            return False
        
        try:
            # Run the blocking SMTP operations in a thread pool
            return await asyncio.get_event_loop().run_in_executor(
                None, self._send_email_sync, message, to_addresses
            )
        except Exception as e:
            self._log_error(e, "via Email")
            return False
    
    def _send_email_sync(self, message: NotificationMessage, 
                        to_addresses: Optional[List[str]] = None) -> bool:
        """Synchronous email sending method."""
        try:
            # Use custom recipients if provided, otherwise use configured ones
            recipients = to_addresses if to_addresses else self.to_addresses
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_address
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"QuantsLab: {message.title or 'Notification'}"
            
            # Add plain text and HTML parts
            text_part = MIMEText(message.message, 'plain')
            html_part = MIMEText(self.format_message(message), 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self._log_success("via Email")
            return True
            
        except Exception as e:
            self._log_error(e, "via Email")
            return False
    
    async def send_attachment(self, message: NotificationMessage, 
                            attachment_path: str, attachment_name: Optional[str] = None,
                            to_addresses: Optional[List[str]] = None) -> bool:
        """
        Send email with attachment.
        
        Args:
            message: The notification message
            attachment_path: Path to the attachment file
            attachment_name: Optional custom name for the attachment
            to_addresses: Optional custom recipient list, defaults to configured recipients
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled():
            return False
            
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._send_email_with_attachment_sync, message, attachment_path, attachment_name, to_addresses
            )
        except Exception as e:
            self._log_error(e, "via Email with attachment")
            return False
    
    def _send_email_with_attachment_sync(self, message: NotificationMessage, 
                                       attachment_path: str, attachment_name: Optional[str] = None,
                                       to_addresses: Optional[List[str]] = None) -> bool:
        """Synchronous email sending with attachment."""
        try:
            # Use custom recipients if provided, otherwise use configured ones
            recipients = to_addresses if to_addresses else self.to_addresses
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"QuantsLab: {message.title or 'Notification'}"
            
            # Add body
            body_part = MIMEText(self.format_message(message), 'html')
            msg.attach(body_part)
            
            # Add attachment
            with open(attachment_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                
            encoders.encode_base64(part)
            
            filename = attachment_name or attachment_path.split('/')[-1]
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            
            msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self._log_success("with attachment via Email")
            return True
            
        except Exception as e:
            self._log_error(e, "via Email with attachment")
            return False