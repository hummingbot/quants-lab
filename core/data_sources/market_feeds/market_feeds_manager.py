import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Type, Any, Optional

from .connector_base import ConnectorBase
from .trades_feed_base import TradesFeedBase
from .oi_feed_base import OIFeedBase


class MarketFeedsManager:
    """
    Automatically discovers and manages market feed connectors and their available feed types.
    Scans the market_feeds directory structure to find connectors and feeds.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._connectors: Dict[str, Type[ConnectorBase]] = {}
        self._feeds: Dict[str, Dict[str, Type]] = {}  # connector_name -> {feed_type -> feed_class}
        self._feed_base_classes = {
            "trades_feed": TradesFeedBase,
            "oi_feed": OIFeedBase,
            # Add more feed types here as they are created
            # "funding_rate_feed": FundingRateFeedBase,
        }
        self._discover_feeds()
    
    def _discover_feeds(self):
        """Discover all available connectors and their feeds."""
        market_feeds_path = Path(__file__).parent
        
        # Scan all subdirectories (connector directories)
        for connector_dir in market_feeds_path.iterdir():
            if not connector_dir.is_dir() or connector_dir.name.startswith('_'):
                continue
                
            connector_name = self._extract_connector_name(connector_dir.name)
            if not connector_name:
                continue
                
            self.logger.debug(f"Scanning connector directory: {connector_dir.name}")
            
            # Find connector base class
            connector_base_class = self._find_connector_base(connector_dir, connector_name)
            if connector_base_class:
                self._connectors[connector_name] = connector_base_class
                self._feeds[connector_name] = {}
                
                # Find all feed classes for this connector
                self._discover_connector_feeds(connector_dir, connector_name)
    
    def _extract_connector_name(self, dir_name: str) -> Optional[str]:
        """Extract connector name from directory name."""
        # Handle patterns like 'binance_perpetual' -> 'binance'
        if '_' in dir_name:
            return dir_name.split('_')[0]
        return dir_name
    
    def _find_connector_base(self, connector_dir: Path, connector_name: str) -> Optional[Type[ConnectorBase]]:
        """Find the connector base class in the connector directory."""
        # Look for files that match the pattern: {connector_name}_*_base.py
        for py_file in connector_dir.glob("*_base.py"):
            if connector_name in py_file.name:
                try:
                    module_path = f"core.data_sources.market_feeds.{connector_dir.name}.{py_file.stem}"
                    module = importlib.import_module(module_path)
                    
                    # Find classes that inherit from ConnectorBase
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, ConnectorBase) and 
                            obj != ConnectorBase and 
                            connector_name.lower() in name.lower()):
                            self.logger.info(f"Found connector base: {name} in {module_path}")
                            return obj
                            
                except Exception as e:
                    self.logger.warning(f"Error importing {module_path}: {e}")
        
        return None
    
    def _discover_connector_feeds(self, connector_dir: Path, connector_name: str):
        """Discover all feed types available for a connector."""
        for feed_type, base_class in self._feed_base_classes.items():
            feed_class = self._find_feed_class(connector_dir, connector_name, feed_type, base_class)
            if feed_class:
                self._feeds[connector_name][feed_type] = feed_class
                self.logger.info(f"Found {feed_type} for {connector_name}: {feed_class.__name__}")
    
    def _find_feed_class(self, connector_dir: Path, connector_name: str, 
                        feed_type: str, base_class: Type) -> Optional[Type]:
        """Find a specific feed class type in the connector directory."""
        # Look for files that contain the feed type name
        feed_type_name = feed_type.replace('_feed', '')
        
        for py_file in connector_dir.glob("*.py"):
            if feed_type_name in py_file.name and not py_file.name.endswith('_base.py'):
                try:
                    module_path = f"core.data_sources.market_feeds.{connector_dir.name}.{py_file.stem}"
                    module = importlib.import_module(module_path)
                    
                    # Find classes that inherit from the specific feed base class
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, base_class) and 
                            obj != base_class and
                            connector_name.lower() in name.lower()):
                            return obj
                            
                except Exception as e:
                    self.logger.warning(f"Error importing {module_path}: {e}")
        
        return None
    
    @property
    def available_connectors(self) -> List[str]:
        """Get list of available connector names."""
        return list(self._connectors.keys())
    
    @property
    def available_feeds(self) -> Dict[str, List[str]]:
        """Get dictionary of connector names and their available feed types."""
        return {connector: list(feeds.keys()) for connector, feeds in self._feeds.items()}
    
    def get_connector(self, connector_name: str, **kwargs) -> ConnectorBase:
        """Create an instance of a connector."""
        if connector_name not in self._connectors:
            raise ValueError(f"Connector '{connector_name}' not found. Available: {self.available_connectors}")
        
        connector_class = self._connectors[connector_name]
        return connector_class(**kwargs)
    
    def get_feed(self, connector_name: str, feed_type: str, connector: Optional[ConnectorBase] = None, **kwargs):
        """Create an instance of a feed for a specific connector."""
        if connector_name not in self._feeds:
            raise ValueError(f"Connector '{connector_name}' not found. Available: {self.available_connectors}")
        
        if feed_type not in self._feeds[connector_name]:
            available_feeds = list(self._feeds[connector_name].keys())
            raise ValueError(f"Feed type '{feed_type}' not available for connector '{connector_name}'. Available: {available_feeds}")
        
        # Create connector if not provided
        if connector is None:
            connector = self.get_connector(connector_name, **kwargs)
        
        feed_class = self._feeds[connector_name][feed_type]
        return feed_class(connector)
    
    def get_connector_with_feeds(self, connector_name: str, feed_types: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Get a connector instance with all or specified feeds attached."""
        if connector_name not in self._feeds:
            raise ValueError(f"Connector '{connector_name}' not found. Available: {self.available_connectors}")
        
        # Create connector
        connector = self.get_connector(connector_name, **kwargs)
        
        # Determine which feeds to create
        if feed_types is None:
            feed_types = list(self._feeds[connector_name].keys())
        
        # Create feeds
        feeds = {}
        for feed_type in feed_types:
            if feed_type in self._feeds[connector_name]:
                feeds[feed_type] = self.get_feed(connector_name, feed_type, connector)
            else:
                self.logger.warning(f"Feed type '{feed_type}' not available for connector '{connector_name}'")
        
        return {
            "connector": connector,
            "feeds": feeds
        }
    
    def has_feed(self, connector_name: str, feed_type: str) -> bool:
        """Check if a specific feed type is available for a connector."""
        return (connector_name in self._feeds and 
                feed_type in self._feeds[connector_name])
    
    def get_feed_info(self) -> Dict[str, Dict[str, str]]:
        """Get detailed information about available feeds."""
        info = {}
        for connector_name, feeds in self._feeds.items():
            info[connector_name] = {}
            for feed_type, feed_class in feeds.items():
                info[connector_name][feed_type] = feed_class.__name__
        return info
    
    def print_available_feeds(self):
        """Print a formatted list of available connectors and feeds."""
        print("Available Market Feeds:")
        print("=" * 50)
        
        if not self._feeds:
            print("No feeds discovered.")
            return
        
        for connector_name, feeds in self._feeds.items():
            print(f"\n{connector_name}:")
            if feeds:
                for feed_type in feeds.keys():
                    print(f"  - {feed_type}")
            else:
                print("  No feeds available")