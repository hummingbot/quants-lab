"""
Centralized data path management for QuantsLab.

This module provides consistent path access to all data directories
without requiring root_path parameters throughout the codebase.
"""
import os
from pathlib import Path
from typing import Optional


class DataPaths:
    """Centralized data path management."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize data paths.
        
        Args:
            base_path: Optional custom base path. If not provided, 
                      automatically detects project root.
        """
        if base_path:
            self._base_path = Path(base_path).resolve()
        else:
            # Auto-detect project root by finding the directory containing 'core'
            current = Path(__file__).resolve().parent.parent
            while current != current.parent:
                if (current / 'core').exists() and (current / 'app').exists():
                    self._base_path = current
                    break
                current = current.parent
            else:
                # Fallback to parent of core directory
                self._base_path = Path(__file__).resolve().parent.parent
        
        # Define data root
        self._data_root = self._base_path / 'app' / 'data'
        
        # Ensure data directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required data directories exist."""
        directories = [
            self.data_root,
            self.candles_dir,
            self.trades_dir,
            self.oi_dir,
            self.backtesting_dir,
            self.live_bot_databases_dir,
            self.cache_dir,
            self.processed_dir,
            self.raw_dir,
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def base_path(self) -> Path:
        """Get the project base path."""
        return self._base_path
    
    @property
    def data_root(self) -> Path:
        """Get the data root directory (app/data)."""
        return self._data_root
    
    @property
    def candles_dir(self) -> Path:
        """Get the candles directory."""
        return self._data_root / 'cache' / 'candles'
    
    @property
    def trades_dir(self) -> Path:
        """Get the trades cache directory."""
        return self._data_root / 'cache' / 'trades'
    
    @property
    def oi_dir(self) -> Path:
        """Get the open interest cache directory."""
        return self._data_root / 'cache' / 'oi'
    
    @property
    def backtesting_dir(self) -> Path:
        """Get the backtesting directory."""
        return self._data_root / 'processed' / 'backtesting'
    
    @property
    def live_bot_databases_dir(self) -> Path:
        """Get the live bot databases directory."""
        return self._data_root / 'processed' / 'live_bot_databases'
    
    @property
    def cache_dir(self) -> Path:
        """Get the cache directory."""
        return self._data_root / 'cache'
    
    @property
    def processed_dir(self) -> Path:
        """Get the processed data directory."""
        return self._data_root / 'processed'
    
    @property
    def raw_dir(self) -> Path:
        """Get the raw data directory."""
        return self._data_root / 'raw'
    
    def get_candles_path(self, filename: str) -> Path:
        """Get full path for a candles file."""
        return self.candles_dir / filename
    
    def get_backtesting_db_path(self, db_name: str = "optimization_database.db") -> Path:
        """Get full path for a backtesting database."""
        return self.backtesting_dir / db_name
    
    def get_live_bot_db_path(self, db_name: str) -> Path:
        """Get full path for a live bot database."""
        return self.live_bot_databases_dir / db_name
    
    # Legacy compatibility methods
    def get_legacy_path(self, *parts) -> str:
        """
        Get path in legacy format (for backward compatibility).
        
        Args:
            *parts: Path components relative to data root
            
        Returns:
            String path for legacy code compatibility
        """
        if parts and parts[0] == 'data':
            # Remove 'data' prefix as it's already in data_root
            parts = parts[1:]
        
        # Map legacy paths to new structure
        if parts and parts[0] == 'candles':
            return str(self.candles_dir / Path(*parts[1:]))
        elif parts and parts[0] == 'backtesting':
            return str(self.backtesting_dir / Path(*parts[1:]))
        elif parts and parts[0] == 'live_bot_databases':
            return str(self.live_bot_databases_dir / Path(*parts[1:]))
        else:
            return str(self.data_root / Path(*parts))


# Global instance for easy access
data_paths = DataPaths()