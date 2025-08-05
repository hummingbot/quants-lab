import os
import logging
from datetime import datetime
from nexus.__version__ import __version__

def setup_logging(strategy_name, log_level=logging.INFO, verbose=False, log_to_file=True):
    """
    Configures logging for the trading framework.

    Parameters:
    - strategy_name
    - log_level
    - verbose (enable console logging)
    - log_to_file
    """
    logger = logging.getLogger()
    logger.setLevel(log_level) 

    # Create formatter
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    
    # Create log directory for the strategy
    log_dir = os.path.join('log', strategy_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, strategy_name+'.log')

    # Check if the file exists and delete it if it does
    if os.path.exists(log_file):
        os.remove(log_file)

    # Create file handler
    if log_to_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Create console handler
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)   

    logger.info(f"NEXUS v{__version__} {datetime.now().strftime("%a %d-%m-%y %H:%M:%S")}") 
    