# controller/core/logger.py
import logging
import sys
from controller.config import BASE_DIR

LOG_FILE = BASE_DIR / "controller_overall.log"

def setup_logger():
    # Force the root logger to use UTF-8 for the file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # Standard terminal output
        ]
    )

def log(message, level="INFO"):
    # Optional: Strip emojis if the console still complains
    # safe_message = message.encode('ascii', 'ignore').decode('ascii')
    
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)

# Initialize on import
setup_logger()