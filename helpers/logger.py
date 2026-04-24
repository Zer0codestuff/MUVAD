import logging
import os

class ColoredLogger(logging.Formatter):
    # Color decorators
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, format: str):    
        self.formats = {
            logging.DEBUG: self.grey + format + self.reset,
            logging.INFO: self.green + format + self.reset,
            logging.WARNING: self.yellow + format + self.reset,
            logging.ERROR: self.red + format + self.reset,
            logging.CRITICAL: self.bold_red + format + self.reset
        }

    def format(self, record) -> str:
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def getLogger(name: str) -> logging.Logger:
    """Create logger (or return existing one if already configured)"""

    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if logger already has handlers
    if logger.handlers:
        return logger
    
    format = "%(asctime)s - %(levelname)-8s - %(filename)s:%(lineno)d - %(name)s: %(message)s"

    # Print colored log in console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredLogger(format))
    logger.addHandler(console_handler)

    # Print log in file
    log_file = os.path.join(os.path.dirname(__file__), 'experiment.log')
    file_handler = logging.FileHandler(log_file, encoding="utf-8", errors="replace")
    file_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(file_handler)

    return logger
