"""
Logger utility for Luna Face Recognition
"""

import logging
from datetime import datetime


class Logger:
    """
    A Logger class for logging messages
    """

    def __init__(self):
        self.logger = logging.getLogger("luna_face_recog")
        self.logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warn(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)