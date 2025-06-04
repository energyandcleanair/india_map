"""The logging configuration for the pm25ml package."""

import sys
from logging import DEBUG, Formatter, Logger, StreamHandler, getLogger

logger: Logger = getLogger("pm25ml")
logger.setLevel(DEBUG)
console_handler = StreamHandler(sys.stdout)
console_handler.setLevel(DEBUG)
console_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)
