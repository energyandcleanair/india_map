
from logging import getLogger, StreamHandler, Formatter, DEBUG
import sys

logger = getLogger("pm25ml")
logger.setLevel(DEBUG)
console_handler = StreamHandler(sys.stdout)
console_handler.setLevel(DEBUG)
console_handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
