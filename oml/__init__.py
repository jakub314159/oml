import logging
import os

logger = logging.getLogger("oml")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
