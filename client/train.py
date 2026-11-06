import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import flwr as fl
import numpy as np

from shared.client import HSIClient
from shared.data import mad_normalize_hsi, split_train_val


DATA_FILE = os.environ["DATA_FILE"]
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "server:8080")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.15"))
SEED = int(os.environ.get("SEED", "42"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "15"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "5"))
LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")


def _setup_client_logging() -> None:
    logs_dir = Path(LOG_DIR)
    if not logs_dir.is_absolute():
        logs_dir = Path("/app") / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"client_{CLIENT_ID}_{ts}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info("Client log file initialized at %s", log_path)

_setup_client_logging()

x_raw = np.load(DATA_FILE)
x_data = mad_normalize_hsi(x_raw)
idxs = np.arange(len(x_data))
train_idxs, val_idxs = split_train_val(idxs, val_ratio=VAL_RATIO, seed=SEED + CLIENT_ID)

flower_client = HSIClient(
    cid=CLIENT_ID,
    x_data=x_data,
    split=(train_idxs, val_idxs),
    batch_size=BATCH_SIZE,
    seed=SEED,
)

for attempt in range(1, MAX_RETRIES + 1):
    try:
        fl.client.start_client(
            server_address=SERVER_ADDRESS,
            client=flower_client.to_client(),
        )
        break
    except Exception:
        if attempt == MAX_RETRIES:
            raise
        time.sleep(RETRY_DELAY)
