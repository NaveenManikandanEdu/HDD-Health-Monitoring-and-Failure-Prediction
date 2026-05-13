# controller/main.py

from controller.core.monitor import start_monitor
from controller.core.processor import (
    check_and_trigger_batch,
    check_training_trigger
)
from controller.core.logger import log


def startup_reconciliation():
    """
    On system startup:
    1. Check CSV staging (7 CSV → preprocessing)
    2. Check processed parquet buffer (7 parquet → training)
    """

    log("System Startup: Checking CSV staging buffer...")
    check_and_trigger_batch()

    log("System Startup: Checking processed parquet buffer...")
    check_training_trigger()


if __name__ == "__main__":
    print("=" * 60)
    print("  HDD PREDICTION CONTROLLER ACTIVE")
    print("=" * 60)

    startup_reconciliation()

    try:
        start_monitor()
    except KeyboardInterrupt:
        log("System shut down by user.")