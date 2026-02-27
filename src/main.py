from winops import set_dpi_awareness
set_dpi_awareness()

import sys
import os

print("frozen:", getattr(sys, "frozen", False))
print("_MEIPASS:", getattr(sys, "_MEIPASS", None))
print("exe:", sys.executable)
print("-" * 40)

# normal imports
from watchdog import run_watchdog
from utils import setup_logger

def main():
    logger = setup_logger()
    run_watchdog()

if __name__ == "__main__":
    main()
