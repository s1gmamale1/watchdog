import logging
import os
import sys
import traceback
import time
from datetime import datetime

from steps.memreduct import run as mem_run
from steps.rdp import run as rdp_run
from utils import load_yaml, exe_dir


def setup_boot_logger() -> logging.Logger:
    logs_dir = os.path.join(exe_dir(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"boot_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("boot")
    log.info("Log: %s", log_path)
    return log


def main():
    log = setup_boot_logger()
    log.info("=== BOOT STARTED ===")

    try:
        cfg = load_yaml("config/regions.yaml")
        log.info("Config loaded")
    except Exception:
        log.exception("Failed to load regions.yaml")
        raise

    paths = cfg.get("paths", {})

    # Step 1: MemReduct
    try:
        log.info("Step 1: MemReduct")
        mem_run({"exe_path": paths["memreduct_exe"][0]})
        log.info("MemReduct done")
    except Exception:
        log.exception("MemReduct step failed")
        raise

    time.sleep(2)

    # Step 2: RDP
    try:
        log.info("Step 2: RDPClient")
        print("2️⃣  Starting RDPClient...")
        rdp_run()
        print("   ✅ RDPClient complete\n")
        log.info("RDPClient done")
    except Exception:
        log.exception("RDP step failed")
        raise

    log.info("=== BOOT COMPLETE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # main() already logged the traceback via log.exception().
        # Fallback: if logger setup itself failed, write a raw crash file.
        try:
            logs_dir = os.path.join(exe_dir(), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            crash_path = os.path.join(logs_dir, f"boot_crash_{ts}.log")
            with open(crash_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)
