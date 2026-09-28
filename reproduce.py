"""Rebuild the complete retrospective experiment and final inference artifacts."""

import argparse
import subprocess
import sys
from pathlib import Path


COMMANDS = [
    ["forecast.py", "--data", "okm_augumented_2021.csv", "--output", "outputs"],
    ["audit.py"], ["advanced_forecast.py"], ["deploy.py", "train"], ["deploy.py", "predict"],
    ["research.py", "--stage", "9"], ["research.py", "--stage", "10"],
    ["research.py", "--stage", "11"], ["risk.py"], ["research.py", "--stage", "13"],
    ["reliability.py"], ["release.py", "train"], ["release.py", "predict"],
    ["final_report.py"], ["-m", "unittest", "discover", "-s", "tests"],
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print commands without running them")
    args = parser.parse_args()
    for command in COMMANDS:
        print(" ".join([sys.executable, *command]), flush=True)
        if not args.list:
            subprocess.run([sys.executable, *command], cwd=Path(__file__).resolve().parent, check=True)
