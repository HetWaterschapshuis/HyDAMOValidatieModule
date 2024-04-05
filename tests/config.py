"""config for tests"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.joinpath("data")

COVERAGE = {"AHN": DATA_DIR.joinpath(r"dtm")}
