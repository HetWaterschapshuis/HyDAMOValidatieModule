__author__ = ["Het Waterschapshuis", "D2HYDRO", "HKV", "HydroConsult"]
__copyright__ = "Copyright 2021, HyDAMO ValidatieTool"
__credits__ = ["D2HYDRO", "HKV", "HydroConsult"]
__version__ = "1.3.0"

__license__ = "MIT"
__maintainer__ = "Daniel Tollenaar"
__email__ = "daniel@d2hydro.nl"

import fiona  # top-level import to avoid fiona import issue: https://github.com/conda-forge/fiona-feedstock/issues/213
from pathlib import Path
from hydamo_validation.functions import topologic as topologic_functions
from hydamo_validation.functions import logic as logic_functions
from hydamo_validation.functions import general as general_functions
from hydamo_validation.validator import validator

__all__ = [
    "fiona",
    "topologic_functions",
    "logic_functions",
    "general_functions",
    "validator",
    "SCHEMAS_PATH",
]
