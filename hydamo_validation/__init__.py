__author__ = ["Het Waterschapshuis", "D2HYDRO", "HKV", "HydroConsult"]
__copyright__ = "Copyright 2026, HyDAMO ValidatieTool"
__credits__ = ["Het Waterschapshuis", "D2HYDRO", "HKV", "HydroConsult"]
__version__ = "1.4.2"

__license__ = "MIT"
__maintainer__ = "Philip Hansmann"
__email__ = "hydamo-validatietool@hetwaterschapshuis.nl"

from hydamo_validation.functions import topologic as topologic_functions
from hydamo_validation.functions import logic as logic_functions
from hydamo_validation.functions import general as general_functions
from hydamo_validation.validator import validator

__all__ = [
    "topologic_functions",
    "logic_functions",
    "general_functions",
    "validator",
    "SCHEMAS_PATH",
]
