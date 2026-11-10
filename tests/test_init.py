"""
Test voor __init__.py
"""

from hydamo_validation.__init__ import (
    __contributors__,
    __copyright__,
    __license__,
    __maintainer__,
    __email__,
)

contributors = ["Het Waterschapshuis", "D2Hydro", "HKV", "HydroConsult"]
copyright = "Copyright (c) 2021-2026 Het Waterschapshuis"

license = "MIT"
maintainer = "Philip Hansmann"
email = "hydamo-validatietool@hetwaterschapshuis.nl"


def test_contributors():
    assert __contributors__ == contributors


def test_copyright():
    assert __copyright__ == copyright


def test_license():
    assert __license__ == license


def test_maintainer():
    assert __maintainer__ == maintainer


def test_email():
    assert __email__ == email
