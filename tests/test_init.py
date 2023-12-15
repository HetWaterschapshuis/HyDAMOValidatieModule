"""
Test voor __init__.py
"""

from hydamo_validation.__init__ import (__author__,
                                        __copyright__,
                                        __credits__,
                                        __version__,
                                        __license__,
                                        __maintainer__,
                                        __email__)

author = ["Het Waterschapshuis", "D2HYDRO", "HKV", "HydroConsult"]
copyright = "Copyright 2021, HyDAMO ValidatieTool"
credits = ["D2HYDRO", "HKV", "HydroConsult"]

license = "MIT"
maintainer = "Daniel Tollenaar"
email = "daniel@d2hydro.nl"


def test_author():
    assert __author__ == author


def test_copyright():
    assert __copyright__ == copyright


def test_credits():
    assert __credits__ == credits


def test_license():
    assert __license__ == license


def test_maintainer():
    assert __maintainer__ == maintainer


def test_email():
    assert __email__ == email
