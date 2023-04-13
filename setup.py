# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import pkg_resources  # part of setuptools

from hydamo_validation import __version__

#%%
with open('README.md', encoding='utf8') as f:
    long_description = f.read()

setup(
    name='hydamo_validation',
    version=__version__,
    description='Validation module for HyDAMO data',
    long_description=long_description,
    packages=find_packages(),
    package_data={
            "hydamo_validation": ["schemas/hydamo/*.json","schemas/rules/*.json"],
        },
    python_requires='<=3.9',
    install_requires=[
        'geopandas<=0.10.2',
        'pandas<=1.5.3',
        'rasterio==1.2.10',
        'rasterstats<=0.16.0'
    ],
    keywords='HyDAMO data validation',
)
