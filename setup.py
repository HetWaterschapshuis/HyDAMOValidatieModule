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
    python_requires='>=3.8',
    install_requires=[
        'geopandas',
        'rasterio',
        'rasterstats'
    ],
    keywords='HyDAMO data validation',
)
