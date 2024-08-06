import io
import os
from pathlib import Path

from setuptools import find_packages, setup


NAME = 'Loan Prediction App'
DESCRIPTION = 'This is Loan Prediction Model'
EMAIL = 'didihbabs@gmail.com'
AUTHOR = 'BABS'
URL = 'https://github.com/DIDIHBABS/loans_prediction_mlops'
REQUIRES_PYTHON = '>=3.7.0'


setup(
    name= NAME,
    version='0.0.1',
    description= DESCRIPTION,
    url = URL,
    authors_email = EMAIL,
    REQUIRES_PYTHON = '>=3.7.0',
    install_requires=[
        'requests',
        'importlib-metadata; python_version<"3.10"',
    ],
)