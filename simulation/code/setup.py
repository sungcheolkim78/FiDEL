
from setuptools import setup, find_packages

VERSION = "0.0.0a0"
DESCRIPTION = ("Implementation of FiDEL strategy for "
                "aggregating predictions by binary "
                "classifiers.")


setup(
    name="fd",
    author="Robert Vogel",
    description=DESCRIPTION,
    version=VERSION,
    packages=["fd"]
    )
