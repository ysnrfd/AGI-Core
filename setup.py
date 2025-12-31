# setup.py
"""
Setup script for AGI Core
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="agi-core",
    version="0.1.0",
    author="YSNRFD AGI",
    description="A modular AGI Core framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ysnrfd/agi-core",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "agi-core=agi_core.main:main",
        ],
    },
)