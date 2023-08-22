"""
Setup module for Blush refinement
"""

from setuptools import setup, find_packages

import os

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
readme_path = f"{lib_folder}/README.md"

# Load requirements
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path, "r") as f:
        install_requires = f.read().splitlines()
else:
    print("Could not find requirements file.")
    exit(1)

# Load readme file for long description
if os.path.isfile(readme_path):
    with open(readme_path, "r") as f:
        readme = f.read()
else:
    readme = None

# Add command line option
setup(
    name="relion_blush",
    description="The Blush functionality of RELION",
    version="0.0.1",
    long_description=readme,
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.5",
)

# Download, install and load model
# import relion_blush
# relion_blush.install_and_load_model("v1.0")
