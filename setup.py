from setuptools import setup, find_packages
from pathlib import Path
import subprocess
import sys

# Function to read the requirements.txt file
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Safely read the README.md file
def read_readme():
    readme_path = Path(__file__).parent / 'README.md'
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""
    
def version():
    with open(Path(__file__).parent / 'version', 'r') as file:
        v = file.readline()
    return v

setup(
    name='nn-stat',
    version=version(),
    description="Neural Network Performance Analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="ABrain One and contributors",
    author_email="AI@ABrain.one",
    url="https://ABrain.one",
    packages=find_packages(include=["ab.*"]),
    install_requires=read_requirements(),
    extras_require = {'dataset': ['nn-dataset']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,
)
