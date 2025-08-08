#!/usr/bin/env python3
"""
Setup script for ML & LLM Finance Portfolio
"""

from setuptools import setup, find_packages
import os


# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="ml-llm-finance-portfolio",
    version="0.1.0",
    author="Milena Georgieva",
    author_email="milenageorgieva@example.com",
    description="ML & LLM Finance Portfolio - Financial Analysis, Machine Learning & Agentic Workflows",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mimitheone/ml-llm-portfolio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-portfolio=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="machine-learning, finance, llm, ai, ml, portfolio, regression, classification",
    project_urls={
        "Bug Reports": "https://github.com/mimitheone/ml-llm-portfolio/issues",
        "Source": "https://github.com/mimitheone/ml-llm-portfolio",
        "Documentation": "https://github.com/mimitheone/ml-llm-portfolio#readme",
    },
)
