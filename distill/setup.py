"""
Setup script for Frontier Braille Model Corpus package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="frontier-braille-corpus",
    version="1.0.0",
    author="Manus AI",
    description="ML-ready corpus for training dynamic-vocabulary, contraction-learning models on 8-dot Braille",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manus-ai/frontier-braille-corpus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "frontier-braille-generate=generate_corpus:main",
            "frontier-braille-evaluate=evaluation:main",
        ],
    },
)
