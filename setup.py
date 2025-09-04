"""Setup configuration for EGW Query Expansion"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="egw-query-expansion",
    version="0.1.0",
    author="EGW Query Expansion Team",
    author_email="contact@example.com",
    description="Entropic Gromov-Wasserstein Query Expansion with Hybrid Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/egw-query-expansion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "scikit-learn>=1.3.0",
        "packaging>=21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "jupyter>=1.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": ["faiss-gpu>=1.7.4"],
    },
    package_data={"egw_query_expansion": ["configs/*.yaml"]},
    entry_points={
        "console_scripts": [
            "egw-troubleshoot=egw_query_expansion.cli.troubleshoot:main",
            "egw-expand=egw_query_expansion.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/example/egw-query-expansion/issues",
        "Source": "https://github.com/example/egw-query-expansion",
        "Documentation": "https://egw-query-expansion.readthedocs.io/",
    },
)
