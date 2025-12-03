"""Setup script for Shodh-Memory Python package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent.parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="shodh-memory",
    version="0.1.0",
    author="Roshera Team",
    author_email="hello@roshera.com",
    description="Offline, user-isolated memory layer for AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roshera/shodh-memory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.990",
        ]
    },
    include_package_data=True,
    package_data={
        "shodh_memory": ["bin/*"],
    },
    entry_points={
        "console_scripts": [
            "shodh-memory=shodh_memory.client:main",
        ],
    },
)
