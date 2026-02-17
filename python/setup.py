"""Setup script for Shodh-Memory Python package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="shodh-memory",
    version="0.1.8",
    author="Shodh Team",
    author_email="29.varuns@gmail.com",
    description="Cognitive memory system for AI agents - biological memory processing in a single binary",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/varun29ankuS/shodh-memory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "langchain": [
            "langchain-core>=0.2.0",
        ],
        "llamaindex": [
            "llama-index-core>=0.10.0",
        ],
        "openai-agents": [
            "openai-agents>=0.0.7",
            "pydantic>=2.0.0",
        ],
        "all": [
            "langchain-core>=0.2.0",
            "llama-index-core>=0.10.0",
            "openai-agents>=0.0.7",
            "pydantic>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.990",
        ],
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
