#!/usr/bin/env python3
"""
DJ Music Cleaner - Setup Configuration
Professional-grade DJ music library management tool with unified architecture
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies for basic functionality
core_requirements = [
    "mutagen>=1.47.0",      # Audio metadata handling
    "numpy>=1.24.0",        # Numerical processing
    "tqdm>=4.66.0",         # Progress bars
    "soundfile>=0.12.0",    # Audio file I/O
    "python-dotenv>=1.0.0", # Environment variables
]

# Audio analysis dependencies
audio_requirements = [
    "aubio>=0.4.9",         # Primary audio analysis engine
    "librosa>=0.10.0",      # Enhanced audio analysis (fallback)
    "scipy>=1.10.0",        # Signal processing
]

# Online enhancement dependencies
online_requirements = [
    "pyacoustid>=1.2.2",        # AcoustID fingerprinting
    "musicbrainzngs>=0.7.1",    # MusicBrainz API
    "requests>=2.28.0",         # HTTP requests
]

# Advanced features
advanced_requirements = [
    "pyloudnorm>=0.1.1",        # Loudness normalization
]

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

# Full installation includes all optional features
full_requirements = (
    core_requirements + 
    audio_requirements + 
    online_requirements + 
    advanced_requirements
)

setup(
    # Package information
    name="dj-music-cleaner",
    version="2.0.0",
    author="RamC Venkatasamy",
    author_email="ramc46@example.com",  # Update with actual email
    description="Professional-grade DJ music library management with unified architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ramc46/dj-music-cleaner",
    project_urls={
        "Bug Reports": "https://github.com/ramc46/dj-music-cleaner/issues",
        "Source": "https://github.com/ramc46/dj-music-cleaner",
        "Documentation": "https://github.com/ramc46/dj-music-cleaner#readme",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "test_*", "*.tests*"]),
    include_package_data=True,
    
    # Dependencies
    install_requires=core_requirements,
    extras_require={
        "audio": audio_requirements,
        "online": online_requirements, 
        "advanced": advanced_requirements,
        "full": full_requirements,
        "dev": dev_requirements,
    },
    
    # Console entry points
    entry_points={
        "console_scripts": [
            # Main unified CLI interface
            "dj-music-cleaner=djmusiccleaner.cli.unified_cli:main",
            
            # Alternative entry points
            "djmusiccleaner=djmusiccleaner.cli.unified_cli:main",
        ],
    },
    
    # Python version and classifiers
    python_requires=">=3.9",
    classifiers=[
        # Development status
        "Development Status :: 5 - Production/Stable",
        
        # Intended audience
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Environment
        "Environment :: Console",
        "Operating System :: OS Independent",
        
        # Programming language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Topic classification
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: System :: Archiving :: Packaging",
        "Topic :: Utilities",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "dj", "music", "audio", "metadata", "bpm", "key-detection", 
        "rekordbox", "music-library", "audio-analysis", "mp3", "flac",
        "cue-points", "beatgrid", "energy-analysis", "professional-audio"
    ],
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
)