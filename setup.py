#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="djmusiccleaner",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "mutagen>=1.47",
        "numpy>=1.24",
        "tqdm>=4.66",
        "pyacoustid>=1.2.2",
        "musicbrainzngs>=0.7",
        "python-dotenv>=1.0.0",
        "soundfile>=0.12",
        "aubio>=0.5.0",
        "pyloudnorm>=0.1.1",
        "librosa>=0.10",
    ],
    entry_points={
        "console_scripts": [
            "dj-music-cleaner=djmusiccleaner.dj_music_cleaner:cli_main",
        ],
        # GUI scripts moved to archive
        # "gui_scripts": [
        #     "dj-music-analyzer=djmusiccleaner.dj_music_gui:main",
        # ],
    },
    python_requires=">=3.8",
    description="DJ Music Analyzer and Cleaner - Command Line Tool",
    author="Ram C",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio",
    ],
)
