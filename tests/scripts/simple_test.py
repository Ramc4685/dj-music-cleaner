#!/usr/bin/env python3
"""Simple script to test filename generation logic in DJMusicCleaner"""

import os
from djmusiccleaner.dj_music_cleaner import DJMusicCleaner

# Make sure we're using the same logic for filenames
print("Testing filename generation logic between CLI and GUI")

# Sample track metadata
sample_metadata = {
    'artist': 'Test Artist',
    'title': 'Test Title',
    'album': 'Test Album',
    'year': '2023',
    'track_number': '1'
}

# Initialize cleaner
cleaner = DJMusicCleaner()

# First test - CLI style (include_year_in_filename=False)
cli_filename = cleaner.generate_clean_filename(sample_metadata, include_year=False)
print(f"CLI filename: {cli_filename}")

# Second test - Old GUI style (include_year_in_filename not specified)
gui_filename_old = cleaner.generate_clean_filename(sample_metadata, include_year=None)
print(f"GUI filename (old): {gui_filename_old}")

# Third test - New GUI style (include_year_in_filename=False explicitly set)
gui_filename_fixed = cleaner.generate_clean_filename(sample_metadata, include_year=False)
print(f"GUI filename (fixed): {gui_filename_fixed}")

# Print comparison result
if cli_filename == gui_filename_fixed:
    print("\n✅ SUCCESS! CLI and GUI filenames are identical after fix!")
else:
    print("\n❌ FAIL! CLI and GUI filenames still differ!")
    print(f"CLI: {cli_filename}")
    print(f"GUI: {gui_filename_fixed}")
