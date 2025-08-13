#!/usr/bin/env python3
"""Test script to compare filename generation logic between CLI and GUI"""

from djmusiccleaner.dj_music_cleaner import DJMusicCleaner

# Sample track metadata
artist = "Test Artist"
title = "Test Title" 
year = "2023"

# Initialize cleaner
cleaner = DJMusicCleaner()

# CLI style - no year included
cli_filename = cleaner.generate_clean_filename(artist, title)
print(f"CLI filename: {cli_filename}")

# GUI style with year
gui_filename_with_year = cleaner.generate_clean_filename(artist, title, year)
print(f"GUI filename with year: {gui_filename_with_year}")

print("\nNote: Both CLI and GUI should be configured to use the same style.")
print("If CLI uses 'artist - title', GUI should also use 'artist - title'.")
print("Currently, we've updated the GUI to match the CLI's behavior by not including the year.")
