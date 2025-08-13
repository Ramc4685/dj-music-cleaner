#!/usr/bin/env python3
# Test script to compare CLI and GUI file processing behavior

import os
import shutil
import sys
from djmusiccleaner.dj_music_cleaner import DJMusicCleaner

# Test directories
TEST_INPUT = '/Users/ramc/Documents/Code/dj-music-cleaner/test_input'
CLI_OUTPUT = '/Users/ramc/Documents/Code/dj-music-cleaner/test_output_cli'
GUI_OUTPUT = '/Users/ramc/Documents/Code/dj-music-cleaner/test_output_gui'

# Create output dirs if they don't exist
os.makedirs(CLI_OUTPUT, exist_ok=True)
os.makedirs(GUI_OUTPUT, exist_ok=True)

# Clear output directories
for folder in [CLI_OUTPUT, GUI_OUTPUT]:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Error clearing {file_path}: {e}')

print("=== TESTING CLI VS GUI PROCESSING ===")
print(f"Input folder: {TEST_INPUT}")
print(f"CLI output: {CLI_OUTPUT}")
print(f"GUI output: {GUI_OUTPUT}")
print("="*40)

# Initialize the cleaner
cleaner = DJMusicCleaner(acoustid_api_key=os.getenv("ACOUSTID_API_KEY", ""))

# Process with CLI parameters (similar to main() in dj_music_cleaner.py)
print("\nProcessing with CLI parameters...")
cli_files = cleaner.process_folder(
    input_folder=TEST_INPUT,
    output_folder=CLI_OUTPUT,
    enhance_online=False,  # No online lookup for speed
    include_year_in_filename=False,
    dj_analysis=False,     # No DJ analysis for speed
    analyze_quality=True,
    detect_key=False,
    detect_cues=False,
    calculate_energy=False,
    normalize_loudness=False,
    rekordbox_xml=None,
    export_xml=False,
    generate_report=True,
    high_quality_only=False,
    detailed_report=True,
    rekordbox_preserve=False,
    dry_run=False,
    workers=0,
    skip_id3=False
)

# Process with GUI parameters (as in dj_music_gui.py after our fix)
print("\nProcessing with GUI parameters...")
gui_files = cleaner.process_folder(
    input_folder=TEST_INPUT,
    output_folder=GUI_OUTPUT,
    enhance_online=False,
    dj_analysis=False,
    high_quality_only=False,
    generate_report=True,
    detailed_report=True,
    include_year_in_filename=False,
    analyze_quality=True,
    detect_key=False,
    detect_cues=False,
    calculate_energy=False,
    normalize_loudness=False,
    dry_run=False,
    workers=0,
    skip_id3=False
)

# Compare results
print("\n=== RESULTS COMPARISON ===")
print(f"CLI files processed: {len(cli_files)}")
print(f"GUI files processed: {len(gui_files)}")

# Compare output directories
cli_output_files = sorted(os.listdir(CLI_OUTPUT))
gui_output_files = sorted(os.listdir(GUI_OUTPUT))

print("\nCLI output filenames:")
for f in cli_output_files:
    if f.endswith('.mp3'):
        print(f"  - {f}")

print("\nGUI output filenames:")
for f in gui_output_files:
    if f.endswith('.mp3'):
        print(f"  - {f}")

# Check if output is identical
if cli_output_files == gui_output_files:
    print("\n✅ SUCCESS! CLI and GUI output filenames are identical!")
else:
    print("\n❌ FAIL! CLI and GUI output filenames differ!")
    
    # Find differences
    cli_set = set(cli_output_files)
    gui_set = set(gui_output_files)
    
    print("\nFiles in CLI but not in GUI:")
    for f in cli_set - gui_set:
        if f.endswith('.mp3'):
            print(f"  - {f}")
    
    print("\nFiles in GUI but not in CLI:")
    for f in gui_set - cli_set:
        if f.endswith('.mp3'):
            print(f"  - {f}")
