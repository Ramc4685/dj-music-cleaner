#!/usr/bin/env python3
"""
Comprehensive test script to compare CLI and GUI file processing
This script directly calls the DJMusicCleaner to process files with CLI and GUI settings
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from djmusiccleaner.dj_music_cleaner import DJMusicCleaner

# Test directories
TEST_INPUT = os.path.abspath('./test_input')
CLI_OUTPUT = os.path.abspath('./test_output_cli')
GUI_OUTPUT = os.path.abspath('./test_output_gui')

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

# ---- CLI MODE TESTING -----
print("\n🔍 TESTING CLI MODE...")
# Run the actual CLI script with command line args
cli_cmd = [
    sys.executable,
    "-m", "djmusiccleaner.dj_music_cleaner",
    "-i", TEST_INPUT,
    "-o", CLI_OUTPUT,
    "--workers", "1",  # Use single worker for stability
    "--no-dj",         # Disable DJ analysis for speed
    "--no-report"      # Disable report generation for simplicity
]

print(f"Running CLI command: {' '.join(cli_cmd)}")
try:
    # Use subprocess.run with capture_output to prevent output flooding
    subprocess.run(cli_cmd, check=True, capture_output=True)
    print("✅ CLI processing completed")
except subprocess.CalledProcessError as e:
    print(f"❌ CLI processing failed: {e}")
    print(f"STDOUT: {e.stdout.decode()}")
    print(f"STDERR: {e.stderr.decode()}")

# ---- GUI MODE TESTING -----
print("\n🔍 TESTING GUI MODE...")
# Use the DJMusicCleaner directly, like the GUI does
try:
    # Initialize DJMusicCleaner
    cleaner = DJMusicCleaner()
    
    # Process with GUI parameters
    print("Running GUI-style processing...")
    gui_files = cleaner.process_folder(
        input_folder=TEST_INPUT,
        output_folder=GUI_OUTPUT,
        enhance_online=False,
        dj_analysis=False,
        high_quality_only=False,
        generate_report=False,
        detailed_report=False,
        include_year_in_filename=False,  # Match CLI behavior for Artist - Title format
        analyze_quality=True,
        detect_key=False,
        detect_cues=False,
        calculate_energy=False,
        normalize_loudness=False,
        dry_run=False,
        workers=1,                       # Use single worker for stability
        skip_id3=False
    )
    print(f"✅ GUI processing completed with {len(gui_files) if gui_files else 0} files")
except Exception as e:
    import traceback
    print(f"❌ GUI processing failed: {e}")
    traceback.print_exc()

# ----- RESULTS COMPARISON ------
print("\n=== RESULTS COMPARISON ===")

# Compare output directories
cli_output_files = sorted([f for f in os.listdir(CLI_OUTPUT) if f.endswith('.mp3')])
gui_output_files = sorted([f for f in os.listdir(GUI_OUTPUT) if f.endswith('.mp3')])

print(f"\nCLI files found: {len(cli_output_files)}")
print(f"GUI files found: {len(gui_output_files)}")

# Show files side by side
print("\n{:<50} | {:<50}".format("CLI OUTPUT", "GUI OUTPUT"))
print("-" * 102)

# Get max number of files to display
max_files = max(len(cli_output_files), len(gui_output_files))
for i in range(max_files):
    cli_file = cli_output_files[i] if i < len(cli_output_files) else ""
    gui_file = gui_output_files[i] if i < len(gui_output_files) else ""
    print("{:<50} | {:<50}".format(cli_file, gui_file))

# Check if output is identical
if cli_output_files == gui_output_files:
    print("\n✅ SUCCESS! CLI and GUI output filenames are IDENTICAL!")
else:
    print("\n❌ FAIL! CLI and GUI output filenames DIFFER!")
    
    # Find differences
    cli_set = set(cli_output_files)
    gui_set = set(gui_output_files)
    
    print("\nFiles in CLI but not in GUI:")
    for f in cli_set - gui_set:
        print(f"  - {f}")
    
    print("\nFiles in GUI but not in CLI:")
    for f in gui_set - cli_set:
        print(f"  - {f}")

print("\nTest completed!")
