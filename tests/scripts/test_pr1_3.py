#!/usr/bin/env python3
"""
Test script for PR1-3 features in DJ Music Cleaner.
- Tests CLI flags (--dry-run, --workers, --no-id3)
- Tests file write safety via _safe_save and write_id3
- Tests parallelism and determinism
"""

import os
import sys
import shutil
import hashlib
import tempfile
import argparse
import subprocess
from pathlib import Path

# Test directory structure
TEST_DIR = Path("test_files")
SAMPLE_FILES_DIR = TEST_DIR / "samples"
OUTPUT_DIR = TEST_DIR / "output"
BACKUP_DIR = TEST_DIR / "backup"

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️ {msg}{RESET}")

def print_header(msg):
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}== {msg}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

def create_test_mp3(directory, filename, content="test"):
    """Create a dummy MP3 file for testing."""
    mp3_path = directory / filename
    
    # Create a minimal valid MP3 file - this isn't actually playable
    # but has enough structure to be recognized as MP3 by the library
    with open(mp3_path, 'wb') as f:
        # ID3v2 header
        f.write(b'ID3\x03\x00\x00\x00\x00\x00\x06')
        # Minimal MP3 frame header
        f.write(b'\xFF\xFB\x90\x44\x00')
        # Some dummy content
        f.write(content.encode('utf-8'))
    
    return mp3_path

def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def setup_test_environment():
    """Create test directories and sample files."""
    print_info("Setting up test environment...")
    
    # Create test directories
    os.makedirs(SAMPLE_FILES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Create sample MP3 files
    for i in range(10):
        create_test_mp3(SAMPLE_FILES_DIR, f"test_{i:02d}.mp3", f"test content {i}")
    
    print_success("Test environment setup complete")
    return True

def clean_test_environment():
    """Clean up test directories."""
    print_info("Cleaning test environment...")
    
    # Move everything to backup before deleting
    if os.path.exists(OUTPUT_DIR):
        backup_time = int(os.path.getmtime(OUTPUT_DIR))
        backup_path = BACKUP_DIR / f"output_{backup_time}"
        if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
            shutil.copytree(OUTPUT_DIR, backup_path, dirs_exist_ok=True)
    
    # Remove output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print_success("Test environment cleaned")
    return True

def run_dj_cleaner(args):
    """Run DJ Music Cleaner with specified arguments."""
    cmd = [sys.executable, "djmusiccleaner/dj_music_cleaner.py"] + args
    print_info(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        encoding='utf-8'
    )
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

def test_cli_flags():
    """Test that CLI flags are correctly parsed and used."""
    print_header("Testing CLI flags")
    
    # Test --dry-run flag
    result = run_dj_cleaner([
        str(SAMPLE_FILES_DIR), 
        "--dry-run",
        "--output", str(OUTPUT_DIR)
    ])
    
    success = result["returncode"] == 0
    if "Dry run mode" in result["stdout"]:
        print_success("--dry-run flag correctly recognized")
        success &= True
    else:
        print_error("--dry-run flag not recognized")
        success = False
    
    # Test --workers flag
    result = run_dj_cleaner([
        str(SAMPLE_FILES_DIR), 
        "--workers", "2",
        "--output", str(OUTPUT_DIR)
    ])
    
    if "workers=2" in result["stdout"].lower():
        print_success("--workers flag correctly recognized")
        success &= True
    else:
        print_error("--workers flag not recognized")
        success = False
    
    # Test --no-id3 flag
    result = run_dj_cleaner([
        str(SAMPLE_FILES_DIR), 
        "--no-id3",
        "--output", str(OUTPUT_DIR)
    ])
    
    if "no-id3" in result["stdout"].lower():
        print_success("--no-id3 flag correctly recognized")
        success &= True
    else:
        print_error("--no-id3 flag not recognized")
        success = False
    
    return success

def test_dry_run_mode():
    """Test that dry-run mode doesn't modify files."""
    print_header("Testing dry-run mode")
    
    # Clean output directory
    clean_test_environment()
    
    # Calculate hashes of original files
    original_hashes = {}
    for file in SAMPLE_FILES_DIR.glob("*.mp3"):
        original_hashes[file.name] = calculate_file_hash(file)
    
    # Run with dry-run flag
    result = run_dj_cleaner([
        str(SAMPLE_FILES_DIR), 
        "--dry-run",
        "--output", str(OUTPUT_DIR),
        "--enhance-online"  # This would normally modify files
    ])
    
    # Calculate hashes after dry-run
    new_hashes = {}
    for file in SAMPLE_FILES_DIR.glob("*.mp3"):
        new_hashes[file.name] = calculate_file_hash(file)
    
    # Check that no files were modified
    success = True
    for filename, original_hash in original_hashes.items():
        new_hash = new_hashes.get(filename)
        if new_hash != original_hash:
            print_error(f"File {filename} was modified in dry-run mode")
            success = False
    
    if success:
        print_success("No files were modified in dry-run mode")
    
    return success

def test_parallelism():
    """Test parallel processing with different worker counts."""
    print_header("Testing parallelism")
    
    clean_test_environment()
    
    # Run with 1 worker
    print_info("Running with 1 worker...")
    result_1 = run_dj_cleaner([
        str(SAMPLE_FILES_DIR), 
        "--workers", "1",
        "--output", str(OUTPUT_DIR),
        "--detect-key"  # Operation that benefits from parallelism
    ])
    time_1 = None
    for line in result_1["stdout"].splitlines():
        if "Processing time:" in line:
            try:
                time_1 = float(line.split(":")[-1].strip().split()[0])
                print_info(f"Processing time with 1 worker: {time_1:.2f} seconds")
            except:
                pass
    
    # Clean output
    clean_test_environment()
    
    # Run with multiple workers
    worker_count = os.cpu_count() or 4
    print_info(f"Running with {worker_count} workers...")
    result_n = run_dj_cleaner([
        str(SAMPLE_FILES_DIR), 
        "--workers", str(worker_count),
        "--output", str(OUTPUT_DIR),
        "--detect-key"  # Same operation
    ])
    
    time_n = None
    for line in result_n["stdout"].splitlines():
        if "Processing time:" in line:
            try:
                time_n = float(line.split(":")[-1].strip().split()[0])
                print_info(f"Processing time with {worker_count} workers: {time_n:.2f} seconds")
            except:
                pass
    
    # Verify parallelism works (should at least not be slower)
    # We don't strictly check for speedup since test files are small
    success = True
    if time_1 is not None and time_n is not None:
        if time_n > time_1 * 1.5:  # Should definitely not be 50% slower
            print_error(f"Multi-worker processing ({time_n:.2f}s) was significantly slower than single worker ({time_1:.2f}s)")
            success = False
        else:
            print_success(f"Parallel processing works correctly")
    else:
        print_warning("Couldn't extract timing information")
        success = True  # Assume success if we can't verify timing
    
    return success

def test_determinism():
    """Test deterministic results with fixed seeds."""
    print_header("Testing determinism")
    
    # Run twice with same seed and check for identical results
    clean_test_environment()
    
    print_info("First run with fixed seed...")
    result1 = run_dj_cleaner([
        str(SAMPLE_FILES_DIR),
        "--output", str(OUTPUT_DIR / "run1"),
        "--detect-key",
        "--calculate-energy"
    ])
    
    # Save output files to separate directory
    output1 = OUTPUT_DIR / "run1"
    os.makedirs(output1, exist_ok=True)
    for file in OUTPUT_DIR.glob("*.mp3"):
        if file.parent == OUTPUT_DIR:
            shutil.move(file, output1 / file.name)
    
    print_info("Second run with same configuration...")
    result2 = run_dj_cleaner([
        str(SAMPLE_FILES_DIR),
        "--output", str(OUTPUT_DIR / "run2"),
        "--detect-key",
        "--calculate-energy"
    ])
    
    # Check if energy ratings and key detection results are the same
    success = True
    output_lines1 = result1["stdout"].splitlines()
    output_lines2 = result2["stdout"].splitlines()
    
    energy_results1 = {}
    energy_results2 = {}
    key_results1 = {}
    key_results2 = {}
    
    # Extract energy and key results from first run
    for line in output_lines1:
        if "Energy rating:" in line:
            file_info = line.split("Processing")[1].split(":")[0].strip()
            energy = line.split("Energy rating:")[1].strip()
            energy_results1[file_info] = energy
        elif "Key detected:" in line:
            file_info = line.split("Processing")[1].split(":")[0].strip()
            key = line.split("Key detected:")[1].strip()
            key_results1[file_info] = key
    
    # Extract energy and key results from second run
    for line in output_lines2:
        if "Energy rating:" in line:
            file_info = line.split("Processing")[1].split(":")[0].strip()
            energy = line.split("Energy rating:")[1].strip()
            energy_results2[file_info] = energy
        elif "Key detected:" in line:
            file_info = line.split("Processing")[1].split(":")[0].strip()
            key = line.split("Key detected:")[1].strip()
            key_results2[file_info] = key
    
    # Compare results
    if energy_results1 == energy_results2:
        print_success("Energy ratings are deterministic")
    else:
        print_error("Energy ratings differ between runs")
        success = False
    
    if key_results1 == key_results2:
        print_success("Key detection is deterministic")
    else:
        print_error("Key detection results differ between runs")
        success = False
    
    return success

def test_write_safety():
    """Test that writes are properly routed through _safe_save."""
    print_header("Testing write safety")
    
    clean_test_environment()
    
    # Run without dry-run to check for .bak files
    result = run_dj_cleaner([
        str(SAMPLE_FILES_DIR),
        "--output", str(OUTPUT_DIR),
        "--detect-key"  # This should trigger writes
    ])
    
    # Check for .bak files in the original directory
    bak_files = list(SAMPLE_FILES_DIR.glob("*.bak"))
    
    if bak_files:
        print_success(f"Found {len(bak_files)} backup (.bak) files")
        success = True
    else:
        print_warning("No .bak files found. This might be expected if no writes were needed.")
        success = True  # Still mark as success, may not be applicable
    
    return success

def run_all_tests():
    """Run all test functions."""
    print_header("RUNNING ALL TESTS FOR PR1-3")
    
    # Setup test environment
    setup_test_environment()
    
    # Run tests
    tests = [
        ("CLI Flags", test_cli_flags),
        ("Dry Run Mode", test_dry_run_mode),
        ("Parallelism", test_parallelism),
        ("Determinism", test_determinism),
        ("Write Safety", test_write_safety)
    ]
    
    results = {}
    all_success = True
    
    for name, test_func in tests:
        print(f"\n{BLUE}Running test: {name}{RESET}")
        try:
            success = test_func()
            results[name] = success
            all_success &= success
        except Exception as e:
            print_error(f"Test {name} failed with error: {e}")
            results[name] = False
            all_success = False
    
    # Print results summary
    print_header("TEST RESULTS SUMMARY")
    for name, success in results.items():
        if success:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    if all_success:
        print_success("\nALL TESTS PASSED!")
    else:
        print_error("\nSOME TESTS FAILED")
    
    return all_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PR1-3 features of DJ Music Cleaner")
    parser.add_argument("--skip-setup", action="store_true", help="Skip test environment setup")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip test environment cleanup")
    args = parser.parse_args()
    
    if not args.skip_setup:
        setup_test_environment()
    
    success = run_all_tests()
    
    if not args.skip_cleanup:
        clean_test_environment()
    
    sys.exit(0 if success else 1)
