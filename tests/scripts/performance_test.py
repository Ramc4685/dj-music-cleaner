#!/usr/bin/env python3
"""
Performance test for DJ Music Cleaner with and without optimizations
"""
import os
import time
import shutil
import tempfile
from djmusiccleaner.dj_music_cleaner import DJMusicCleaner

def run_performance_test(input_folder, use_optimizations=True):
    """Run a performance test with or without optimizations"""
    # Create temporary output folder
    output_folder = tempfile.mkdtemp(prefix="dj_cleaner_test_")
    
    print(f"\n{'=' * 50}")
    print(f"TESTING WITH {'OPTIMIZATIONS ENABLED' if use_optimizations else 'OPTIMIZATIONS DISABLED'}")
    print(f"{'=' * 50}")
    
    # Create cleaner instance
    cleaner = DJMusicCleaner()
    
    # Disable optimizations if needed
    if not use_optimizations:
        # Monkey patch to disable caching
        cleaner._get_audio_file = lambda filepath: None  # Force cache miss
        cleaner._queue_tag_update = lambda filepath, tag_dict: None  # Disable batching
        cleaner._flush_tag_updates = lambda dry_run=False: None  # Disable flushing
    
    # Measure processing time
    start_time = time.time()
    
    # Process the folder with DJMusicCleaner
    cleaner.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        enhance_online=False,  # Skip online processing for test
        dj_analysis=True,
        analyze_quality=True,
        detect_key=True,
        detect_cues=False,  # Skip cue detection to save time
        calculate_energy=True,
        workers=0,  # Auto-determine worker count
        dry_run=True  # Don't actually modify files
    )
    
    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Files processed: {len(cleaner.stats.get('processed_files', []))}")
    
    # Clean up temp directory
    shutil.rmtree(output_folder)
    
    return elapsed_time

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DJ Music Cleaner performance')
    parser.add_argument('input_folder', help='Path to folder containing MP3 files for testing')
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a valid directory")
        exit(1)
    
    # Run tests with and without optimizations
    time_with_opt = run_performance_test(args.input_folder, use_optimizations=True)
    time_without_opt = run_performance_test(args.input_folder, use_optimizations=False)
    
    # Calculate and report improvement
    improvement = (time_without_opt - time_with_opt) / time_without_opt * 100
    
    print(f"\n{'=' * 50}")
    print(f"PERFORMANCE TEST RESULTS:")
    print(f"{'=' * 50}")
    print(f"Time with optimizations:    {time_with_opt:.2f} seconds")
    print(f"Time without optimizations: {time_without_opt:.2f} seconds")
    print(f"Performance improvement:    {improvement:.1f}%")
    
    if improvement > 0:
        print("\n✅ Optimizations were effective!")
    else:
        print("\n⚠️ Optimizations did not improve performance in this test.")
