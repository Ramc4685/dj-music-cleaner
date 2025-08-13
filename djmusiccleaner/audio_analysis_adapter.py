#!/usr/bin/env python3
"""
DJ Music Cleaner - Audio Analysis Adapter

This module serves as an adapter between the main engine (dj_music_cleaner.py) and 
the isolated audio analysis service. It provides drop-in replacements for the 
existing DJ analysis functions that redirect to the isolated service.
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, Tuple, List, Optional, Union

# Import our client module for process-isolated analysis
from djmusiccleaner.audio_analysis_client import get_audio_analysis_client

# Global singleton client instance (lazy-loaded on first use)
_audio_client = None


def get_client():
    """
    Get the audio analysis client singleton instance.
    
    Returns:
        AudioAnalysisClient instance
    """
    global _audio_client
    
    if _audio_client is None:
        _audio_client = get_audio_analysis_client()
        
    return _audio_client


def detect_bpm(filepath):
    """
    Process-isolated BPM detection using aubio.
    
    This function provides a drop-in replacement for the existing detect_bpm method
    in DJMusicCleaner, but runs the analysis in a separate process for stability.
    
    Args:
        filepath: Path to the audio file
        
    Returns:
        Tuple of (bpm, confidence) or (None, 0) on error
    """
    try:
        print(f"   🔍 Analyzing BPM (isolated) for {os.path.basename(filepath)}...")
        
        client = get_client()
        if not client.available:
            print("   ⚠️ Isolated audio analysis service not available - BPM detection skipped")
            return None, 0.0
        
        bpm, confidence = client.detect_bpm(filepath)
        
        if bpm is not None:
            print(f"   🎵 Detected BPM: {bpm:.2f} (confidence: {confidence:.2f})")
        else:
            print("   ❌ Could not detect BPM")
        
        return bpm, confidence
        
    except Exception as e:
        print(f"   ❌ BPM detection error: {e}")
        traceback.print_exc()
        return None, 0.0


def detect_musical_key(filepath):
    """
    Process-isolated musical key detection using aubio.
    
    This function provides a drop-in replacement for the existing detect_musical_key method
    in DJMusicCleaner, but runs the analysis in a separate process for stability.
    
    Args:
        filepath: Path to the audio file
        
    Returns:
        Tuple of (key, confidence, camelot_key) or (None, 0, None) on error
    """
    try:
        print(f"   🔍 Analyzing musical key (isolated) for {os.path.basename(filepath)}...")
        
        client = get_client()
        if not client.available:
            print("   ⚠️ Isolated audio analysis service not available - key detection skipped")
            return None, 0, None
        
        key, camelot, confidence = client.detect_key(filepath)
        
        if key is not None:
            print(f"   🎹 Detected key: {key} (Camelot: {camelot}) with confidence: {confidence:.2f}")
            return key, confidence, camelot
        else:
            print("   ❌ Could not detect musical key")
            return None, 0, None
        
    except Exception as e:
        print(f"   ❌ Key detection error: {e}")
        traceback.print_exc()
        return None, 0, None


def calculate_energy_rating(filepath):
    """
    Process-isolated energy rating calculation using aubio.
    
    This function provides a drop-in replacement for the existing calculate_energy_rating method
    in DJMusicCleaner, but runs the analysis in a separate process for stability.
    
    Args:
        filepath: Path to the audio file
        
    Returns:
        Tuple of (energy_score, characteristics) or (None, None) on error
    """
    try:
        print(f"   🔍 Analyzing energy (isolated) for {os.path.basename(filepath)}...")
        
        client = get_client()
        if not client.available:
            print("   ⚠️ Isolated audio analysis service not available - energy calculation skipped")
            return None
        
        energy_score, characteristics = client.calculate_energy(filepath)
        
        if energy_score is not None:
            print(f"   ⚡ Track energy: {energy_score}/10")
        else:
            print("   ❌ Could not calculate energy rating")
        
        return energy_score
        
    except Exception as e:
        print(f"   ❌ Energy calculation error: {e}")
        traceback.print_exc()
        return None


def detect_cue_points(filepath, output_file=None):
    """
    Process-isolated cue point detection using aubio.
    
    This function provides a drop-in replacement for the existing detect_cue_points method
    in DJMusicCleaner, but runs the analysis in a separate process for stability.
    
    Args:
        filepath: Path to the audio file
        output_file: Optional path for saving cue points
        
    Returns:
        List of dictionaries containing cue points information or None on error
    """
    try:
        print(f"   🔍 Analyzing track structure for cue points (isolated) for {os.path.basename(filepath)}...")
        
        client = get_client()
        if not client.available:
            print("   ⚠️ Isolated audio analysis service not available - cue point detection skipped")
            return None
        
        cue_points, sections = client.detect_cue_points(filepath, output_file)
        
        if cue_points:
            print(f"   🎵 Detected {len(cue_points)} cue points")
            
            # Print a few of them for information
            for i, cue in enumerate(cue_points[:3]):
                print(f"     • {cue['description']} at {cue['position']:.2f}s")
                
            if len(cue_points) > 3:
                print(f"     • ...and {len(cue_points) - 3} more")
                
            return cue_points
        else:
            print("   ⚠️ No cue points detected")
            return None
        
    except Exception as e:
        print(f"   ❌ Cue point detection error: {e}")
        traceback.print_exc()
        return None


def check_status():
    """
    Check the status of the audio analysis service.
    
    Returns:
        Dictionary with status information
    """
    try:
        client = get_client()
        return client.check_status()
    except Exception as e:
        print(f"Error checking audio analysis service status: {e}")
        return {
            'available': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Simple test if run directly
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        
        if os.path.exists(filepath):
            status = check_status()
            print(f"Audio analysis service status: {'Available' if status.get('available', False) else 'Unavailable'}")
            
            print(f"\nTesting BPM detection on {filepath}")
            bpm, confidence = detect_bpm(filepath)
            print(f"BPM: {bpm}, Confidence: {confidence:.2f}")
            
            print(f"\nTesting key detection on {filepath}")
            key = detect_musical_key(filepath)
            print(f"Key: {key}")
            
            print(f"\nTesting energy calculation on {filepath}")
            energy = calculate_energy_rating(filepath)
            print(f"Energy score: {energy}/10")
            
            print(f"\nTesting cue point detection on {filepath}")
            cue_points = detect_cue_points(filepath)
        else:
            print(f"File not found: {filepath}")
    else:
        print("Usage: python audio_analysis_adapter.py <audio_file>")
