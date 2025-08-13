#!/usr/bin/env python3
"""
DJ Analysis Service - Process-isolated audio analysis for DJ features
Uses aubio for stable BPM and key detection without segmentation faults
"""

import os
import sys
import json
import traceback
import numpy as np
from collections import Counter

# Audio file loading
import soundfile as sf

# Aubio imports
try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False
    print("⚠️ aubio not installed. Run: pip install aubio")

# Optional: Legacy libraries for fallback/comparison
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

# Musical key utilities
# Krumhansl-Schmuckler key profiles (same as in original implementation)
K_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
K_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CAMELOT_MAP = {
    'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B', 'B': '1B', 'F#': '2B',
    'C#': '3B', 'G#': '4B', 'D#': '5B', 'A#': '6B', 'F': '7B',
    'Am': '8A', 'Em': '9A', 'Bm': '10A', 'F#m': '11A', 'C#m': '12A', 'G#m': '1A',
    'D#m': '2A', 'A#m': '3A', 'Fm': '4A', 'Cm': '5A', 'Gm': '6A', 'Dm': '7A'
}


def detect_bpm_aubio(filepath):
    """
    Detect BPM using aubio's tempo detection algorithm
    
    Args:
        filepath: Path to audio file
        
    Returns:
        float: BPM value or None if detection failed
    """
    if not AUBIO_AVAILABLE:
        return None
    
    try:
        # Load audio file with soundfile
        print("   🔍 Detecting BPM with aubio...")
        y, sr = sf.read(filepath)
        if len(y.shape) > 1:  # Convert stereo to mono
            y = np.mean(y, axis=1)
            
        # Process in chunks (more memory-efficient)
        win_s = 1024  # FFT size
        hop_s = 512   # hop size
        
        # Create tempo detector
        tempo = aubio.tempo("default", win_s, hop_s, sr)
        
        # Process audio in chunks
        bpms = []
        chunk_size = hop_s
        total_frames = len(y)
        
        for i in range(0, total_frames - chunk_size, chunk_size):
            chunk = y[i:i+chunk_size]
            if len(chunk) < chunk_size:  # Zero-pad the last chunk
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            tempo.do(chunk)
            
            # Collect non-zero tempo estimations
            current_bpm = tempo.get_bpm()
            if current_bpm > 0:
                bpms.append(current_bpm)
        
        # Get median BPM from all chunks for stability
        if bpms:
            bpm = np.median(bpms)
            print(f"   🎵 Detected BPM: {bpm:.1f}")
            return round(bpm, 1)
        else:
            print("   ❌ Could not detect BPM")
            return None
            
    except Exception as e:
        print(f"   ❌ BPM detection error: {e}")
        return None


def detect_key_aubio(filepath):
    """
    Detect musical key using aubio's pitch detection and key estimation
    
    Args:
        filepath: Path to audio file
        
    Returns:
        tuple: (key, confidence) or (None, 0) if detection failed
    """
    if not AUBIO_AVAILABLE:
        return None, 0
        
    try:
        print("   🔍 Detecting musical key using aubio...")
        
        # Load audio file with soundfile
        y, sr = sf.read(filepath)
        if len(y.shape) > 1:  # Convert stereo to mono
            y = np.mean(y, axis=1)
        
        # Process in chunks for memory efficiency
        win_s = 4096  # FFT size
        hop_s = 1024  # hop size
        
        # Use chroma filter to get harmonic content
        chromagram = aubio.cvec(win_s)
        fft = aubio.fft(win_s)
        pvoc = aubio.pvoc(win_s, hop_s)
        filter_bank = aubio.filterbank(12, win_s)
        filter_bank.set_triangle_bands(aubio.hztomel(40), aubio.hztomel(sr/2), 48)
        
        # Create buffer for chroma vector
        chroma = np.zeros((12,))
        sample_count = 0
        
        # Process audio in chunks
        for i in range(0, len(y) - hop_s, hop_s):
            chunk = y[i:i+hop_s]
            if len(chunk) < hop_s:  # Zero-pad the last chunk
                chunk = np.pad(chunk, (0, hop_s - len(chunk)))
            
            # Calculate spectrum and chromagram
            fft.do(chunk, chromagram)
            pvoc.do(chunk, chromagram)
            filter_bank.do(chromagram, chunk)
            
            # Accumulate normalized chroma vectors
            current_chroma = chunk / (np.max(chunk) + 1e-10)  # Normalize
            chroma += current_chroma
            sample_count += 1
        
        # Average the chroma vectors
        if sample_count > 0:
            chroma /= sample_count
        
        # Normalize for key correlation
        chroma_norm = chroma / (np.sum(chroma) + 1e-10)
        
        # Key detection using Krumhansl-Schmuckler algorithm (same as original)
        majors = []
        minors = []
        
        for i in range(12):
            # Correlation with major key profile
            major_corr = np.corrcoef(np.roll(chroma_norm, -i), K_MAJOR)[0, 1]
            majors.append(major_corr)
            
            # Correlation with minor key profile
            minor_corr = np.corrcoef(np.roll(chroma_norm, -i), K_MINOR)[0, 1]
            minors.append(minor_corr)
        
        # Find highest correlation
        max_major_corr = max(majors)
        max_minor_corr = max(minors)
        
        # Determine key
        if max_major_corr > max_minor_corr:
            key_idx = np.argmax(majors)
            key = PITCHES[key_idx]
            confidence = max_major_corr
        else:
            key_idx = np.argmax(minors)
            key = PITCHES[key_idx] + 'm'
            confidence = max_minor_corr
        
        # Normalize confidence to [0,1]
        confidence = max(0, min(1, (confidence + 1) / 2))
        
        print(f"   🎹 Detected key: {key} (confidence: {confidence:.2f})")
        
        # Get Camelot notation
        camelot = CAMELOT_MAP.get(key, 'Unknown')
        print(f"   🔑 Camelot key: {camelot}")
        
        return key, confidence, camelot
        
    except Exception as e:
        print(f"   ❌ Key detection error: {e}")
        traceback.print_exc()
        return None, 0, 'Unknown'


def detect_energy(filepath):
    """
    Detect energy/intensity of the track using aubio
    
    Args:
        filepath: Path to audio file
        
    Returns:
        float: Energy value (0-10) or None if detection failed
    """
    if not AUBIO_AVAILABLE:
        return None
    
    try:
        print("   🔍 Detecting track energy...")
        
        # Load audio
        y, sr = sf.read(filepath)
        if len(y.shape) > 1:  # Convert stereo to mono
            y = np.mean(y, axis=1)
        
        # Create aubio onset detector (for transient detection)
        win_s = 1024
        hop_s = 512
        onset = aubio.onset("hfc", win_s, hop_s, sr)
        
        # Process audio in chunks
        onsets = []
        total_frames = len(y)
        
        for i in range(0, total_frames - hop_s, hop_s):
            chunk = y[i:i+hop_s]
            if len(chunk) < hop_s:
                chunk = np.pad(chunk, (0, hop_s - len(chunk)))
            
            if onset.do(chunk):
                onsets.append(i / sr)  # Convert frame to time
        
        # Calculate onset density (onsets per second)
        if len(onsets) > 1:
            duration = total_frames / sr
            onset_density = len(onsets) / duration
        else:
            onset_density = 0
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(y**2))
        
        # Calculate spectral centroid using aubio
        specdesc = aubio.specdesc("centroid", win_s)
        pv = aubio.pvoc(win_s, hop_s)
        centroids = []
        
        for i in range(0, total_frames - hop_s, hop_s):
            chunk = y[i:i+hop_s]
            if len(chunk) < hop_s:
                chunk = np.pad(chunk, (0, hop_s - len(chunk)))
            
            spec = pv(chunk)
            centroid = specdesc(spec)
            centroids.append(centroid[0])
        
        # Combine features for energy calculation
        if centroids:
            avg_centroid = np.mean(centroids) / (sr/2)  # Normalize by Nyquist frequency
            # Energy formula combining RMS, onset density, and spectral centroid
            energy = (rms * 5 + onset_density * 1.5 + avg_centroid * 3.5) * 10
            
            # Limit to 0-10 range
            energy = max(0, min(10, energy))
            print(f"   ⚡ Track energy: {energy:.1f}/10")
            return round(energy, 1)
        else:
            print("   ❌ Could not detect energy")
            return None
            
    except Exception as e:
        print(f"   ❌ Energy detection error: {e}")
        return None


def analyze_file(filepath, analysis_type):
    """
    Analyze audio file with the specified analysis type
    
    Args:
        filepath: Path to audio file
        analysis_type: Type of analysis ('key', 'bpm', 'energy', 'all')
        
    Returns:
        dict: Analysis results
    """
    results = {}
    
    try:
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}
            
        if not AUBIO_AVAILABLE:
            return {"error": "aubio library not available"}
            
        # Perform requested analysis
        if analysis_type in ('bpm', 'all'):
            results['bpm'] = detect_bpm_aubio(filepath)
            
        if analysis_type in ('key', 'all'):
            key, confidence, camelot = detect_key_aubio(filepath)
            results['key'] = key
            results['key_confidence'] = confidence
            results['camelot'] = camelot
            
        if analysis_type in ('energy', 'all'):
            results['energy'] = detect_energy(filepath)
            
        return results
        
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_analysis_service():
    """
    Run as a standalone process to handle DJ analysis requests
    Reads requests from stdin and writes results to stdout as JSON
    """
    while True:
        try:
            # Read request from stdin
            request_line = sys.stdin.readline().strip()
            if not request_line:
                continue
                
            # Parse request JSON
            request = json.loads(request_line)
            
            # Process request
            filepath = request.get('filepath')
            analysis_type = request.get('analysis_type', 'all')
            
            # Perform analysis
            results = analyze_file(filepath, analysis_type)
            
            # Send results back as JSON
            response = {
                "success": "error" not in results,
                "results": results
            }
            print(json.dumps(response))
            sys.stdout.flush()
            
        except Exception as e:
            # Handle unexpected errors
            error_response = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(json.dumps(error_response))
            sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line usage
        if sys.argv[1] == "--service":
            run_analysis_service()
        else:
            # Direct analysis of a file
            filepath = sys.argv[1]
            analysis_type = sys.argv[2] if len(sys.argv) > 2 else "all"
            results = analyze_file(filepath, analysis_type)
            print(json.dumps(results, indent=2))
    else:
        # No arguments - run as service
        run_analysis_service()
