#!/usr/bin/env python3
"""
DJ Music Cleaner - Audio Analysis Service

Provides isolated audio analysis features using aubio to prevent 
segmentation faults from crashing the main application.

This module runs as a separate process and communicates with the main
application through a simple pipe-based IPC mechanism.
"""

import os
import sys
import json
import time
import traceback
import numpy as np
from multiprocessing import Process, Pipe
from typing import Dict, Any, Tuple, List, Optional, Union

# Import aubio - our primary audio analysis library
try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False
    print("Warning: aubio not available. DJ analysis features will be limited.")

# Optional imports for format support
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Fallback imports (to be used only if aubio fails)
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOP_SIZE = 512
DEFAULT_WINDOW_SIZE = 2048
DEFAULT_TEMPO_METHOD = 'default'  # other options: 'spectrum', 'energy'
KEY_PROFILES = {
    'major': np.array([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ]),
    'minor': np.array([
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    ])
}
CAMELOT_WHEEL = {
    'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B', 'B': '1B', 
    'F#': '2B', 'C#': '3B', 'G#': '4B', 'D#': '5B', 'A#': '6B', 'F': '7B',
    'Am': '8A', 'Em': '9A', 'Bm': '10A', 'F#m': '11A', 'C#m': '12A', 'G#m': '1A',
    'D#m': '2A', 'A#m': '3A', 'Fm': '4A', 'Cm': '5A', 'Gm': '6A', 'Dm': '7A'
}
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class AudioAnalysisService:
    """
    Service class for audio analysis that runs in an isolated process.
    
    Provides methods for BPM detection, key detection, and energy analysis
    using aubio as the primary engine.
    """
    
    def __init__(self):
        """Initialize the audio analysis service"""
        self.fallback_mode = not AUBIO_AVAILABLE
        self.load_errors = 0
        self.analysis_errors = 0
        
    def load_audio(self, filepath: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Load audio file into memory.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            if AUBIO_AVAILABLE:
                # Use aubio's source to load audio
                hop_size = DEFAULT_HOP_SIZE
                source = aubio.source(filepath, DEFAULT_SAMPLE_RATE, hop_size)
                sample_rate = source.samplerate
                
                # Read the entire file
                total_frames = 0
                audio_buffer = np.zeros([0], dtype=np.float32)
                
                while True:
                    samples, read = source()
                    if read == 0:
                        break
                    audio_buffer = np.append(audio_buffer, samples)
                    total_frames += read
                    
                return audio_buffer, sample_rate
                
            elif SOUNDFILE_AVAILABLE:
                # Fallback to soundfile
                audio_data, sample_rate = sf.read(filepath, dtype='float32')
                if len(audio_data.shape) > 1:  # Convert stereo to mono
                    audio_data = np.mean(audio_data, axis=1)
                return audio_data, sample_rate
                
            elif LIBROSA_AVAILABLE:
                # Last resort fallback to librosa
                audio_data, sample_rate = librosa.load(filepath, sr=DEFAULT_SAMPLE_RATE)
                return audio_data, sample_rate
                
            else:
                print(f"Error: No audio loading libraries available")
                self.load_errors += 1
                return None, DEFAULT_SAMPLE_RATE
                
        except Exception as e:
            print(f"Error loading audio file {filepath}: {str(e)}")
            traceback.print_exc()
            self.load_errors += 1
            return None, DEFAULT_SAMPLE_RATE
    
    def detect_bpm(self, filepath: str) -> Dict[str, Any]:
        """
        Detect BPM (tempo) of an audio file using aubio.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            Dictionary with BPM and confidence information
        """
        result = {
            'bpm': None,
            'confidence': 0.0,
            'success': False,
            'error': None
        }
        
        if not os.path.exists(filepath):
            result['error'] = f"File not found: {filepath}"
            return result
            
        try:
            # Load audio data
            audio_data, sample_rate = self.load_audio(filepath)
            if audio_data is None:
                result['error'] = "Failed to load audio data"
                return result
                
            if AUBIO_AVAILABLE:
                # Use aubio's tempo detection
                win_s = DEFAULT_WINDOW_SIZE
                hop_s = DEFAULT_HOP_SIZE
                
                tempo = aubio.tempo(DEFAULT_TEMPO_METHOD, win_s, hop_s, sample_rate)
                
                # Process the audio in chunks
                beats = []
                total_frames = len(audio_data)
                
                # Process in windows of hop_size
                for i in range(0, total_frames - hop_s, hop_s):
                    chunk = audio_data[i:i+hop_s]
                    if len(chunk) < hop_s:
                        chunk = np.pad(chunk, (0, hop_s - len(chunk)))
                    
                    is_beat = tempo(chunk.astype(np.float32))
                    if is_beat:
                        beats.append(i / sample_rate)
                
                # Get the BPM value
                bpm = tempo.get_bpm()
                
                # Calculate confidence based on beat strength and consistency
                if len(beats) > 2:
                    # Calculate inter-beat intervals and their consistency
                    ibis = np.diff(beats)
                    # Use the coefficient of variation as a measure of stability (lower is better)
                    cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
                    confidence = max(0, min(1, 1.0 - cv))
                else:
                    confidence = 0.4  # Low confidence if few beats detected
                
                # Update with results
                result.update({
                    'bpm': round(float(bpm), 2),
                    'confidence': float(confidence),
                    'beats': len(beats),
                    'beat_times': beats[:10],  # Just the first few for diagnostic purposes
                    'success': True
                })
                
            elif LIBROSA_AVAILABLE:
                # Fallback to librosa
                print("Falling back to librosa for BPM detection")
                tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
                
                # Calculate confidence similar to above
                if len(beat_frames) > 4:
                    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
                    ibis = np.diff(beat_times)
                    cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
                    confidence = max(0, min(1, 1.0 - cv))
                else:
                    confidence = 0.4
                
                result.update({
                    'bpm': round(float(tempo), 2),
                    'confidence': float(confidence),
                    'success': True,
                    'using_fallback': True
                })
            else:
                result['error'] = "No BPM detection libraries available"
                
            return result
            
        except Exception as e:
            traceback.print_exc()
            result['error'] = str(e)
            self.analysis_errors += 1
            return result
    
    def detect_key(self, filepath: str) -> Dict[str, Any]:
        """
        Detect musical key of an audio file using aubio.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            Dictionary with key and confidence information
        """
        result = {
            'key': None,
            'camelot_key': None,
            'confidence': 0.0,
            'success': False,
            'error': None
        }
        
        if not os.path.exists(filepath):
            result['error'] = f"File not found: {filepath}"
            return result
            
        try:
            # Load audio data
            audio_data, sample_rate = self.load_audio(filepath)
            if audio_data is None:
                result['error'] = "Failed to load audio data"
                return result
            
            if AUBIO_AVAILABLE:
                # Use aubio's pitch detection and chromagram for key detection
                win_s = DEFAULT_WINDOW_SIZE
                hop_s = DEFAULT_HOP_SIZE
                
                # Create a chromagram
                notes_o = aubio.notes(hop_size=hop_s, buf_size=win_s, samplerate=sample_rate)
                
                # Create a filter for the notes
                chroma = np.zeros(12)
                
                # Process the audio in chunks to extract notes
                total_frames = len(audio_data)
                notes = []
                
                # Process in windows of hop_size
                for i in range(0, total_frames - hop_s, hop_s):
                    chunk = audio_data[i:i+hop_s]
                    if len(chunk) < hop_s:
                        chunk = np.pad(chunk, (0, hop_s - len(chunk)))
                    
                    new_note = notes_o(chunk.astype(np.float32))
                    if new_note[0] != 0:  # If a note was detected
                        # Convert frequency to MIDI note
                        midi_note = aubio.freqtomidi(new_note[0])
                        # Add to chromagram
                        chroma[int(midi_note) % 12] += new_note[1]  # add velocity
                        notes.append(midi_note)
                
                # Normalize chromagram
                if np.sum(chroma) > 0:
                    chroma = chroma / np.sum(chroma)
                
                # Correlate with key profiles
                major_corr = np.zeros(12)
                minor_corr = np.zeros(12)
                
                for i in range(12):
                    # Shift key profile to match the pitch class
                    shifted_major = np.roll(KEY_PROFILES['major'], i)
                    shifted_minor = np.roll(KEY_PROFILES['minor'], i)
                    
                    # Calculate correlation
                    major_corr[i] = np.corrcoef(chroma, shifted_major)[0, 1]
                    minor_corr[i] = np.corrcoef(chroma, shifted_minor)[0, 1]
                
                # Determine key and scale
                is_major = np.max(major_corr) >= np.max(minor_corr)
                
                if is_major:
                    key_idx = int(np.argmax(major_corr))
                    key_format = PITCH_CLASSES[key_idx]
                    confidence = float(np.max(major_corr))
                else:
                    key_idx = int(np.argmax(minor_corr))
                    key_format = PITCH_CLASSES[key_idx] + 'm'
                    confidence = float(np.max(minor_corr))
                
                # Normalize confidence to [0,1] range
                confidence = max(0, min(1, (confidence + 1) / 2))
                
                # Get Camelot notation
                camelot = CAMELOT_WHEEL.get(key_format, 'Unknown')
                
                result.update({
                    'key': key_format,
                    'camelot_key': camelot,
                    'confidence': confidence,
                    'is_major': bool(is_major),
                    'chroma_values': chroma.tolist(),
                    'success': True
                })
                
            elif LIBROSA_AVAILABLE:
                # Fallback to librosa
                print("Falling back to librosa for key detection")
                
                # Extract chroma features
                chromagram = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
                chroma = np.mean(chromagram, axis=1)
                
                # Calculate correlation with key profiles
                major_corr = np.zeros(12)
                minor_corr = np.zeros(12)
                
                for i in range(12):
                    # Shift key profile to match the pitch class
                    shifted_major = np.roll(KEY_PROFILES['major'], i)
                    shifted_minor = np.roll(KEY_PROFILES['minor'], i)
                    
                    # Calculate correlation
                    major_corr[i] = np.corrcoef(chroma, shifted_major)[0, 1]
                    minor_corr[i] = np.corrcoef(chroma, shifted_minor)[0, 1]
                
                # Determine key and scale
                is_major = np.max(major_corr) >= np.max(minor_corr)
                
                if is_major:
                    key_idx = int(np.argmax(major_corr))
                    key_format = PITCH_CLASSES[key_idx]
                    confidence = float(np.max(major_corr))
                else:
                    key_idx = int(np.argmax(minor_corr))
                    key_format = PITCH_CLASSES[key_idx] + 'm'
                    confidence = float(np.max(minor_corr))
                
                # Normalize confidence to [0,1] range
                confidence = max(0, min(1, (confidence + 1) / 2))
                
                # Get Camelot notation
                camelot = CAMELOT_WHEEL.get(key_format, 'Unknown')
                
                result.update({
                    'key': key_format,
                    'camelot_key': camelot,
                    'confidence': confidence,
                    'is_major': bool(is_major),
                    'success': True,
                    'using_fallback': True
                })
            else:
                result['error'] = "No key detection libraries available"
                
            return result
            
        except Exception as e:
            traceback.print_exc()
            result['error'] = str(e)
            self.analysis_errors += 1
            return result
    
    def calculate_energy(self, filepath: str) -> Dict[str, Any]:
        """
        Calculate energy rating of a track for DJ applications.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            Dictionary with energy rating and characteristics
        """
        result = {
            'energy_score': None,
            'characteristics': {},
            'success': False,
            'error': None
        }
        
        if not os.path.exists(filepath):
            result['error'] = f"File not found: {filepath}"
            return result
            
        try:
            # Load audio data
            audio_data, sample_rate = self.load_audio(filepath)
            if audio_data is None:
                result['error'] = "Failed to load audio data"
                return result
                
            if AUBIO_AVAILABLE:
                # Use aubio to extract features for energy calculation
                win_s = DEFAULT_WINDOW_SIZE
                hop_s = DEFAULT_HOP_SIZE
                
                # Extract onset strength
                onset = aubio.onset("energy", win_s, hop_s, sample_rate)
                
                # Process the audio in chunks
                total_frames = len(audio_data)
                onset_values = []
                
                # Process in windows of hop_size
                for i in range(0, total_frames - hop_s, hop_s):
                    chunk = audio_data[i:i+hop_s]
                    if len(chunk) < hop_s:
                        chunk = np.pad(chunk, (0, hop_s - len(chunk)))
                    
                    onset_val = onset(chunk.astype(np.float32))
                    onset_values.append(float(onset_val))
                
                # Calculate RMS energy
                n_frames = len(onset_values)
                frame_duration = hop_s / sample_rate
                total_duration = n_frames * frame_duration
                
                # Divide into sections for dynamics analysis
                if total_duration > 30:  # Only for tracks longer than 30 seconds
                    n_sections = int(total_duration / 10)  # 10-second sections
                    section_frames = int(n_frames / n_sections)
                    section_energies = []
                    
                    for i in range(n_sections):
                        start = i * section_frames
                        end = start + section_frames
                        section = onset_values[start:end]
                        section_energies.append(np.mean(section))
                    
                    # Calculate energy variance (for build-ups/drops)
                    energy_variance = np.var(section_energies)
                    
                    # Calculate dynamics (difference between max and min sections)
                    if len(section_energies) > 0:
                        dynamics_range = max(section_energies) - min(section_energies)
                    else:
                        dynamics_range = 0
                else:
                    energy_variance = 0
                    dynamics_range = 0
                    section_energies = []
                
                # Calculate overall metrics
                mean_energy = np.mean(onset_values) if onset_values else 0
                peak_energy = np.max(onset_values) if onset_values else 0
                energy_variance_overall = np.var(onset_values) if onset_values else 0
                
                # Normalize values
                normalized_mean = min(1.0, mean_energy / 0.5)  # 0.5 is a typical threshold
                normalized_variance = min(1.0, energy_variance_overall / 0.1)
                normalized_dynamics = min(1.0, dynamics_range / 0.3)
                
                # Calculate overall energy score (1-10 scale)
                energy_score = 1 + 9 * (
                    0.5 * normalized_mean + 
                    0.3 * normalized_variance +
                    0.2 * normalized_dynamics
                )
                
                # Compile characteristics
                characteristics = {
                    'mean_energy': float(mean_energy),
                    'peak_energy': float(peak_energy),
                    'energy_variance': float(energy_variance_overall),
                    'dynamics_range': float(dynamics_range),
                    'build_drop_score': float(min(10, energy_variance * 20)),
                    'section_profile': [float(e) for e in section_energies[:10]]  # Just first 10 for brevity
                }
                
                result.update({
                    'energy_score': round(float(energy_score), 1),
                    'characteristics': characteristics,
                    'success': True
                })
                
            elif LIBROSA_AVAILABLE:
                # Fallback to librosa
                print("Falling back to librosa for energy calculation")
                
                # Extract features
                rmse = librosa.feature.rms(y=audio_data)[0]
                onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
                
                # Calculate energy metrics
                mean_energy = np.mean(rmse)
                peak_energy = np.max(rmse)
                energy_variance = np.var(rmse)
                
                # Calculate dynamics
                if len(rmse) > 10:
                    # Split into 10 sections
                    section_length = len(rmse) // 10
                    sections = [rmse[i*section_length:(i+1)*section_length] for i in range(10)]
                    section_energies = [np.mean(section) for section in sections]
                    dynamics_range = max(section_energies) - min(section_energies)
                else:
                    dynamics_range = 0
                    section_energies = []
                
                # Calculate overall energy score (1-10 scale)
                energy_score = 1 + 9 * (
                    min(1.0, mean_energy / 0.2) * 0.5 +  # Different scale for librosa
                    min(1.0, energy_variance / 0.05) * 0.3 +
                    min(1.0, dynamics_range / 0.15) * 0.2
                )
                
                # Compile characteristics
                characteristics = {
                    'mean_energy': float(mean_energy),
                    'peak_energy': float(peak_energy),
                    'energy_variance': float(energy_variance),
                    'dynamics_range': float(dynamics_range),
                    'section_profile': [float(e) for e in section_energies]
                }
                
                result.update({
                    'energy_score': round(float(energy_score), 1),
                    'characteristics': characteristics,
                    'success': True,
                    'using_fallback': True
                })
            else:
                result['error'] = "No energy analysis libraries available"
                
            return result
            
        except Exception as e:
            traceback.print_exc()
            result['error'] = str(e)
            self.analysis_errors += 1
            return result
    
    def detect_cue_points(self, filepath: str, output_file: str = None) -> Dict[str, Any]:
        """
        Detect ideal cue points for DJ mixing using aubio.
        
        Args:
            filepath: Path to the audio file
            output_file: Optional path for saving cue points
            
        Returns:
            Dictionary with cue points information
        """
        result = {
            'cue_points': [],
            'sections': [],
            'success': False,
            'error': None
        }
        
        if not os.path.exists(filepath):
            result['error'] = f"File not found: {filepath}"
            return result
            
        try:
            print("   👁️ Analyzing track structure for cue points (isolated)...")
            
            # Load audio data
            audio_data, sample_rate = self.load_audio(filepath)
            if audio_data is None:
                result['error'] = "Failed to load audio data"
                return result
            
            if AUBIO_AVAILABLE:
                # Use aubio for beat and onset detection
                hop_size = DEFAULT_HOP_SIZE
                win_size = DEFAULT_WINDOW_SIZE
                
                # Tempo detection
                tempo = aubio.tempo('default', win_size, hop_size, sample_rate)
                beats = []
                
                # Onset detection
                onset = aubio.onset('complex', win_size, hop_size, sample_rate)
                onsets = []
                
                # Process audio in chunks
                total_frames = len(audio_data)
                total_seconds = total_frames / sample_rate
                
                # Process in windows of hop_size
                for i in range(0, total_frames - hop_size, hop_size):
                    chunk = audio_data[i:i+hop_size]
                    if len(chunk) < hop_size:
                        chunk = np.pad(chunk, (0, hop_size - len(chunk)))
                    
                    # Detect beats
                    is_beat = tempo(chunk.astype(np.float32))
                    if is_beat:
                        beat_time = i / sample_rate
                        beats.append(beat_time)
                    
                    # Detect onsets
                    is_onset = onset(chunk.astype(np.float32))
                    if is_onset:
                        onset_time = i / sample_rate
                        onsets.append(onset_time)
                
                # Get the BPM value
                bpm = tempo.get_bpm()
                print(f"   ⏰ Detected tempo: {bpm:.1f} BPM")
                
                # Calculate bars (4 beats per bar typically)
                bar_duration = 4 * 60 / bpm if bpm > 0 else 4
                
                # Calculate metrics for energy levels in different sections
                # This helps identify significant changes in the track
                section_duration = 4 * bar_duration  # Analyze 4 bars at a time
                num_sections = int(total_seconds / section_duration)
                section_energies = []
                
                # Calculate energy for each section
                for i in range(num_sections):
                    start_sec = i * section_duration
                    end_sec = min((i + 1) * section_duration, total_seconds)
                    
                    start_frame = int(start_sec * sample_rate)
                    end_frame = int(end_sec * sample_rate)
                    
                    section_audio = audio_data[start_frame:end_frame]
                    if len(section_audio) > 0:
                        section_energy = np.sqrt(np.mean(section_audio ** 2))
                        section_energies.append({
                            'start': start_sec,
                            'end': end_sec,
                            'energy': float(section_energy)
                        })
                
                # Find significant changes in energy between adjacent sections
                energy_changes = []
                for i in range(1, len(section_energies)):
                    prev_energy = section_energies[i-1]['energy']
                    curr_energy = section_energies[i]['energy']
                    
                    change_ratio = curr_energy / prev_energy if prev_energy > 0 else 1
                    
                    if change_ratio > 1.4 or change_ratio < 0.7:  # Significant change (up or down)
                        energy_changes.append({
                            'position': section_energies[i]['start'],
                            'ratio': float(change_ratio),
                            'type': 'increase' if change_ratio > 1 else 'decrease'
                        })
                
                # Find potential sections (intro, verse, chorus, outro)
                intro_end = None
                main_drop = None
                outro_start = None
                sections = []
                
                if energy_changes and beats:
                    # Find intro end - first significant increase in energy after 16 bars
                    min_intro_time = 16 * bar_duration
                    for change in energy_changes:
                        if change['position'] > min_intro_time and change['ratio'] > 1.4:
                            # Align to nearest beat
                            nearest_beat = min(beats, key=lambda x: abs(x - change['position']))
                            intro_end = nearest_beat
                            sections.append({
                                'position': intro_end,
                                'type': 'intro_end',
                                'description': 'End of intro'
                            })
                            break
                    
                    # Find main drop - largest energy increase
                    max_increase = max([c for c in energy_changes if c['type'] == 'increase'], 
                                      key=lambda x: x['ratio'], 
                                      default=None)
                    if max_increase:
                        # Align to nearest beat
                        nearest_beat = min(beats, key=lambda x: abs(x - max_increase['position']))
                        main_drop = nearest_beat
                        sections.append({
                            'position': main_drop,
                            'type': 'main_drop',
                            'description': 'Main drop'
                        })
                    
                    # Find outro start - last significant decrease in energy
                    min_outro_time = total_seconds - 32 * bar_duration
                    for change in reversed(energy_changes):
                        if change['position'] > min_outro_time and change['ratio'] < 0.7:
                            # Align to nearest beat
                            nearest_beat = min(beats, key=lambda x: abs(x - change['position']))
                            outro_start = nearest_beat
                            sections.append({
                                'position': outro_start,
                                'type': 'outro_start',
                                'description': 'Start of outro'
                            })
                            break
                
                # Generate cue points
                cue_points = []
                
                # Add intro cue
                cue_points.append({
                    'position': 0.0,
                    'type': 'intro',
                    'description': 'Start of track'
                })
                
                # Add significant cue points based on section analysis
                for section in sections:
                    cue_points.append({
                        'position': section['position'],
                        'type': section['type'],
                        'description': section['description']
                    })
                
                # Find additional mix points (every 16 bars starting after intro)
                if beats and len(beats) > 16:
                    beats_per_bar = 4
                    bars_per_mix = 16
                    beats_per_mix = beats_per_bar * bars_per_mix
                    
                    # Start from intro end or a reasonable position
                    start_beat_idx = 0
                    if intro_end is not None:
                        # Find beat index closest to intro end
                        start_beat_idx = next((i for i, beat in enumerate(beats) if beat >= intro_end), 0)
                    
                    # Add mix points every 16 bars
                    for i in range(start_beat_idx, len(beats), beats_per_mix):
                        if i < len(beats):
                            # Skip if too close to an existing cue point
                            too_close = False
                            for cue in cue_points:
                                if abs(beats[i] - cue['position']) < bar_duration:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                cue_points.append({
                                    'position': beats[i],
                                    'type': 'mix',
                                    'description': f'Mix point ({i // beats_per_bar} bars)'
                                })
                
                # Sort cue points by position
                cue_points.sort(key=lambda x: x['position'])
                
                # Write cue points to output file if specified
                if output_file and cue_points:
                    try:
                        with open(output_file, 'w') as f:
                            json.dump(cue_points, f, indent=2)
                        print(f"   ✅ Cue points saved to {output_file}")
                    except Exception as e:
                        print(f"   ⚠️ Failed to save cue points: {str(e)}")
                
                result.update({
                    'bpm': float(bpm),
                    'total_duration': float(total_seconds),
                    'bar_duration': float(bar_duration),
                    'num_beats': len(beats),
                    'num_sections': len(sections),
                    'cue_points': cue_points,
                    'sections': sections,
                    'success': True
                })
                return result
                
            # If we get here, aubio processing wasn't available
            result['error'] = "Aubio not available for cue point detection"
            return result
            
        except Exception as e:
            result['error'] = f"Error detecting cue points: {str(e)}"
            traceback.print_exc()
            self.analysis_errors += 1
            return result
    
    def process_request(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request from the client.
        
        Args:
            command: The command to execute
            params: Parameters for the command
            
        Returns:
            Dictionary with results
        """
        result = {
            'success': False, 
            'error': None,
            'command': command
        }
        
        try:
            if command == 'ping':
                result['success'] = True
                result['response'] = 'pong'
                result['libraries'] = {
                    'aubio': AUBIO_AVAILABLE,
                    'librosa': LIBROSA_AVAILABLE,
                    'soundfile': SOUNDFILE_AVAILABLE
                }
                
            elif command == 'detect_bpm':
                if 'filepath' not in params:
                    result['error'] = "Missing required parameter: filepath"
                else:
                    bpm_result = self.detect_bpm(params['filepath'])
                    result.update(bpm_result)
                    
            elif command == 'detect_key':
                if 'filepath' not in params:
                    result['error'] = "Missing required parameter: filepath"
                else:
                    key_result = self.detect_key(params['filepath'])
                    result.update(key_result)
                    
            elif command == 'calculate_energy':
                if 'filepath' not in params:
                    result['error'] = "Missing required parameter: filepath"
                else:
                    energy_result = self.calculate_energy(params['filepath'])
                    result.update(energy_result)
                    
            elif command == 'detect_cue_points':
                if 'filepath' not in params:
                    result['error'] = "Missing required parameter: filepath"
                else:
                    output_file = params.get('output_file')
                    cue_result = self.detect_cue_points(params['filepath'], output_file)
                    result.update(cue_result)
                    
            else:
                result['error'] = f"Unknown command: {command}"
                
            return result
            
        except Exception as e:
            traceback.print_exc()
            result['error'] = str(e)
            return result


def audio_analysis_worker(conn):
    """
    Worker function for the audio analysis service.
    
    Args:
        conn: Connection pipe to communicate with the client
    """
    service = AudioAnalysisService()
    
    try:
        # Send ready signal
        conn.send({'status': 'ready'})
        
        # Process commands
        while True:
            if not conn.poll(1):  # Check for messages, timeout after 1 second
                continue
                
            try:
                message = conn.recv()
                
                if message.get('command') == 'exit':
                    break
                    
                # Process the command
                command = message.get('command', '')
                params = message.get('params', {})
                
                result = service.process_request(command, params)
                
                # Send the result back
                conn.send(result)
                
            except EOFError:
                # Parent process has closed the connection
                break
            except Exception as e:
                traceback.print_exc()
                conn.send({
                    'success': False,
                    'error': str(e),
                    'command': message.get('command', 'unknown')
                })
                
    except Exception as e:
        traceback.print_exc()
        
    finally:
        # Clean up
        conn.close()
        print("Audio analysis worker exiting")


def start_audio_analysis_service():
    """
    Start the audio analysis service in a separate process.
    
    Returns:
        Tuple of (process, connection)
    """
    parent_conn, child_conn = Pipe()
    process = Process(
        target=audio_analysis_worker,
        args=(child_conn,),
        daemon=True
    )
    process.start()
    
    # Wait for ready signal
    if parent_conn.poll(5):  # Wait up to 5 seconds for ready signal
        response = parent_conn.recv()
        if response.get('status') == 'ready':
            return process, parent_conn
    
    # If we get here, something went wrong
    process.terminate()
    raise RuntimeError("Failed to start audio analysis service")


if __name__ == "__main__":
    # Simple test if run directly
    service = AudioAnalysisService()
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        
        if os.path.exists(filepath):
            print(f"Testing BPM detection on {filepath}")
            bpm_result = service.detect_bpm(filepath)
            print(json.dumps(bpm_result, indent=2))
            
            print(f"\nTesting key detection on {filepath}")
            key_result = service.detect_key(filepath)
            print(json.dumps(key_result, indent=2))
            
            print(f"\nTesting energy calculation on {filepath}")
            energy_result = service.calculate_energy(filepath)
            print(json.dumps(energy_result, indent=2))
        else:
            print(f"File not found: {filepath}")
    else:
        print("Usage: python audio_analysis_service.py <audio_file>")
