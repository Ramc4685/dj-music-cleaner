"""
Unified Audio Analysis Service

This service consolidates audio analysis functionality from:
1. The original monolithic implementation
2. The process-isolated dj_analysis_service.py
3. The evolved service architecture

Provides stable, multiprocessing-safe audio analysis with aubio as primary engine.
"""

import os
import sys
import json
import time
import traceback
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from multiprocessing import Process, Pipe, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import asdict

# Audio processing imports
try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False
    aubio = None

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Fallback audio loading
try:
    from .librosa_compatibility import librosa_compat as librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from ..core.models import TrackMetadata
from ..core.exceptions import AudioAnalysisError
from ..utils.text import standardize_genre
from ..utils.logging_config import get_logger, get_app_logger


# Musical analysis constants
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles
KEY_PROFILES = {
    'major': np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    'minor': np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
}

# Camelot wheel mapping
CAMELOT_WHEEL = {
    'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B', 'B': '1B',
    'F#': '2B', 'C#': '3B', 'G#': '4B', 'D#': '5B', 'A#': '6B', 'F': '7B',
    'Am': '8A', 'Em': '9A', 'Bm': '10A', 'F#m': '11A', 'C#m': '12A', 'G#m': '1A',
    'D#m': '2A', 'A#m': '3A', 'Fm': '4A', 'Cm': '5A', 'Gm': '6A', 'Dm': '7A'
}

# Default analysis parameters
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_HOP_SIZE = 512
DEFAULT_WINDOW_SIZE = 2048


class UnifiedAudioAnalysisService:
    """
    Unified audio analysis service combining all previous implementations
    
    Features:
    - Stable aubio-based analysis (primary)
    - Librosa fallback support
    - Multiprocessing safety
    - Comprehensive error handling
    - Professional DJ-focused features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified audio analysis service"""
        self.config = config or {}
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Service configuration
        self.prefer_aubio = self.config.get('prefer_aubio', True)
        self.fallback_to_librosa = self.config.get('fallback_to_librosa', True)
        self.analysis_timeout = self.config.get('analysis_timeout', 30)
        self.sample_rate = self.config.get('sample_rate', DEFAULT_SAMPLE_RATE)
        
        # Performance tracking
        self.analysis_count = 0
        self.cache_hits = 0
        self.fallback_uses = 0
        self.total_analysis_time = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Validate available libraries
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Validate available audio analysis libraries"""
        available_libs = []
        
        if AUBIO_AVAILABLE:
            available_libs.append('aubio')
        if LIBROSA_AVAILABLE:
            available_libs.append('librosa')
        if SOUNDFILE_AVAILABLE:
            available_libs.append('soundfile')
        
        if not available_libs:
            raise AudioAnalysisError(
                "No audio analysis libraries available",
                details="Please install aubio, librosa, or soundfile"
            )
        
        print(f"   🎵 Audio libraries available: {', '.join(available_libs)}")
    
    def analyze_track(self, filepath: str, analysis_options: Optional[Dict[str, bool]] = None) -> TrackMetadata:
        """
        Perform comprehensive audio analysis on a track
        
        Args:
            filepath: Path to audio file
            analysis_options: Dict specifying which analyses to perform
            
        Returns:
            TrackMetadata object with analysis results
        """
        start_time = time.time()
        
        # Default analysis options
        options = analysis_options or {
            'bpm': True,
            'key': True,
            'energy': True,
            'quality': False,  # More expensive
            'cue_points': False  # Most expensive
        }
        
        try:
            # Initialize metadata object
            metadata = TrackMetadata()
            metadata.filepath = filepath
            metadata.filename = os.path.basename(filepath)
            
            # Validate file existence and basic properties
            if not os.path.exists(filepath):
                error_msg = f"File not found: {filepath}"
                self.logger.error(error_msg)
                metadata.analysis_errors.append(error_msg)
                metadata.analysis_status = "failed_file_not_found"
                return metadata
            
            if not os.path.isfile(filepath):
                error_msg = f"Path is not a file: {filepath}"
                self.logger.error(error_msg)
                metadata.analysis_errors.append(error_msg)
                metadata.analysis_status = "failed_not_a_file"
                return metadata
            
            # Check file size (skip empty files)
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                error_msg = f"File is empty: {filepath}"
                self.logger.warning(error_msg)
                metadata.analysis_errors.append(error_msg)
                metadata.analysis_status = "failed_empty_file"
                return metadata
            
            if file_size < 1024:  # Less than 1KB
                error_msg = f"File too small ({file_size} bytes): {filepath}"
                self.logger.warning(error_msg)
                metadata.analysis_errors.append(error_msg)
                metadata.analysis_status = "failed_file_too_small"
                return metadata
            
            # Check file permissions
            if not os.access(filepath, os.R_OK):
                error_msg = f"File not readable: {filepath}"
                self.logger.error(error_msg)
                metadata.analysis_errors.append(error_msg)
                metadata.analysis_status = "failed_permission_denied"
                return metadata
            
            # Load audio data with error handling
            try:
                audio_data, sample_rate = self._load_audio(filepath)
                if audio_data is None:
                    error_msg = "Failed to load audio data"
                    self.logger.error(f"{error_msg}: {filepath}")
                    metadata.analysis_errors.append(error_msg)
                    metadata.analysis_status = "failed_audio_loading"
                    return metadata
                
                metadata.duration = len(audio_data) / sample_rate
                
            except AudioAnalysisError as e:
                error_msg = f"Audio loading error: {str(e)}"
                self.logger.error(error_msg)
                metadata.analysis_errors.append(error_msg)
                metadata.analysis_status = "failed_audio_loading"
                return metadata
            
            # Perform requested analyses
            if options.get('bpm', True):
                bpm_result = self._analyze_bpm(audio_data, sample_rate, filepath)
                metadata.bpm = bpm_result.get('bpm')
                metadata.bpm_confidence = bpm_result.get('confidence', 0.0)
            
            if options.get('key', True):
                key_result = self._analyze_key(audio_data, sample_rate, filepath)
                metadata.musical_key = key_result.get('key', '')
                metadata.key_confidence = key_result.get('confidence', 0.0)
                metadata.camelot_key = key_result.get('camelot_key', '')
            
            if options.get('energy', True):
                energy_result = self._analyze_energy(audio_data, sample_rate, filepath)
                metadata.energy_level = energy_result.get('energy_score')
            
            if options.get('quality', False):
                quality_result = self._analyze_quality(audio_data, sample_rate, filepath)
                metadata.quality_score = quality_result.get('quality_score', 0.0)
                metadata.dynamic_range = quality_result.get('dynamic_range')
                metadata.peak_level = quality_result.get('peak_level')
                metadata.rms_level = quality_result.get('rms_level')
            
            if options.get('cue_points', False):
                cue_result = self._analyze_cue_points(audio_data, sample_rate, filepath)
                metadata.cue_points = cue_result.get('cue_points', [])
            
            # Update performance metrics
            processing_time = time.time() - start_time
            metadata.processing_time = processing_time
            metadata.analysis_status = "success"
            
            with self._lock:
                self.analysis_count += 1
                self.total_analysis_time += processing_time
            
            return metadata
            
        except Exception as e:
            # Log the unexpected error and return failed metadata
            error_msg = f"Unexpected analysis error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Return a minimal metadata object with error info
            metadata = TrackMetadata()
            metadata.filepath = filepath
            metadata.filename = os.path.basename(filepath)
            metadata.analysis_errors.append(error_msg)
            metadata.analysis_status = "failed_unexpected_error"
            metadata.processing_time = time.time() - start_time
            
            return metadata
    
    def _load_audio(self, filepath: str) -> Tuple[Optional[np.ndarray], int]:
        """Load audio file using available libraries with robust error handling"""
        last_error = None
        
        # Try aubio first (most stable)
        if AUBIO_AVAILABLE:
            try:
                return self._load_with_aubio(filepath)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Aubio loading failed for {os.path.basename(filepath)}: {str(e)}")
        
        # Try soundfile as fallback
        if SOUNDFILE_AVAILABLE:
            try:
                return self._load_with_soundfile(filepath)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Soundfile loading failed for {os.path.basename(filepath)}: {str(e)}")
        
        # Try librosa compatibility layer as last resort
        if LIBROSA_AVAILABLE and self.fallback_to_librosa:
            try:
                with self._lock:
                    self.fallback_uses += 1
                return self._load_with_librosa(filepath)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Librosa loading failed for {os.path.basename(filepath)}: {str(e)}")
        
        # If all libraries failed
        if last_error:
            raise AudioAnalysisError(f"All audio loading libraries failed. Last error: {str(last_error)}", filepath=filepath)
        else:
            raise AudioAnalysisError("No audio loading library available", filepath=filepath)
    
    def _load_with_aubio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio using aubio"""
        try:
            hop_size = DEFAULT_HOP_SIZE
            source = aubio.source(filepath, self.sample_rate, hop_size)
            sample_rate = source.samplerate
            
            # Read entire file
            audio_buffer = np.zeros([0], dtype=np.float32)
            
            while True:
                samples, read = source()
                if read == 0:
                    break
                audio_buffer = np.append(audio_buffer, samples)
            
            return audio_buffer, sample_rate
            
        except Exception as e:
            error_str = str(e)
            # Check for specific file-not-found errors
            if "No such file or directory" in error_str or "Failed opening" in error_str:
                raise AudioAnalysisError(f"File not found or unreadable: {os.path.basename(filepath)}")
            else:
                raise AudioAnalysisError(f"Aubio loading failed: {error_str}")
    
    def _load_with_soundfile(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile"""
        try:
            audio_data, sample_rate = sf.read(filepath, dtype='float32')
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            return audio_data, sample_rate
            
        except Exception as e:
            raise AudioAnalysisError(f"Soundfile loading failed: {str(e)}")
    
    def _load_with_librosa(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio using librosa compatibility layer"""
        try:
            audio_data, sample_rate = librosa.load(filepath, sr=self.sample_rate)
            return audio_data, sample_rate
            
        except Exception as e:
            raise AudioAnalysisError(f"Librosa loading failed: {str(e)}")
    
    def _analyze_bpm(self, audio_data: np.ndarray, sample_rate: int, filepath: str) -> Dict[str, Any]:
        """Analyze BPM using available methods"""
        try:
            if AUBIO_AVAILABLE and self.prefer_aubio:
                return self._analyze_bpm_aubio(audio_data, sample_rate)
            elif LIBROSA_AVAILABLE and self.fallback_to_librosa:
                with self._lock:
                    self.fallback_uses += 1
                return self._analyze_bpm_librosa(audio_data, sample_rate)
            else:
                raise AudioAnalysisError("No BPM analysis library available")
                
        except Exception as e:
            raise AudioAnalysisError(
                f"BPM analysis failed: {str(e)}",
                filepath=filepath
            )
    
    def _analyze_bpm_aubio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """BPM analysis using aubio"""
        try:
            hop_size = DEFAULT_HOP_SIZE
            win_size = DEFAULT_WINDOW_SIZE
            
            # Create tempo detector
            tempo = aubio.tempo("default", win_size, hop_size, sample_rate)
            
            # Process audio in chunks
            beats = []
            total_frames = len(audio_data)
            
            for i in range(0, total_frames - hop_size, hop_size):
                chunk = audio_data[i:i+hop_size]
                if len(chunk) < hop_size:
                    chunk = np.pad(chunk, (0, hop_size - len(chunk)))
                
                is_beat = tempo(chunk.astype(np.float32))
                if is_beat:
                    beat_time = i / sample_rate
                    beats.append(beat_time)
            
            # Get BPM value
            bpm = float(tempo.get_bpm())
            
            # Calculate confidence based on beat consistency
            confidence = 0.7  # Default confidence for aubio
            if len(beats) > 4:
                # Calculate inter-beat intervals consistency
                ibis = np.diff(beats)
                if len(ibis) > 0:
                    cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
                    confidence = max(0.2, min(0.95, 1.0 - cv))
            
            return {
                'bpm': round(bpm, 1),
                'confidence': confidence,
                'beats_detected': len(beats),
                'method': 'aubio'
            }
            
        except Exception as e:
            raise AudioAnalysisError(f"Aubio BPM analysis failed: {str(e)}")
    
    def _analyze_bpm_librosa(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """BPM analysis using librosa compatibility layer"""
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            # Calculate confidence
            confidence = 0.6  # Default for librosa fallback
            if len(beat_frames) > 4:
                beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
                ibis = np.diff(beat_times)
                if len(ibis) > 0:
                    cv = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 1.0
                    confidence = max(0.2, min(0.9, 1.0 - cv))
            
            return {
                'bpm': round(float(tempo), 1),
                'confidence': confidence,
                'beats_detected': len(beat_frames),
                'method': 'librosa'
            }
            
        except Exception as e:
            raise AudioAnalysisError(f"Librosa BPM analysis failed: {str(e)}")
    
    def _analyze_key(self, audio_data: np.ndarray, sample_rate: int, filepath: str) -> Dict[str, Any]:
        """Analyze musical key using available methods"""
        try:
            if AUBIO_AVAILABLE and self.prefer_aubio:
                return self._analyze_key_aubio(audio_data, sample_rate)
            elif LIBROSA_AVAILABLE and self.fallback_to_librosa:
                with self._lock:
                    self.fallback_uses += 1
                return self._analyze_key_librosa(audio_data, sample_rate)
            else:
                raise AudioAnalysisError("No key analysis library available")
                
        except Exception as e:
            raise AudioAnalysisError(
                f"Key analysis failed: {str(e)}",
                filepath=filepath
            )
    
    def _analyze_key_aubio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Key analysis using aubio with Krumhansl-Schmuckler algorithm"""
        try:
            hop_size = DEFAULT_HOP_SIZE
            win_size = DEFAULT_WINDOW_SIZE
            
            # Create notes detector for chromagram
            notes_o = aubio.notes(hop_size=hop_size, buf_size=win_size, samplerate=sample_rate)
            
            # Build chromagram
            chroma = np.zeros(12)
            total_frames = len(audio_data)
            
            for i in range(0, total_frames - hop_size, hop_size):
                chunk = audio_data[i:i+hop_size]
                if len(chunk) < hop_size:
                    chunk = np.pad(chunk, (0, hop_size - len(chunk)))
                
                new_note = notes_o(chunk.astype(np.float32))
                if new_note[0] != 0:  # Note detected
                    midi_note = aubio.freqtomidi(new_note[0])
                    chroma[int(midi_note) % 12] += new_note[1]  # Add velocity
            
            # Normalize chromagram
            if np.sum(chroma) > 0:
                chroma = chroma / np.sum(chroma)
            else:
                return {'key': '', 'confidence': 0.0, 'camelot_key': '', 'method': 'aubio'}
            
            # Key detection using Krumhansl-Schmuckler profiles
            major_correlations = np.zeros(12)
            minor_correlations = np.zeros(12)
            
            for i in range(12):
                shifted_major = np.roll(KEY_PROFILES['major'], i)
                shifted_minor = np.roll(KEY_PROFILES['minor'], i)
                
                # Calculate correlations
                if np.std(chroma) > 0:
                    major_correlations[i] = np.corrcoef(chroma, shifted_major)[0, 1]
                    minor_correlations[i] = np.corrcoef(chroma, shifted_minor)[0, 1]
            
            # Determine key
            max_major = np.max(major_correlations)
            max_minor = np.max(minor_correlations)
            
            if max_major >= max_minor:
                key_idx = int(np.argmax(major_correlations))
                key = PITCH_CLASSES[key_idx]
                confidence = float(max_major)
            else:
                key_idx = int(np.argmax(minor_correlations))
                key = PITCH_CLASSES[key_idx] + 'm'
                confidence = float(max_minor)
            
            # Normalize confidence
            confidence = max(0, min(1, (confidence + 1) / 2))
            
            # Get Camelot notation
            camelot_key = CAMELOT_WHEEL.get(key, '')
            
            return {
                'key': key,
                'confidence': confidence,
                'camelot_key': camelot_key,
                'method': 'aubio'
            }
            
        except Exception as e:
            raise AudioAnalysisError(f"Aubio key analysis failed: {str(e)}")
    
    def _analyze_key_librosa(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Key analysis using librosa compatibility layer"""
        try:
            # Extract chroma features
            chromagram = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
            chroma = np.mean(chromagram, axis=1)
            
            # Key detection using Krumhansl-Schmuckler profiles
            major_correlations = np.zeros(12)
            minor_correlations = np.zeros(12)
            
            for i in range(12):
                shifted_major = np.roll(KEY_PROFILES['major'], i)
                shifted_minor = np.roll(KEY_PROFILES['minor'], i)
                
                major_correlations[i] = np.corrcoef(chroma, shifted_major)[0, 1]
                minor_correlations[i] = np.corrcoef(chroma, shifted_minor)[0, 1]
            
            # Determine key
            max_major = np.max(major_correlations)
            max_minor = np.max(minor_correlations)
            
            if max_major >= max_minor:
                key_idx = int(np.argmax(major_correlations))
                key = PITCH_CLASSES[key_idx]
                confidence = float(max_major)
            else:
                key_idx = int(np.argmax(minor_correlations))
                key = PITCH_CLASSES[key_idx] + 'm'
                confidence = float(max_minor)
            
            # Normalize confidence
            confidence = max(0, min(1, (confidence + 1) / 2))
            
            # Get Camelot notation
            camelot_key = CAMELOT_WHEEL.get(key, '')
            
            return {
                'key': key,
                'confidence': confidence,
                'camelot_key': camelot_key,
                'method': 'librosa'
            }
            
        except Exception as e:
            raise AudioAnalysisError(f"Librosa key analysis failed: {str(e)}")
    
    def _analyze_energy(self, audio_data: np.ndarray, sample_rate: int, filepath: str) -> Dict[str, Any]:
        """Analyze track energy level"""
        try:
            if AUBIO_AVAILABLE and self.prefer_aubio:
                return self._analyze_energy_aubio(audio_data, sample_rate)
            else:
                return self._analyze_energy_basic(audio_data, sample_rate)
                
        except Exception as e:
            raise AudioAnalysisError(
                f"Energy analysis failed: {str(e)}",
                filepath=filepath
            )
    
    def _analyze_energy_aubio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Energy analysis using aubio"""
        try:
            hop_size = DEFAULT_HOP_SIZE
            win_size = DEFAULT_WINDOW_SIZE
            
            # Create onset detector for energy analysis
            onset = aubio.onset("energy", win_size, hop_size, sample_rate)
            
            # Process audio
            total_frames = len(audio_data)
            onset_values = []
            
            for i in range(0, total_frames - hop_size, hop_size):
                chunk = audio_data[i:i+hop_size]
                if len(chunk) < hop_size:
                    chunk = np.pad(chunk, (0, hop_size - len(chunk)))
                
                onset_val = onset(chunk.astype(np.float32))
                onset_values.append(float(onset_val))
            
            # Calculate energy metrics
            mean_energy = np.mean(onset_values) if onset_values else 0
            peak_energy = np.max(onset_values) if onset_values else 0
            energy_variance = np.var(onset_values) if onset_values else 0
            
            # Calculate energy score (1-10 scale)
            energy_score = 1 + 9 * min(1.0, mean_energy / 0.5)
            
            return {
                'energy_score': round(float(energy_score), 1),
                'mean_energy': float(mean_energy),
                'peak_energy': float(peak_energy),
                'energy_variance': float(energy_variance),
                'method': 'aubio'
            }
            
        except Exception as e:
            raise AudioAnalysisError(f"Aubio energy analysis failed: {str(e)}")
    
    def _analyze_energy_basic(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Basic energy analysis using RMS"""
        try:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Calculate spectral energy (simple approach)
            fft = np.fft.fft(audio_data)
            spectral_energy = np.mean(np.abs(fft))
            
            # Combine for energy score
            energy_score = 1 + 9 * min(1.0, (rms * 2 + spectral_energy / 1000) / 2)
            
            return {
                'energy_score': round(float(energy_score), 1),
                'rms_energy': float(rms),
                'spectral_energy': float(spectral_energy),
                'method': 'basic'
            }
            
        except Exception as e:
            raise AudioAnalysisError(f"Basic energy analysis failed: {str(e)}")
    
    def _analyze_quality(self, audio_data: np.ndarray, sample_rate: int, filepath: str) -> Dict[str, Any]:
        """Analyze audio quality metrics"""
        try:
            # RMS level
            rms_level = float(np.sqrt(np.mean(audio_data ** 2)))
            
            # Peak level
            peak_level = float(np.max(np.abs(audio_data)))
            
            # Dynamic range (simplified)
            # Split audio into segments and calculate range
            segment_length = sample_rate // 2  # 0.5 second segments
            if len(audio_data) > segment_length:
                segments = [audio_data[i:i+segment_length] for i in range(0, len(audio_data), segment_length)]
                segment_rms = [np.sqrt(np.mean(seg ** 2)) for seg in segments if len(seg) > 0]
                
                if len(segment_rms) > 1:
                    dynamic_range = float(np.max(segment_rms) - np.min(segment_rms))
                else:
                    dynamic_range = 0.0
            else:
                dynamic_range = 0.0
            
            # Quality score (0-10 based on dynamic range and levels)
            if dynamic_range > 0.3:
                quality_score = 8.0  # High dynamic range
            elif dynamic_range > 0.15:
                quality_score = 6.0  # Medium dynamic range
            elif dynamic_range > 0.05:
                quality_score = 4.0  # Low dynamic range
            else:
                quality_score = 2.0  # Very compressed
            
            # Penalize clipping
            if peak_level > 0.98:
                quality_score *= 0.7
            
            return {
                'quality_score': round(quality_score, 1),
                'dynamic_range': round(dynamic_range, 3),
                'peak_level': round(peak_level, 3),
                'rms_level': round(rms_level, 3)
            }
            
        except Exception as e:
            raise AudioAnalysisError(f"Quality analysis failed: {str(e)}")
    
    def _analyze_cue_points(self, audio_data: np.ndarray, sample_rate: int, filepath: str) -> Dict[str, Any]:
        """Analyze and suggest cue points for DJ use"""
        try:
            # This is a simplified cue point detection
            # In a full implementation, this would use the advanced cue detection service
            
            duration = len(audio_data) / sample_rate
            cue_points = []
            
            # Add start cue
            cue_points.append({
                'position': 0.0,
                'type': 'intro',
                'description': 'Start of track'
            })
            
            # Add potential mix points (every 16 bars at 128 BPM)
            bar_duration = 60.0 / 128.0 * 4  # 4 beats per bar at 128 BPM
            mix_interval = bar_duration * 16  # 16 bars
            
            position = mix_interval
            while position < duration - mix_interval:
                cue_points.append({
                    'position': round(position, 1),
                    'type': 'mix',
                    'description': f'Mix point at {position:.1f}s'
                })
                position += mix_interval
            
            return {
                'cue_points': cue_points,
                'total_duration': round(duration, 1)
            }
            
        except Exception as e:
            raise AudioAnalysisError(f"Cue point analysis failed: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the service"""
        with self._lock:
            avg_time = self.total_analysis_time / self.analysis_count if self.analysis_count > 0 else 0
            
            return {
                'total_analyses': self.analysis_count,
                'cache_hits': self.cache_hits,
                'fallback_uses': self.fallback_uses,
                'total_analysis_time': round(self.total_analysis_time, 2),
                'average_analysis_time': round(avg_time, 2),
                'aubio_available': AUBIO_AVAILABLE,
                'librosa_available': LIBROSA_AVAILABLE,
                'soundfile_available': SOUNDFILE_AVAILABLE
            }
    
    def reset_stats(self):
        """Reset performance statistics"""
        with self._lock:
            self.analysis_count = 0
            self.cache_hits = 0
            self.fallback_uses = 0
            self.total_analysis_time = 0.0


# Process isolation support for multiprocessing safety
def isolated_analysis_worker(queue_in: Queue, queue_out: Queue, config: Dict[str, Any]):
    """
    Worker function for process-isolated audio analysis
    
    This provides the same isolation benefits as the original dj_analysis_service.py
    but integrated into the unified architecture.
    """
    try:
        service = UnifiedAudioAnalysisService(config)
        
        # Signal ready
        queue_out.put({'status': 'ready'})
        
        while True:
            try:
                # Get work item
                work_item = queue_in.get(timeout=1)
                
                if work_item.get('command') == 'exit':
                    break
                
                # Process analysis request
                filepath = work_item.get('filepath')
                options = work_item.get('options', {})
                
                if filepath:
                    result = service.analyze_track(filepath, options)
                    response = {
                        'success': True,
                        'result': asdict(result)
                    }
                else:
                    response = {
                        'success': False,
                        'error': 'No filepath provided'
                    }
                
                queue_out.put(response)
                
            except Exception as e:
                error_response = {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                queue_out.put(error_response)
                
    except Exception as e:
        # Fatal error in worker
        queue_out.put({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        })


__all__ = [
    'UnifiedAudioAnalysisService',
    'isolated_analysis_worker'
]