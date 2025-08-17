"""
Advanced Cue Detection Service

Provides intelligent cue point detection using advanced audio analysis techniques.
Builds upon the unified audio analysis service to identify optimal cue points,
hot cues, and loop points for DJ performance.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import numpy as np
    from scipy.signal import find_peaks
    from scipy import ndimage
    SCIPY_AVAILABLE = True
    ArrayType = np.ndarray
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    ArrayType = Any  # Fallback type hint

from ...core.models import TrackMetadata
from ...core.exceptions import AudioAnalysisError
from ..unified_audio_analysis import UnifiedAudioAnalysisService


@dataclass
class CuePoint:
    """Advanced cue point with confidence and metadata"""
    position_seconds: float
    position_samples: int
    cue_type: str  # 'hot_cue', 'loop_in', 'loop_out', 'intro', 'outro', 'drop', 'breakdown'
    confidence: float
    description: str
    color: str = "#FF0000"  # Default red
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CueDetectionResult:
    """Result of cue detection analysis"""
    filepath: str
    cue_points: List[CuePoint]
    analysis_time: float
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AdvancedCueDetectionService:
    """
    Advanced cue point detection service
    
    Features:
    - Intelligent hot cue placement based on musical structure
    - Loop point detection with quality scoring
    - Intro/outro boundary detection
    - Drop and breakdown identification
    - Confidence scoring for all cue points
    - Customizable detection parameters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the cue detection service"""
        self.config = config or {}
        
        # Configuration
        self.enable_hot_cues = self.config.get('enable_hot_cues', True)
        self.enable_loop_points = self.config.get('enable_loop_points', True)
        self.enable_structure_analysis = self.config.get('enable_structure_analysis', True)
        self.max_cue_points = self.config.get('max_cue_points', 8)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Analysis parameters
        self.hop_length = self.config.get('hop_length', 512)
        self.frame_length = self.config.get('frame_length', 2048)
        self.sample_rate = self.config.get('sample_rate', 44100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Dependencies
        self.audio_service = UnifiedAudioAnalysisService(config)
        
        # Performance tracking
        self.stats = {
            'tracks_analyzed': 0,
            'cue_points_detected': 0,
            'total_analysis_time': 0.0,
            'average_cue_points_per_track': 0.0
        }
    
    def detect_cue_points(self, filepath: str, audio_result: Optional[TrackMetadata] = None, options: Optional[Dict[str, Any]] = None) -> CueDetectionResult:
        """
        Detect cue points for an audio file
        
        Args:
            filepath: Path to the audio file
            audio_result: Pre-computed audio analysis (optional)
            options: Detection options
            
        Returns:
            CueDetectionResult with detected cue points
        """
        start_time = time.time()
        options = options or {}
        
        result = CueDetectionResult(
            filepath=filepath,
            cue_points=[],
            analysis_time=0.0,
            success=False
        )
        
        
        try:
            # Check for scipy availability
            if not SCIPY_AVAILABLE:
                raise AudioAnalysisError("Advanced cue detection requires scipy (pip install scipy)")
            
            # Use provided analysis or fallback to internal analysis
            if audio_result is None or not hasattr(audio_result, 'audio_features'):
                # Fallback: get basic audio analysis
                audio_result = self.audio_service.analyze_track(filepath, {
                    'bpm': True,
                    'energy': True,
                    'spectral': True,
                    'onset_detection': True,
                    'beat_tracking': True
                })
                
                if not audio_result or not hasattr(audio_result, 'audio_features'):
                    raise AudioAnalysisError("Failed to get audio analysis")
            
            # Extract audio features for cue detection
            audio_data = audio_result.audio_features.get('audio_data')
            beats = audio_result.audio_features.get('beats', [])
            onsets = audio_result.audio_features.get('onsets', [])
            energy_profile = audio_result.audio_features.get('energy_profile', [])
            spectral_features = audio_result.audio_features.get('spectral_features', {})
            
            if audio_data is None:
                raise AudioAnalysisError("Audio data not available")
            
            cue_points = []
            
            # Detect different types of cue points
            if self.enable_hot_cues:
                hot_cues = self._detect_hot_cues(audio_data, beats, onsets, energy_profile)
                cue_points.extend(hot_cues)
            
            if self.enable_loop_points:
                loop_points = self._detect_loop_points(audio_data, beats, spectral_features)
                cue_points.extend(loop_points)
            
            if self.enable_structure_analysis:
                structure_cues = self._detect_structure_cues(audio_data, energy_profile, spectral_features)
                cue_points.extend(structure_cues)
            
            # Filter by confidence and limit count
            cue_points = [cp for cp in cue_points if cp.confidence >= self.confidence_threshold]
            cue_points.sort(key=lambda x: x.confidence, reverse=True)
            cue_points = cue_points[:self.max_cue_points]
            
            # Sort by position for final result
            cue_points.sort(key=lambda x: x.position_seconds)
            
            result.cue_points = cue_points
            result.success = True
            
            # Update statistics
            with self._lock:
                self.stats['tracks_analyzed'] += 1
                self.stats['cue_points_detected'] += len(cue_points)
                if self.stats['tracks_analyzed'] > 0:
                    self.stats['average_cue_points_per_track'] = (
                        self.stats['cue_points_detected'] / self.stats['tracks_analyzed']
                    )
            
            return result
            
        except Exception as e:
            result.errors.append(f"Cue detection failed: {str(e)}")
            return result
            
        finally:
            result.analysis_time = time.time() - start_time
            with self._lock:
                self.stats['total_analysis_time'] += result.analysis_time
    
    def _detect_hot_cues(self, audio_data: ArrayType, beats: List[float], 
                        onsets: List[float], energy_profile: List[float]) -> List[CuePoint]:
        """Detect optimal hot cue positions"""
        hot_cues = []
        
        if not beats or len(audio_data) == 0:
            return hot_cues
        
        try:
            # Convert to numpy arrays for easier processing
            beats = np.array(beats)
            onsets = np.array(onsets) if onsets else np.array([])
            energy = np.array(energy_profile) if energy_profile else np.array([])
            
            # Strategy 1: Find energy peaks that align with beats
            if len(energy) > 0 and len(beats) > 0:
                # Normalize energy
                energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
                
                # Find prominent energy peaks
                peaks, properties = find_peaks(energy_normalized, 
                                             prominence=0.3, 
                                             distance=int(self.sample_rate * 2))
                
                # Convert peak indices to time positions
                peak_times = peaks / self.sample_rate * len(energy_normalized) / len(audio_data)
                
                # Find nearest beats to energy peaks
                for peak_time in peak_times:
                    # Find closest beat
                    beat_distances = np.abs(beats - peak_time)
                    closest_beat_idx = np.argmin(beat_distances)
                    
                    if beat_distances[closest_beat_idx] < 1.0:  # Within 1 second
                        beat_time = beats[closest_beat_idx]
                        beat_sample = int(beat_time * self.sample_rate)
                        
                        # Calculate confidence based on energy level and beat alignment
                        energy_confidence = energy_normalized[peaks[np.argmin(np.abs(peak_times - peak_time))]]
                        alignment_confidence = 1.0 - (beat_distances[closest_beat_idx] / 1.0)
                        confidence = (energy_confidence + alignment_confidence) / 2
                        
                        hot_cues.append(CuePoint(
                            position_seconds=beat_time,
                            position_samples=beat_sample,
                            cue_type='hot_cue',
                            confidence=confidence,
                            description=f"Energy peak at {beat_time:.1f}s",
                            color="#FF4444"
                        ))
            
            # Strategy 2: Use onset detection for precise timing
            if len(onsets) > 0:
                # Find strong onsets
                onset_strengths = []  # This would come from onset detection algorithm
                
                for i, onset_time in enumerate(onsets[:10]):  # Limit to first 10 onsets
                    # Simple confidence based on position (prefer onsets not at the very beginning)
                    position_confidence = min(1.0, onset_time / 30.0)  # Ramp up over first 30 seconds
                    
                    if position_confidence > 0.3:
                        onset_sample = int(onset_time * self.sample_rate)
                        
                        hot_cues.append(CuePoint(
                            position_seconds=onset_time,
                            position_samples=onset_sample,
                            cue_type='hot_cue',
                            confidence=position_confidence,
                            description=f"Onset at {onset_time:.1f}s",
                            color="#44FF44"
                        ))
            
        except Exception as e:
            # Log error but don't fail completely
            pass
        
        return hot_cues
    
    def _detect_loop_points(self, audio_data: ArrayType, beats: List[float], 
                           spectral_features: Dict[str, Any]) -> List[CuePoint]:
        """Detect optimal loop points"""
        loop_points = []
        
        if not beats or len(beats) < 8:  # Need at least 8 beats for meaningful loops
            return loop_points
        
        try:
            beats = np.array(beats)
            
            # Look for 4, 8, 16, and 32-beat loop opportunities
            for loop_length in [4, 8, 16, 32]:
                if len(beats) < loop_length:
                    continue
                
                # Try different starting positions
                for start_idx in range(0, len(beats) - loop_length, loop_length // 2):
                    loop_start = beats[start_idx]
                    loop_end = beats[start_idx + loop_length - 1]
                    
                    # Calculate loop quality metrics
                    loop_duration = loop_end - loop_start
                    
                    # Prefer loops of certain durations (8-32 seconds)
                    duration_score = 1.0
                    if loop_duration < 8:
                        duration_score = loop_duration / 8.0
                    elif loop_duration > 32:
                        duration_score = 32.0 / loop_duration
                    
                    # Position scoring (prefer middle sections)
                    total_duration = audio_data.shape[0] / self.sample_rate
                    position_ratio = loop_start / total_duration
                    position_score = 1.0 - abs(position_ratio - 0.5)  # Prefer middle
                    
                    confidence = (duration_score + position_score) / 2
                    
                    if confidence > 0.4:
                        # Add loop in point
                        loop_points.append(CuePoint(
                            position_seconds=loop_start,
                            position_samples=int(loop_start * self.sample_rate),
                            cue_type='loop_in',
                            confidence=confidence,
                            description=f"{loop_length}-beat loop start",
                            color="#0088FF"
                        ))
                        
                        # Add loop out point
                        loop_points.append(CuePoint(
                            position_seconds=loop_end,
                            position_samples=int(loop_end * self.sample_rate),
                            cue_type='loop_out',
                            confidence=confidence,
                            description=f"{loop_length}-beat loop end",
                            color="#0044AA"
                        ))
            
        except Exception as e:
            # Log error but don't fail completely
            pass
        
        return loop_points
    
    def _detect_structure_cues(self, audio_data: ArrayType, energy_profile: List[float], 
                              spectral_features: Dict[str, Any]) -> List[CuePoint]:
        """Detect structural elements like intro, outro, drops, breakdowns"""
        structure_cues = []
        
        if len(audio_data) == 0:
            return structure_cues
        
        try:
            total_duration = audio_data.shape[0] / self.sample_rate
            energy = np.array(energy_profile) if energy_profile else np.array([])
            
            if len(energy) == 0:
                return structure_cues
            
            # Smooth energy for structure analysis
            energy_smooth = ndimage.gaussian_filter1d(energy, sigma=2.0)
            
            # Intro detection (typically lower energy at the beginning)
            intro_duration = min(30.0, total_duration * 0.15)  # First 15% or 30 seconds
            intro_samples = int(intro_duration / total_duration * len(energy_smooth))
            
            if intro_samples > 0:
                intro_energy = energy_smooth[:intro_samples]
                intro_end_idx = self._find_energy_transition(intro_energy, direction='up')
                
                if intro_end_idx is not None:
                    intro_end_time = (intro_end_idx / len(energy_smooth)) * total_duration
                    
                    structure_cues.append(CuePoint(
                        position_seconds=intro_end_time,
                        position_samples=int(intro_end_time * self.sample_rate),
                        cue_type='intro',
                        confidence=0.7,
                        description=f"Intro end at {intro_end_time:.1f}s",
                        color="#FF8800"
                    ))
            
            # Outro detection (typically lower energy at the end)
            outro_start = max(total_duration - 30.0, total_duration * 0.85)  # Last 15% or 30 seconds
            outro_start_idx = int(outro_start / total_duration * len(energy_smooth))
            
            if outro_start_idx < len(energy_smooth):
                outro_energy = energy_smooth[outro_start_idx:]
                outro_start_transition = self._find_energy_transition(outro_energy, direction='down')
                
                if outro_start_transition is not None:
                    outro_time = outro_start + (outro_start_transition / len(outro_energy)) * (total_duration - outro_start)
                    
                    structure_cues.append(CuePoint(
                        position_seconds=outro_time,
                        position_samples=int(outro_time * self.sample_rate),
                        cue_type='outro',
                        confidence=0.7,
                        description=f"Outro start at {outro_time:.1f}s",
                        color="#8800FF"
                    ))
            
            # Drop detection (significant energy increases)
            drops = self._find_drops(energy_smooth, total_duration)
            structure_cues.extend(drops)
            
            # Breakdown detection (significant energy decreases)
            breakdowns = self._find_breakdowns(energy_smooth, total_duration)
            structure_cues.extend(breakdowns)
            
        except Exception as e:
            # Log error but don't fail completely
            pass
        
        return structure_cues
    
    def _find_energy_transition(self, energy: ArrayType, direction: str) -> Optional[int]:
        """Find significant energy transitions"""
        if len(energy) < 10:
            return None
        
        # Calculate energy derivative
        energy_diff = np.diff(energy)
        
        # Smooth the derivative
        energy_diff_smooth = ndimage.gaussian_filter1d(energy_diff, sigma=1.0)
        
        if direction == 'up':
            # Find significant increases
            threshold = np.std(energy_diff_smooth) * 1.5
            candidates = np.where(energy_diff_smooth > threshold)[0]
        else:
            # Find significant decreases
            threshold = -np.std(energy_diff_smooth) * 1.5
            candidates = np.where(energy_diff_smooth < threshold)[0]
        
        return candidates[0] if len(candidates) > 0 else None
    
    def _find_drops(self, energy: ArrayType, total_duration: float) -> List[CuePoint]:
        """Find energy drops (sudden increases)"""
        drops = []
        
        try:
            # Calculate energy derivative
            energy_diff = np.diff(energy)
            
            # Find significant positive changes
            threshold = np.std(energy_diff) * 2.0
            drop_indices = np.where(energy_diff > threshold)[0]
            
            # Filter out drops too close together
            filtered_drops = []
            last_drop = -1000
            
            for drop_idx in drop_indices:
                if drop_idx - last_drop > len(energy) * 0.05:  # At least 5% of song apart
                    filtered_drops.append(drop_idx)
                    last_drop = drop_idx
            
            # Convert to cue points
            for drop_idx in filtered_drops[:3]:  # Limit to 3 drops
                drop_time = (drop_idx / len(energy)) * total_duration
                confidence = min(1.0, energy_diff[drop_idx] / threshold)
                
                drops.append(CuePoint(
                    position_seconds=drop_time,
                    position_samples=int(drop_time * self.sample_rate),
                    cue_type='drop',
                    confidence=confidence,
                    description=f"Drop at {drop_time:.1f}s",
                    color="#FF0088"
                ))
                
        except Exception as e:
            # Log error but don't fail completely
            pass
        
        return drops
    
    def _find_breakdowns(self, energy: ArrayType, total_duration: float) -> List[CuePoint]:
        """Find breakdowns (sudden decreases in energy)"""
        breakdowns = []
        
        try:
            # Calculate energy derivative
            energy_diff = np.diff(energy)
            
            # Find significant negative changes
            threshold = -np.std(energy_diff) * 2.0
            breakdown_indices = np.where(energy_diff < threshold)[0]
            
            # Filter out breakdowns too close together
            filtered_breakdowns = []
            last_breakdown = -1000
            
            for breakdown_idx in breakdown_indices:
                if breakdown_idx - last_breakdown > len(energy) * 0.05:  # At least 5% of song apart
                    filtered_breakdowns.append(breakdown_idx)
                    last_breakdown = breakdown_idx
            
            # Convert to cue points
            for breakdown_idx in filtered_breakdowns[:3]:  # Limit to 3 breakdowns
                breakdown_time = (breakdown_idx / len(energy)) * total_duration
                confidence = min(1.0, abs(energy_diff[breakdown_idx]) / abs(threshold))
                
                breakdowns.append(CuePoint(
                    position_seconds=breakdown_time,
                    position_samples=int(breakdown_time * self.sample_rate),
                    cue_type='breakdown',
                    confidence=confidence,
                    description=f"Breakdown at {breakdown_time:.1f}s",
                    color="#00FF88"
                ))
                
        except Exception as e:
            # Log error but don't fail completely
            pass
        
        return breakdowns
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        with self._lock:
            stats = self.stats.copy()
            if stats['total_analysis_time'] > 0:
                stats['average_analysis_time'] = stats['total_analysis_time'] / max(1, stats['tracks_analyzed'])
            else:
                stats['average_analysis_time'] = 0.0
            return stats


__all__ = ['AdvancedCueDetectionService', 'CuePoint', 'CueDetectionResult']