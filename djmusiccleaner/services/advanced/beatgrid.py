"""
Beat Grid Service

Provides advanced beat grid generation and analysis for precise DJ mixing.
Creates accurate beat grids with tempo variations, downbeat detection,
and bar/phrase structure analysis.
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from ...core.models import TrackMetadata
from ...core.exceptions import AudioAnalysisError
from ..unified_audio_analysis import UnifiedAudioAnalysisService


@dataclass
class BeatGridPoint:
    """Individual beat grid point"""
    position_seconds: float
    position_samples: int
    beat_number: int  # 1-4 within a bar
    bar_number: int
    tempo_bpm: float
    confidence: float
    is_downbeat: bool = False


@dataclass
class BeatGrid:
    """Complete beat grid for a track"""
    filepath: str
    beats: List[BeatGridPoint]
    average_bpm: float
    tempo_stability: float  # 0-1, how stable the tempo is
    downbeats: List[float]  # Positions of detected downbeats
    bars: List[Tuple[float, float]]  # Bar boundaries (start, end)
    phrases: List[Tuple[float, float]]  # Phrase boundaries (start, end)
    time_signature: Tuple[int, int]  # e.g., (4, 4)
    analysis_time: float
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BeatGridService:
    """
    Advanced beat grid generation service
    
    Features:
    - High-precision beat tracking
    - Tempo variation analysis
    - Downbeat detection
    - Bar and phrase structure analysis
    - Time signature detection
    - Beat grid validation and correction
    - Export to various DJ software formats
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the beat grid service"""
        self.config = config or {}
        
        # Configuration
        self.precision_mode = self.config.get('precision_mode', True)
        self.enable_downbeat_detection = self.config.get('enable_downbeat_detection', True)
        self.enable_phrase_analysis = self.config.get('enable_phrase_analysis', True)
        self.tempo_smoothing = self.config.get('tempo_smoothing', 0.1)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
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
            'beat_grids_generated': 0,
            'total_analysis_time': 0.0,
            'average_beats_per_track': 0.0
        }
    
    def generate_beat_grid(self, filepath: str, audio_result: Optional['TrackMetadata'] = None, options: Optional[Dict[str, Any]] = None) -> BeatGrid:
        """
        Generate a comprehensive beat grid for an audio file
        
        Args:
            filepath: Path to the audio file
            audio_result: Pre-computed audio analysis (optional)
            options: Generation options
            
        Returns:
            BeatGrid with complete beat and structure information
        """
        start_time = time.time()
        options = options or {}
        
        beat_grid = BeatGrid(
            filepath=filepath,
            beats=[],
            average_bpm=0.0,
            tempo_stability=0.0,
            downbeats=[],
            bars=[],
            phrases=[],
            time_signature=(4, 4),  # Default
            analysis_time=0.0,
            success=False
        )
        
        try:
            # Check for numpy availability
            if not NUMPY_AVAILABLE:
                raise AudioAnalysisError("Advanced beatgrid generation requires numpy (pip install numpy)")
            
            # Use provided analysis or fallback to internal analysis
            if audio_result is None or not hasattr(audio_result, 'audio_features'):
                # Fallback: get comprehensive audio analysis
                audio_result = self.audio_service.analyze_track(filepath, {
                    'bpm': True,
                    'beat_tracking': True,
                    'onset_detection': True,
                    'tempo_tracking': True,
                    'spectral': True
                })
                
                if not audio_result or not hasattr(audio_result, 'audio_features'):
                    raise AudioAnalysisError("Failed to get audio analysis")
            
            audio_features = audio_result.audio_features
            beats = audio_features.get('beats', [])
            tempo_track = audio_features.get('tempo_track', [])
            onsets = audio_features.get('onsets', [])
            spectral_features = audio_features.get('spectral_features', {})
            
            if not beats:
                raise AudioAnalysisError("No beats detected")
            
            # Convert beats to numpy array for easier processing
            beats = np.array(beats)
            
            # Generate beat grid points
            beat_points = self._generate_beat_points(beats, tempo_track)
            
            # Detect downbeats
            if self.enable_downbeat_detection:
                downbeats = self._detect_downbeats(beats, spectral_features, audio_features)
                beat_grid.downbeats = downbeats
                
                # Update beat points with downbeat information
                self._mark_downbeats(beat_points, downbeats)
            
            # Generate bar structure
            bars = self._generate_bars(beat_points)
            beat_grid.bars = bars
            
            # Generate phrase structure
            if self.enable_phrase_analysis:
                phrases = self._detect_phrases(beat_points, spectral_features)
                beat_grid.phrases = phrases
            
            # Calculate tempo statistics
            tempos = [bp.tempo_bpm for bp in beat_points]
            beat_grid.average_bpm = np.mean(tempos)
            beat_grid.tempo_stability = self._calculate_tempo_stability(tempos)
            
            # Detect time signature
            beat_grid.time_signature = self._detect_time_signature(beat_points, downbeats)
            
            # Validate and correct beat grid
            if self.precision_mode:
                beat_points = self._validate_and_correct_beat_grid(beat_points)
            
            beat_grid.beats = beat_points
            beat_grid.success = True
            
            # Update statistics
            with self._lock:
                self.stats['tracks_analyzed'] += 1
                self.stats['beat_grids_generated'] += 1
                self.stats['average_beats_per_track'] = (
                    (self.stats['average_beats_per_track'] * (self.stats['tracks_analyzed'] - 1) + len(beat_points))
                    / self.stats['tracks_analyzed']
                )
            
            return beat_grid
            
        except Exception as e:
            beat_grid.errors.append(f"Beat grid generation failed: {str(e)}")
            return beat_grid
            
        finally:
            beat_grid.analysis_time = time.time() - start_time
            with self._lock:
                self.stats['total_analysis_time'] += beat_grid.analysis_time
    
    def _generate_beat_points(self, beats: np.ndarray, tempo_track: List[float]) -> List[BeatGridPoint]:
        """Generate detailed beat grid points from beat positions"""
        beat_points = []
        
        if len(beats) == 0:
            return beat_points
        
        try:
            # Calculate tempo between consecutive beats
            beat_intervals = np.diff(beats)
            beat_tempos = 60.0 / beat_intervals  # Convert interval to BPM
            
            # Smooth tempo if tempo tracking not available
            if not tempo_track:
                # Apply smoothing to reduce tempo fluctuations
                from scipy import ndimage
                beat_tempos = ndimage.gaussian_filter1d(beat_tempos, sigma=self.tempo_smoothing)
            
            # Generate beat points
            for i, beat_time in enumerate(beats):
                # Determine tempo for this beat
                if tempo_track and i < len(tempo_track):
                    tempo = tempo_track[i]
                elif i < len(beat_tempos):
                    tempo = beat_tempos[i]
                else:
                    tempo = beat_tempos[-1] if len(beat_tempos) > 0 else 120.0
                
                # Calculate confidence based on tempo stability
                if i > 0 and i < len(beat_tempos):
                    tempo_variance = abs(tempo - np.mean(beat_tempos[max(0, i-2):i+3]))
                    confidence = max(0.1, 1.0 - (tempo_variance / 20.0))  # Normalize variance
                else:
                    confidence = 0.8  # Default confidence for edge cases
                
                # Determine beat number (1-4) - will be refined with downbeat detection
                beat_number = (i % 4) + 1
                bar_number = i // 4 + 1
                
                beat_point = BeatGridPoint(
                    position_seconds=beat_time,
                    position_samples=int(beat_time * self.sample_rate),
                    beat_number=beat_number,
                    bar_number=bar_number,
                    tempo_bpm=tempo,
                    confidence=confidence
                )
                
                beat_points.append(beat_point)
                
        except Exception as e:
            # Return what we have so far
            pass
        
        return beat_points
    
    def _detect_downbeats(self, beats: np.ndarray, spectral_features: Dict[str, Any], 
                         audio_features: Dict[str, Any]) -> List[float]:
        """Detect downbeat positions using multiple methods"""
        downbeats = []
        
        try:
            # Method 1: Use spectral features (bass emphasis on downbeats)
            if 'spectral_centroid' in spectral_features and 'spectral_rolloff' in spectral_features:
                centroid = np.array(spectral_features['spectral_centroid'])
                rolloff = np.array(spectral_features['spectral_rolloff'])
                
                # Calculate bass emphasis (lower centroid and rolloff values)
                bass_emphasis = 1.0 / (centroid + rolloff + 1e-6)  # Add small value to avoid division by zero
                
                # Find beats with high bass emphasis
                downbeat_candidates = []
                for i, beat_time in enumerate(beats[::4]):  # Check every 4th beat (potential downbeats)
                    # Find corresponding spectral frame
                    frame_idx = int(beat_time / len(audio_features.get('audio_data', [1])) * len(bass_emphasis))
                    frame_idx = min(frame_idx, len(bass_emphasis) - 1)
                    
                    if frame_idx >= 0:
                        bass_score = bass_emphasis[frame_idx]
                        downbeat_candidates.append((beat_time, bass_score))
                
                # Sort by bass score and take top candidates
                downbeat_candidates.sort(key=lambda x: x[1], reverse=True)
                downbeats = [candidate[0] for candidate in downbeat_candidates]
            
            # Method 2: Regular pattern assumption (every 4th beat)
            if len(downbeats) < 3:  # Fallback if spectral method didn't work well
                # Assume first beat is a downbeat and every 4th beat after
                downbeats = [beats[i] for i in range(0, len(beats), 4)]
            
            # Method 3: Refine using onset strength
            if 'onsets' in audio_features:
                onsets = np.array(audio_features['onsets'])
                refined_downbeats = []
                
                for downbeat in downbeats:
                    # Find closest strong onset
                    onset_distances = np.abs(onsets - downbeat)
                    closest_onset_idx = np.argmin(onset_distances)
                    
                    if onset_distances[closest_onset_idx] < 0.1:  # Within 100ms
                        refined_downbeats.append(onsets[closest_onset_idx])
                    else:
                        refined_downbeats.append(downbeat)
                
                downbeats = refined_downbeats
            
        except Exception as e:
            # Fallback: assume every 4th beat is a downbeat
            downbeats = [beats[i] for i in range(0, len(beats), 4)]
        
        return sorted(downbeats)
    
    def _mark_downbeats(self, beat_points: List[BeatGridPoint], downbeats: List[float]):
        """Mark downbeats in the beat grid and update beat/bar numbering"""
        if not downbeats:
            return
        
        try:
            downbeats = np.array(downbeats)
            
            # Reset beat and bar numbering based on downbeats
            current_bar = 1
            
            for i, beat_point in enumerate(beat_points):
                # Check if this beat is close to a downbeat
                distances = np.abs(downbeats - beat_point.position_seconds)
                min_distance = np.min(distances)
                
                if min_distance < 0.2:  # Within 200ms
                    beat_point.is_downbeat = True
                    beat_point.beat_number = 1
                    current_bar += 1
                    beat_point.bar_number = current_bar
                else:
                    # Calculate beat number based on distance from last downbeat
                    last_downbeat_idx = np.where(downbeats < beat_point.position_seconds)[0]
                    if len(last_downbeat_idx) > 0:
                        last_downbeat = downbeats[last_downbeat_idx[-1]]
                        
                        # Count beats since last downbeat
                        beats_since_downbeat = 0
                        for j in range(i):
                            if beat_points[j].position_seconds > last_downbeat:
                                beats_since_downbeat += 1
                        
                        beat_point.beat_number = (beats_since_downbeat % 4) + 1
                        beat_point.bar_number = current_bar
                        
        except Exception as e:
            # If marking fails, keep original numbering
            pass
    
    def _generate_bars(self, beat_points: List[BeatGridPoint]) -> List[Tuple[float, float]]:
        """Generate bar boundaries from beat grid"""
        bars = []
        
        if not beat_points:
            return bars
        
        try:
            current_bar_start = None
            current_bar_number = None
            
            for beat_point in beat_points:
                if current_bar_number != beat_point.bar_number:
                    # End previous bar
                    if current_bar_start is not None:
                        bars.append((current_bar_start, beat_point.position_seconds))
                    
                    # Start new bar
                    current_bar_start = beat_point.position_seconds
                    current_bar_number = beat_point.bar_number
            
            # Close the last bar
            if current_bar_start is not None and beat_points:
                last_beat_time = beat_points[-1].position_seconds
                # Estimate bar end based on average beat interval
                if len(beat_points) > 1:
                    avg_beat_interval = (beat_points[-1].position_seconds - beat_points[0].position_seconds) / (len(beat_points) - 1)
                    bar_end = last_beat_time + avg_beat_interval
                else:
                    bar_end = last_beat_time + 0.5  # Fallback
                
                bars.append((current_bar_start, bar_end))
                
        except Exception as e:
            # Return empty list if generation fails
            pass
        
        return bars
    
    def _detect_phrases(self, beat_points: List[BeatGridPoint], spectral_features: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Detect musical phrases (typically 4, 8, or 16 bars)"""
        phrases = []
        
        if len(beat_points) < 32:  # Need at least 8 bars for phrase detection
            return phrases
        
        try:
            # Group beats by bars
            bars = {}
            for beat in beat_points:
                bar_num = beat.bar_number
                if bar_num not in bars:
                    bars[bar_num] = []
                bars[bar_num].append(beat)
            
            bar_numbers = sorted(bars.keys())
            
            # Look for phrase boundaries using spectral changes
            if 'spectral_centroid' in spectral_features:
                centroid = np.array(spectral_features['spectral_centroid'])
                
                # Calculate spectral change between bars
                bar_centroids = []
                for bar_num in bar_numbers:
                    if bars[bar_num]:
                        # Get average position of beats in this bar
                        avg_position = np.mean([b.position_seconds for b in bars[bar_num]])
                        # Convert to spectral frame index
                        frame_idx = int(avg_position / beat_points[-1].position_seconds * len(centroid))
                        frame_idx = min(frame_idx, len(centroid) - 1)
                        bar_centroids.append(centroid[frame_idx])
                    else:
                        bar_centroids.append(0.0)
                
                # Find significant changes in spectral centroid
                bar_changes = np.diff(np.array(bar_centroids))
                change_threshold = np.std(bar_changes) * 1.5
                
                phrase_boundaries = [0]  # Start with first bar
                for i, change in enumerate(bar_changes):
                    if abs(change) > change_threshold:
                        phrase_boundaries.append(i + 1)  # +1 because diff reduces length by 1
                
                phrase_boundaries.append(len(bar_numbers) - 1)  # End with last bar
                
                # Convert bar boundaries to time boundaries
                for i in range(len(phrase_boundaries) - 1):
                    start_bar = bar_numbers[phrase_boundaries[i]]
                    end_bar = bar_numbers[phrase_boundaries[i + 1]]
                    
                    if start_bar in bars and end_bar in bars and bars[start_bar] and bars[end_bar]:
                        phrase_start = bars[start_bar][0].position_seconds
                        phrase_end = bars[end_bar][-1].position_seconds
                        phrases.append((phrase_start, phrase_end))
            
            # Fallback: create regular 8-bar phrases
            if not phrases:
                for i in range(0, len(bar_numbers), 8):
                    end_idx = min(i + 8, len(bar_numbers))
                    start_bar = bar_numbers[i]
                    end_bar = bar_numbers[end_idx - 1]
                    
                    if start_bar in bars and end_bar in bars and bars[start_bar] and bars[end_bar]:
                        phrase_start = bars[start_bar][0].position_seconds
                        phrase_end = bars[end_bar][-1].position_seconds
                        phrases.append((phrase_start, phrase_end))
                        
        except Exception as e:
            # Return empty list if detection fails
            pass
        
        return phrases
    
    def _calculate_tempo_stability(self, tempos: List[float]) -> float:
        """Calculate tempo stability score (0-1)"""
        if len(tempos) < 2:
            return 1.0
        
        try:
            tempo_variance = np.var(tempos)
            mean_tempo = np.mean(tempos)
            
            # Normalize variance by mean tempo
            normalized_variance = tempo_variance / (mean_tempo ** 2) if mean_tempo > 0 else 1.0
            
            # Convert to stability score (lower variance = higher stability)
            stability = max(0.0, min(1.0, 1.0 - (normalized_variance * 10)))
            
            return stability
            
        except Exception:
            return 0.5  # Default moderate stability
    
    def _detect_time_signature(self, beat_points: List[BeatGridPoint], downbeats: List[float]) -> Tuple[int, int]:
        """Detect time signature from beat patterns"""
        if not downbeats or len(downbeats) < 2:
            return (4, 4)  # Default
        
        try:
            # Calculate average number of beats between downbeats
            downbeats = np.array(downbeats)
            downbeat_intervals = np.diff(downbeats)
            
            # Count beats in each interval
            beats_per_measure = []
            for i in range(len(downbeat_intervals)):
                start_time = downbeats[i]
                end_time = downbeats[i + 1]
                
                # Count beats in this interval
                beat_count = 0
                for beat in beat_points:
                    if start_time <= beat.position_seconds < end_time:
                        beat_count += 1
                
                if beat_count > 0:
                    beats_per_measure.append(beat_count)
            
            if beats_per_measure:
                # Most common number of beats per measure
                most_common_beats = max(set(beats_per_measure), key=beats_per_measure.count)
                
                # Map to common time signatures
                if most_common_beats == 3:
                    return (3, 4)
                elif most_common_beats == 2:
                    return (2, 4)
                elif most_common_beats == 6:
                    return (6, 8)
                else:
                    return (4, 4)  # Default to 4/4
            
        except Exception:
            pass
        
        return (4, 4)  # Default
    
    def _validate_and_correct_beat_grid(self, beat_points: List[BeatGridPoint]) -> List[BeatGridPoint]:
        """Validate and correct beat grid for accuracy"""
        if not beat_points:
            return beat_points
        
        try:
            # Check for consistent timing
            positions = [bp.position_seconds for bp in beat_points]
            intervals = np.diff(positions)
            
            # Remove outliers (beats with very different intervals)
            median_interval = np.median(intervals)
            std_interval = np.std(intervals)
            
            corrected_points = [beat_points[0]]  # Keep first beat
            
            for i in range(1, len(beat_points)):
                interval = intervals[i - 1]
                
                # Check if interval is within reasonable bounds
                if abs(interval - median_interval) <= 2 * std_interval:
                    corrected_points.append(beat_points[i])
                else:
                    # Log warning but keep beat with adjusted confidence
                    beat_points[i].confidence *= 0.5
                    corrected_points.append(beat_points[i])
            
            return corrected_points
            
        except Exception:
            return beat_points
    
    def export_beat_grid(self, beat_grid: BeatGrid, format: str = 'json') -> str:
        """Export beat grid to various formats"""
        try:
            if format == 'json':
                return self._export_json(beat_grid)
            elif format == 'rekordbox':
                return self._export_rekordbox(beat_grid)
            elif format == 'serato':
                return self._export_serato(beat_grid)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            raise AudioAnalysisError(f"Beat grid export failed: {str(e)}")
    
    def _export_json(self, beat_grid: BeatGrid) -> str:
        """Export beat grid as JSON"""
        import json
        
        export_data = {
            'filepath': beat_grid.filepath,
            'average_bpm': beat_grid.average_bpm,
            'tempo_stability': beat_grid.tempo_stability,
            'time_signature': beat_grid.time_signature,
            'beats': [
                {
                    'position_seconds': bp.position_seconds,
                    'beat_number': bp.beat_number,
                    'bar_number': bp.bar_number,
                    'tempo_bpm': bp.tempo_bpm,
                    'confidence': bp.confidence,
                    'is_downbeat': bp.is_downbeat
                }
                for bp in beat_grid.beats
            ],
            'downbeats': beat_grid.downbeats,
            'bars': [{'start': bar[0], 'end': bar[1]} for bar in beat_grid.bars],
            'phrases': [{'start': phrase[0], 'end': phrase[1]} for phrase in beat_grid.phrases]
        }
        
        return json.dumps(export_data, indent=2)
    
    def _export_rekordbox(self, beat_grid: BeatGrid) -> str:
        """Export beat grid in Rekordbox format (simplified)"""
        # This would create Rekordbox-compatible beat grid data
        lines = []
        lines.append(f"# Rekordbox Beat Grid for {beat_grid.filepath}")
        lines.append(f"# Average BPM: {beat_grid.average_bpm:.2f}")
        lines.append(f"# Time Signature: {beat_grid.time_signature[0]}/{beat_grid.time_signature[1]}")
        
        for beat in beat_grid.beats:
            lines.append(f"{beat.position_seconds:.3f},{beat.tempo_bpm:.2f},{beat.beat_number},{int(beat.is_downbeat)}")
        
        return '\n'.join(lines)
    
    def _export_serato(self, beat_grid: BeatGrid) -> str:
        """Export beat grid in Serato format (simplified)"""
        # This would create Serato-compatible beat grid data
        lines = []
        lines.append(f"# Serato Beat Grid for {beat_grid.filepath}")
        
        for beat in beat_grid.beats:
            lines.append(f"{beat.position_samples},{beat.tempo_bpm:.1f}")
        
        return '\n'.join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        with self._lock:
            stats = self.stats.copy()
            if stats['total_analysis_time'] > 0:
                stats['average_analysis_time'] = stats['total_analysis_time'] / max(1, stats['tracks_analyzed'])
            else:
                stats['average_analysis_time'] = 0.0
            return stats


__all__ = ['BeatGridService', 'BeatGrid', 'BeatGridPoint']