"""
Energy Calibration Service

Provides advanced energy analysis and calibration for DJ performance optimization.
Analyzes track energy profiles, calibrates levels across collections, and provides
energy-based mixing recommendations.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

try:
    import numpy as np
    from scipy import signal, stats, ndimage
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
    ArrayType = np.ndarray
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    ArrayType = Any

from ...core.models import TrackMetadata
from ...core.exceptions import AudioAnalysisError
from ..unified_audio_analysis import UnifiedAudioAnalysisService


@dataclass
class EnergyProfile:
    """Detailed energy profile for a track"""
    filepath: str
    overall_energy: float  # 0-100 scale
    energy_curve: List[float]  # Energy over time
    energy_peaks: List[Tuple[float, float]]  # (time, energy) peaks
    energy_valleys: List[Tuple[float, float]]  # (time, energy) valleys
    dynamic_range: float  # Difference between peaks and valleys
    rms_energy: float  # Root mean square energy
    peak_energy: float  # Maximum instantaneous energy
    energy_variance: float  # How much energy varies
    energy_distribution: Dict[str, float]  # Low, mid, high energy percentages
    calibration_factor: float  # Multiplier to normalize energy
    recommended_gain: float  # dB adjustment for optimal playback
    energy_rating: str  # "Low", "Medium", "High", "Very High"


@dataclass
class EnergyCalibration:
    """Collection-wide energy calibration settings"""
    collection_name: str
    track_count: int
    energy_statistics: Dict[str, float]
    calibration_curve: List[Tuple[float, float]]  # (input_energy, output_energy)
    reference_tracks: List[str]  # Tracks used as energy references
    created_at: float
    updated_at: float


class EnergyCalibrationService:
    """
    Advanced energy calibration service
    
    Features:
    - Detailed energy profile analysis
    - Collection-wide energy calibration
    - Dynamic range optimization
    - Energy-based track recommendations
    - Gain staging suggestions
    - Energy curve smoothing and enhancement
    - Cross-collection energy normalization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the energy calibration service"""
        self.config = config or {}
        
        # Configuration
        self.enable_dynamic_analysis = self.config.get('enable_dynamic_analysis', True)
        self.enable_frequency_weighting = self.config.get('enable_frequency_weighting', True)
        self.smoothing_window = self.config.get('smoothing_window', 1.0)  # seconds
        self.energy_scale = self.config.get('energy_scale', 100.0)  # 0-100 scale
        
        # Calibration settings
        self.target_energy = self.config.get('target_energy', 70.0)  # Target energy level
        self.dynamic_range_target = self.config.get('dynamic_range_target', 20.0)  # dB
        self.loudness_standard = self.config.get('loudness_standard', 'LUFS')  # LUFS or RMS
        
        # Analysis parameters
        self.sample_rate = self.config.get('sample_rate', 44100)
        self.frame_length = self.config.get('frame_length', 2048)
        self.hop_length = self.config.get('hop_length', 512)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Dependencies
        self.audio_service = UnifiedAudioAnalysisService(config)
        
        # Calibration storage
        self.calibrations: Dict[str, EnergyCalibration] = {}
        
        # Performance tracking
        self.stats = {
            'tracks_analyzed': 0,
            'calibrations_created': 0,
            'total_analysis_time': 0.0,
            'energy_adjustments_applied': 0
        }
    
    def analyze_track_energy(self, filepath: str, options: Optional[Dict[str, Any]] = None) -> EnergyProfile:
        """
        Analyze energy profile for a single track
        
        Args:
            filepath: Path to the audio file
            options: Analysis options
            
        Returns:
            EnergyProfile with comprehensive energy analysis
        """
        start_time = time.time()
        options = options or {}
        
        profile = EnergyProfile(
            filepath=filepath,
            overall_energy=0.0,
            energy_curve=[],
            energy_peaks=[],
            energy_valleys=[],
            dynamic_range=0.0,
            rms_energy=0.0,
            peak_energy=0.0,
            energy_variance=0.0,
            energy_distribution={},
            calibration_factor=1.0,
            recommended_gain=0.0,
            energy_rating="Medium"
        )
        
        
        try:
            # Check for scipy availability
            if not SCIPY_AVAILABLE:
                raise AudioAnalysisError("Energy calibration requires scipy (pip install scipy)")
                
            # Get audio analysis with energy focus
            audio_result = self.audio_service.analyze_track(filepath, {
                'energy': True,
                'spectral': True,
                'dynamics': True,
                'rms_analysis': True
            })
            
            if not audio_result or not hasattr(audio_result, 'audio_features'):
                raise AudioAnalysisError("Failed to get audio analysis")
            
            audio_features = audio_result.audio_features
            audio_data = audio_features.get('audio_data')
            energy_profile = audio_features.get('energy_profile', [])
            spectral_features = audio_features.get('spectral_features', {})
            
            if audio_data is None:
                raise AudioAnalysisError("Audio data not available")
            
            # Generate detailed energy curve
            energy_curve = self._generate_energy_curve(audio_data)
            profile.energy_curve = energy_curve.tolist() if isinstance(energy_curve, np.ndarray) else energy_curve
            
            # Calculate overall energy metrics
            profile.overall_energy = self._calculate_overall_energy(energy_curve)
            profile.rms_energy = self._calculate_rms_energy(audio_data)
            profile.peak_energy = self._calculate_peak_energy(audio_data)
            profile.dynamic_range = self._calculate_dynamic_range(energy_curve)
            profile.energy_variance = np.var(energy_curve) if len(energy_curve) > 0 else 0.0
            
            # Find energy peaks and valleys
            profile.energy_peaks = self._find_energy_peaks(energy_curve)
            profile.energy_valleys = self._find_energy_valleys(energy_curve)
            
            # Analyze frequency-based energy distribution
            if spectral_features:
                profile.energy_distribution = self._analyze_frequency_energy(spectral_features)
            
            # Calculate calibration recommendations
            profile.calibration_factor = self._calculate_calibration_factor(profile)
            profile.recommended_gain = self._calculate_recommended_gain(profile)
            profile.energy_rating = self._classify_energy_level(profile.overall_energy)
            
            # Update statistics
            with self._lock:
                self.stats['tracks_analyzed'] += 1
            
            return profile
            
        except Exception as e:
            raise AudioAnalysisError(f"Energy analysis failed: {str(e)}")
            
        finally:
            analysis_time = time.time() - start_time
            with self._lock:
                self.stats['total_analysis_time'] += analysis_time
    
    def _generate_energy_curve(self, audio_data: ArrayType) -> ArrayType:
        """Generate smooth energy curve over time"""
        try:
            # Ensure mono audio
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Calculate frame size based on smoothing window
            frame_size = int(self.smoothing_window * self.sample_rate)
            hop_size = frame_size // 4  # 75% overlap
            
            # Calculate RMS energy for each frame
            energy_frames = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                energy_frames.append(rms)
            
            energy_curve = np.array(energy_frames)
            
            # Apply frequency weighting if enabled
            if self.enable_frequency_weighting:
                energy_curve = self._apply_frequency_weighting(audio_data, energy_curve)
            
            # Smooth the energy curve
            energy_curve = self._smooth_energy_curve(energy_curve)
            
            # Normalize to 0-100 scale
            if np.max(energy_curve) > 0:
                energy_curve = (energy_curve / np.max(energy_curve)) * self.energy_scale
            
            return energy_curve
            
        except Exception as e:
            # Return flat energy curve as fallback
            return np.ones(100) * 50.0  # Medium energy
    
    def _apply_frequency_weighting(self, audio_data: ArrayType, energy_curve: ArrayType) -> ArrayType:
        """Apply frequency weighting to emphasize musically important frequencies"""
        try:
            # A-weighting filter (approximate)
            # This emphasizes mid frequencies where human hearing is most sensitive
            
            # Calculate STFT for frequency analysis
            f, t, Zxx = signal.stft(audio_data, fs=self.sample_rate, 
                                   nperseg=self.frame_length, 
                                   noverlap=self.frame_length - self.hop_length)
            
            # A-weighting curve approximation
            def a_weighting(freq):
                """Approximate A-weighting curve"""
                if freq <= 0:
                    return 0
                
                f2 = freq ** 2
                f4 = f2 ** 2
                
                # A-weighting formula
                numerator = 12194 ** 2 * f4
                denominator = (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2)
                
                if denominator > 0:
                    return numerator / denominator
                else:
                    return 0
            
            # Apply weighting to each frequency bin
            weighted_magnitude = np.abs(Zxx)
            for i, freq in enumerate(f):
                weight = a_weighting(freq)
                weighted_magnitude[i, :] *= weight
            
            # Calculate weighted energy
            weighted_energy = np.sum(weighted_magnitude ** 2, axis=0)
            
            # Resize to match energy_curve length
            if len(weighted_energy) != len(energy_curve):
                interp_func = interp1d(np.linspace(0, 1, len(weighted_energy)), 
                                     weighted_energy, 
                                     kind='linear', 
                                     fill_value='extrapolate')
                weighted_energy = interp_func(np.linspace(0, 1, len(energy_curve)))
            
            # Combine with original energy curve
            alpha = 0.7  # Weight factor for frequency weighting
            return alpha * weighted_energy + (1 - alpha) * energy_curve
            
        except Exception:
            # Return original curve if weighting fails
            return energy_curve
    
    def _smooth_energy_curve(self, energy_curve: ArrayType) -> ArrayType:
        """Apply smoothing to energy curve"""
        try:
            # Apply Gaussian smoothing
            sigma = len(energy_curve) * 0.01  # 1% of curve length
            smoothed = gaussian_filter1d(energy_curve, sigma=sigma)
            return smoothed
            
        except Exception:
            return energy_curve
    
    def _calculate_overall_energy(self, energy_curve: List[float]) -> float:
        """Calculate overall energy level"""
        if not energy_curve:
            return 0.0
        
        # Use 95th percentile to avoid being skewed by brief peaks
        return float(np.percentile(energy_curve, 95))
    
    def _calculate_rms_energy(self, audio_data: ArrayType) -> float:
        """Calculate RMS energy"""
        try:
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            return float(np.sqrt(np.mean(audio_data ** 2)))
            
        except Exception:
            return 0.0
    
    def _calculate_peak_energy(self, audio_data: ArrayType) -> float:
        """Calculate peak energy"""
        try:
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            return float(np.max(np.abs(audio_data)))
            
        except Exception:
            return 0.0
    
    def _calculate_dynamic_range(self, energy_curve: List[float]) -> float:
        """Calculate dynamic range in dB"""
        if not energy_curve:
            return 0.0
        
        try:
            energy_array = np.array(energy_curve)
            peak = np.max(energy_array)
            rms = np.sqrt(np.mean(energy_array ** 2))
            
            if rms > 0 and peak > 0:
                return float(20 * np.log10(peak / rms))
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _find_energy_peaks(self, energy_curve: List[float]) -> List[Tuple[float, float]]:
        """Find significant energy peaks"""
        if len(energy_curve) < 10:
            return []
        
        try:
            energy_array = np.array(energy_curve)
            
            # Find peaks with minimum prominence
            from scipy.signal import find_peaks
            prominence = np.std(energy_array) * 1.5
            peaks, properties = find_peaks(energy_array, prominence=prominence, distance=10)
            
            # Convert to time/energy pairs
            time_per_frame = len(energy_curve) / (len(energy_curve) - 1) if len(energy_curve) > 1 else 1.0
            
            peak_list = []
            for peak_idx in peaks:
                time_pos = peak_idx * time_per_frame
                energy_val = energy_array[peak_idx]
                peak_list.append((float(time_pos), float(energy_val)))
            
            return peak_list
            
        except Exception:
            return []
    
    def _find_energy_valleys(self, energy_curve: List[float]) -> List[Tuple[float, float]]:
        """Find significant energy valleys"""
        if len(energy_curve) < 10:
            return []
        
        try:
            energy_array = np.array(energy_curve)
            
            # Find valleys by inverting the signal and finding peaks
            inverted = -energy_array
            from scipy.signal import find_peaks
            prominence = np.std(energy_array) * 1.0
            valleys, properties = find_peaks(inverted, prominence=prominence, distance=10)
            
            # Convert to time/energy pairs
            time_per_frame = len(energy_curve) / (len(energy_curve) - 1) if len(energy_curve) > 1 else 1.0
            
            valley_list = []
            for valley_idx in valleys:
                time_pos = valley_idx * time_per_frame
                energy_val = energy_array[valley_idx]
                valley_list.append((float(time_pos), float(energy_val)))
            
            return valley_list
            
        except Exception:
            return []
    
    def _analyze_frequency_energy(self, spectral_features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze energy distribution across frequency bands"""
        distribution = {"low": 0.0, "mid": 0.0, "high": 0.0}
        
        try:
            if 'spectral_centroid' in spectral_features:
                centroid = np.array(spectral_features['spectral_centroid'])
                
                # Simple frequency band classification based on centroid
                mean_centroid = np.mean(centroid)
                
                if mean_centroid < 1000:  # Bass-heavy
                    distribution = {"low": 60.0, "mid": 30.0, "high": 10.0}
                elif mean_centroid < 3000:  # Mid-heavy
                    distribution = {"low": 30.0, "mid": 50.0, "high": 20.0}
                else:  # Treble-heavy
                    distribution = {"low": 20.0, "mid": 30.0, "high": 50.0}
            
            # More sophisticated analysis using spectral rolloff if available
            if 'spectral_rolloff' in spectral_features:
                rolloff = np.array(spectral_features['spectral_rolloff'])
                mean_rolloff = np.mean(rolloff)
                
                # Adjust distribution based on rolloff
                if mean_rolloff < 2000:
                    distribution["low"] += 10.0
                    distribution["high"] -= 5.0
                elif mean_rolloff > 8000:
                    distribution["high"] += 10.0
                    distribution["low"] -= 5.0
                
                # Normalize to 100%
                total = sum(distribution.values())
                if total > 0:
                    distribution = {k: (v / total) * 100 for k, v in distribution.items()}
            
        except Exception:
            # Default balanced distribution
            distribution = {"low": 33.3, "mid": 33.3, "high": 33.4}
        
        return distribution
    
    def _calculate_calibration_factor(self, profile: EnergyProfile) -> float:
        """Calculate calibration factor to normalize energy"""
        try:
            # Target is to normalize overall energy to target level
            if profile.overall_energy > 0:
                return self.target_energy / profile.overall_energy
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _calculate_recommended_gain(self, profile: EnergyProfile) -> float:
        """Calculate recommended gain adjustment in dB"""
        try:
            # Calculate gain to reach target energy level
            if profile.rms_energy > 0:
                target_rms = self.target_energy / 100.0  # Convert to 0-1 scale
                gain_linear = target_rms / profile.rms_energy
                gain_db = 20 * np.log10(gain_linear)
                
                # Limit gain adjustment to reasonable range
                return float(np.clip(gain_db, -20.0, 20.0))
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _classify_energy_level(self, overall_energy: float) -> str:
        """Classify energy level into categories"""
        if overall_energy < 25:
            return "Low"
        elif overall_energy < 50:
            return "Medium"
        elif overall_energy < 75:
            return "High"
        else:
            return "Very High"
    
    def create_collection_calibration(self, profiles: List[EnergyProfile], 
                                    collection_name: str = "default") -> EnergyCalibration:
        """Create calibration settings for a collection of tracks"""
        try:
            if not profiles:
                raise ValueError("No energy profiles provided")
            
            # Calculate collection statistics
            overall_energies = [p.overall_energy for p in profiles]
            dynamic_ranges = [p.dynamic_range for p in profiles]
            rms_energies = [p.rms_energy for p in profiles]
            
            energy_stats = {
                'mean_energy': float(np.mean(overall_energies)),
                'median_energy': float(np.median(overall_energies)),
                'std_energy': float(np.std(overall_energies)),
                'min_energy': float(np.min(overall_energies)),
                'max_energy': float(np.max(overall_energies)),
                'mean_dynamic_range': float(np.mean(dynamic_ranges)),
                'mean_rms': float(np.mean(rms_energies))
            }
            
            # Create calibration curve
            calibration_curve = self._create_calibration_curve(overall_energies)
            
            # Select reference tracks (median energy tracks)
            median_energy = energy_stats['median_energy']
            reference_tracks = []
            for profile in profiles:
                if abs(profile.overall_energy - median_energy) < energy_stats['std_energy'] * 0.5:
                    reference_tracks.append(profile.filepath)
                    if len(reference_tracks) >= 5:  # Limit to 5 reference tracks
                        break
            
            calibration = EnergyCalibration(
                collection_name=collection_name,
                track_count=len(profiles),
                energy_statistics=energy_stats,
                calibration_curve=calibration_curve,
                reference_tracks=reference_tracks,
                created_at=time.time(),
                updated_at=time.time()
            )
            
            # Store calibration
            with self._lock:
                self.calibrations[collection_name] = calibration
                self.stats['calibrations_created'] += 1
            
            return calibration
            
        except Exception as e:
            raise AudioAnalysisError(f"Collection calibration failed: {str(e)}")
    
    def _create_calibration_curve(self, energies: List[float]) -> List[Tuple[float, float]]:
        """Create calibration curve mapping input to output energy levels"""
        try:
            energies = np.array(energies)
            
            # Create percentile-based mapping
            percentiles = [10, 25, 50, 75, 90]
            input_levels = [np.percentile(energies, p) for p in percentiles]
            
            # Map to target distribution (more even spread)
            target_levels = [20, 35, 50, 65, 80]  # Target energy levels
            
            # Create calibration points
            curve = list(zip([float(x) for x in input_levels], [float(y) for y in target_levels]))
            
            return curve
            
        except Exception:
            # Default linear mapping
            return [(0.0, 0.0), (25.0, 25.0), (50.0, 50.0), (75.0, 75.0), (100.0, 100.0)]
    
    def apply_calibration(self, profile: EnergyProfile, collection_name: str = "default") -> EnergyProfile:
        """Apply collection calibration to an energy profile"""
        if collection_name not in self.calibrations:
            return profile
        
        try:
            calibration = self.calibrations[collection_name]
            
            # Apply calibration curve
            calibrated_energy = self._interpolate_calibration_curve(
                profile.overall_energy, 
                calibration.calibration_curve
            )
            
            # Create calibrated profile
            calibrated_profile = EnergyProfile(
                filepath=profile.filepath,
                overall_energy=calibrated_energy,
                energy_curve=profile.energy_curve,  # Keep original curve
                energy_peaks=profile.energy_peaks,
                energy_valleys=profile.energy_valleys,
                dynamic_range=profile.dynamic_range,
                rms_energy=profile.rms_energy,
                peak_energy=profile.peak_energy,
                energy_variance=profile.energy_variance,
                energy_distribution=profile.energy_distribution,
                calibration_factor=calibrated_energy / profile.overall_energy if profile.overall_energy > 0 else 1.0,
                recommended_gain=profile.recommended_gain,
                energy_rating=self._classify_energy_level(calibrated_energy)
            )
            
            with self._lock:
                self.stats['energy_adjustments_applied'] += 1
            
            return calibrated_profile
            
        except Exception as e:
            # Return original profile if calibration fails
            return profile
    
    def _interpolate_calibration_curve(self, input_energy: float, curve: List[Tuple[float, float]]) -> float:
        """Interpolate calibration curve to get output energy"""
        try:
            if not curve:
                return input_energy
            
            # Sort curve by input values
            curve = sorted(curve, key=lambda x: x[0])
            
            # Find surrounding points
            for i in range(len(curve) - 1):
                x1, y1 = curve[i]
                x2, y2 = curve[i + 1]
                
                if x1 <= input_energy <= x2:
                    # Linear interpolation
                    if x2 - x1 != 0:
                        ratio = (input_energy - x1) / (x2 - x1)
                        return y1 + ratio * (y2 - y1)
                    else:
                        return y1
            
            # If outside curve range, use nearest endpoint
            if input_energy < curve[0][0]:
                return curve[0][1]
            else:
                return curve[-1][1]
                
        except Exception:
            return input_energy
    
    def get_mixing_recommendations(self, track1_profile: EnergyProfile, 
                                 track2_profile: EnergyProfile) -> Dict[str, Any]:
        """Get recommendations for mixing two tracks based on energy"""
        try:
            energy_diff = abs(track1_profile.overall_energy - track2_profile.overall_energy)
            
            # Determine mixing compatibility
            if energy_diff < 10:
                compatibility = "Excellent"
            elif energy_diff < 20:
                compatibility = "Good"
            elif energy_diff < 30:
                compatibility = "Fair"
            else:
                compatibility = "Difficult"
            
            # Recommend gain adjustments
            gain_adjustment = 0.0
            if track1_profile.overall_energy > track2_profile.overall_energy:
                gain_adjustment = -(energy_diff / 10.0)  # Reduce louder track
            else:
                gain_adjustment = energy_diff / 10.0  # Boost quieter track
            
            recommendations = {
                'compatibility': compatibility,
                'energy_difference': energy_diff,
                'recommended_gain_adjustment': gain_adjustment,
                'mixing_notes': []
            }
            
            # Add specific mixing notes
            if compatibility == "Difficult":
                recommendations['mixing_notes'].append(
                    f"Large energy difference ({energy_diff:.1f}). Consider using EQ or filters for smoother transition."
                )
            
            if track1_profile.dynamic_range > 25 and track2_profile.dynamic_range < 10:
                recommendations['mixing_notes'].append(
                    "Mixing high dynamic range track with compressed track. Use gradual transition."
                )
            
            return recommendations
            
        except Exception as e:
            return {
                'compatibility': 'Unknown',
                'error': f"Analysis failed: {str(e)}"
            }
    
    def export_calibration(self, collection_name: str, filepath: str):
        """Export calibration settings to file"""
        if collection_name not in self.calibrations:
            raise ValueError(f"Calibration '{collection_name}' not found")
        
        try:
            calibration = self.calibrations[collection_name]
            
            export_data = {
                'collection_name': calibration.collection_name,
                'track_count': calibration.track_count,
                'energy_statistics': calibration.energy_statistics,
                'calibration_curve': calibration.calibration_curve,
                'reference_tracks': calibration.reference_tracks,
                'created_at': calibration.created_at,
                'updated_at': calibration.updated_at
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        except Exception as e:
            raise AudioAnalysisError(f"Calibration export failed: {str(e)}")
    
    def import_calibration(self, filepath: str) -> str:
        """Import calibration settings from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            calibration = EnergyCalibration(
                collection_name=data['collection_name'],
                track_count=data['track_count'],
                energy_statistics=data['energy_statistics'],
                calibration_curve=data['calibration_curve'],
                reference_tracks=data['reference_tracks'],
                created_at=data['created_at'],
                updated_at=time.time()  # Update timestamp
            )
            
            with self._lock:
                self.calibrations[calibration.collection_name] = calibration
            
            return calibration.collection_name
            
        except Exception as e:
            raise AudioAnalysisError(f"Calibration import failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats['calibrations_active'] = len(self.calibrations)
            if stats['total_analysis_time'] > 0:
                stats['average_analysis_time'] = stats['total_analysis_time'] / max(1, stats['tracks_analyzed'])
            else:
                stats['average_analysis_time'] = 0.0
            return stats


__all__ = ['EnergyCalibrationService', 'EnergyProfile', 'EnergyCalibration']