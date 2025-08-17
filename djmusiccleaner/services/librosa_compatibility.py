"""
Librosa Compatibility Layer

This module provides a drop-in replacement for librosa functions using aubio
and other stable libraries. It maintains the exact same API as librosa to
ensure no breaking changes during migration.

Usage:
    # Replace this:
    import librosa
    
    # With this:
    from djmusiccleaner.services.librosa_compatibility import librosa_compat as librosa

The module will automatically fall back to real librosa if aubio is not available
or if a specific function fails, ensuring backward compatibility.
"""

import warnings
import numpy as np
from typing import Tuple, Optional, Union, List, Any

# Try to import our aubio-based library
try:
    from . import aubio_analysis_library as aubio_lib
    AUBIO_LIB_AVAILABLE = True
except ImportError:
    AUBIO_LIB_AVAILABLE = False
    aubio_lib = None

# Try to import actual librosa as fallback
try:
    import librosa as real_librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    real_librosa = None

# Keep track of fallback usage for monitoring
fallback_usage = {
    'functions_called': {},
    'fallback_calls': 0,
    'aubio_calls': 0,
    'total_calls': 0
}


class LibrosaCompatibilityError(Exception):
    """Raised when both aubio and librosa are unavailable"""
    pass


def _track_call(func_name: str, used_fallback: bool = False):
    """Track function usage for monitoring"""
    fallback_usage['total_calls'] += 1
    fallback_usage['functions_called'][func_name] = fallback_usage['functions_called'].get(func_name, 0) + 1
    
    if used_fallback:
        fallback_usage['fallback_calls'] += 1
    else:
        fallback_usage['aubio_calls'] += 1


def get_usage_stats() -> dict:
    """Get usage statistics for monitoring migration progress"""
    stats = fallback_usage.copy()
    if stats['total_calls'] > 0:
        stats['aubio_success_rate'] = stats['aubio_calls'] / stats['total_calls']
        stats['fallback_rate'] = stats['fallback_calls'] / stats['total_calls']
    else:
        stats['aubio_success_rate'] = 0.0
        stats['fallback_rate'] = 0.0
    return stats


def _with_fallback(aubio_func_name: str, librosa_func, *args, **kwargs):
    """
    Execute aubio function with librosa fallback
    
    Args:
        aubio_func_name: Name of function in aubio_lib
        librosa_func: Corresponding librosa function
        *args, **kwargs: Function arguments
        
    Returns:
        Function result
    """
    func_name = librosa_func.__name__ if hasattr(librosa_func, '__name__') else str(librosa_func)
    
    # Try aubio implementation first
    if AUBIO_LIB_AVAILABLE:
        try:
            aubio_func = getattr(aubio_lib, aubio_func_name)
            result = aubio_func(*args, **kwargs)
            _track_call(func_name, used_fallback=False)
            return result
        except Exception as e:
            warnings.warn(f"Aubio {aubio_func_name} failed: {e}. Falling back to librosa.")
    
    # Fallback to librosa
    if LIBROSA_AVAILABLE:
        try:
            result = librosa_func(*args, **kwargs)
            _track_call(func_name, used_fallback=True)
            return result
        except Exception as e:
            _track_call(func_name, used_fallback=True)
            raise LibrosaCompatibilityError(f"Both aubio and librosa failed for {func_name}: {e}")
    
    # Neither available
    _track_call(func_name, used_fallback=True)
    raise LibrosaCompatibilityError(f"Neither aubio nor librosa available for {func_name}")


# Create a mock librosa module structure
class MockLibrosaModule:
    """Mock librosa module that routes to aubio-based implementations"""
    
    def load(self, *args, **kwargs):
        """Load audio file"""
        if AUBIO_LIB_AVAILABLE:
            return aubio_lib.load(*args, **kwargs)
        elif LIBROSA_AVAILABLE:
            return real_librosa.load(*args, **kwargs)
        else:
            raise LibrosaCompatibilityError("No audio loading library available")
    
    class beat:
        @staticmethod
        def beat_track(*args, **kwargs):
            """Beat tracking"""
            return _with_fallback('beat_track', real_librosa.beat.beat_track if LIBROSA_AVAILABLE else None, *args, **kwargs)
    
    class onset:
        @staticmethod
        def onset_detect(*args, **kwargs):
            """Onset detection"""
            return _with_fallback('onset_detect', real_librosa.onset.onset_detect if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod  
        def onset_strength(*args, **kwargs):
            """Onset strength"""
            return _with_fallback('onset_strength', real_librosa.onset.onset_strength if LIBROSA_AVAILABLE else None, *args, **kwargs)
    
    class feature:
        @staticmethod
        def spectral_centroid(*args, **kwargs):
            """Spectral centroid"""
            return _with_fallback('spectral_centroid', real_librosa.feature.spectral_centroid if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod
        def spectral_rolloff(*args, **kwargs):
            """Spectral rolloff"""
            return _with_fallback('spectral_rolloff', real_librosa.feature.spectral_rolloff if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod
        def spectral_bandwidth(*args, **kwargs):
            """Spectral bandwidth"""
            return _with_fallback('spectral_bandwidth', real_librosa.feature.spectral_bandwidth if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod
        def rms(*args, **kwargs):
            """RMS energy"""
            return _with_fallback('rms', real_librosa.feature.rms if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod
        def zero_crossing_rate(*args, **kwargs):
            """Zero crossing rate"""
            return _with_fallback('zero_crossing_rate', real_librosa.feature.zero_crossing_rate if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod
        def chroma_stft(*args, **kwargs):
            """Chroma STFT"""
            return _with_fallback('chroma_stft', real_librosa.feature.chroma_stft if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod
        def chroma_cqt(*args, **kwargs):
            """Chroma CQT - fallback to chroma_stft"""
            return _with_fallback('chroma_stft', real_librosa.feature.chroma_cqt if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod
        def mfcc(*args, **kwargs):
            """MFCC features"""
            return _with_fallback('mfcc', real_librosa.feature.mfcc if LIBROSA_AVAILABLE else None, *args, **kwargs)
    
    class util:
        @staticmethod
        def normalize(*args, **kwargs):
            """Normalize data"""
            return _with_fallback('normalize', real_librosa.util.normalize if LIBROSA_AVAILABLE else None, *args, **kwargs)
        
        @staticmethod
        def fix_length(data, length):
            """Fix array length - simple implementation"""
            if len(data) >= length:
                return data[:length]
            else:
                return np.pad(data, (0, length - len(data)), 'constant', constant_values=0)
        
        @staticmethod
        def resample(y, orig_sr, target_sr):
            """Audio resampling - use scipy or fallback"""
            try:
                from scipy import signal
                return signal.resample(y, int(len(y) * target_sr / orig_sr))
            except ImportError:
                if LIBROSA_AVAILABLE:
                    return real_librosa.util.resample(y, orig_sr, target_sr)
                else:
                    warnings.warn("No resampling library available. Returning original signal.")
                    return y
    
    class decompose:
        @staticmethod
        def hpss(*args, **kwargs):
            """Harmonic-percussive source separation"""
            return _with_fallback('hpss', real_librosa.decompose.hpss if LIBROSA_AVAILABLE else None, *args, **kwargs)
    
    class effects:
        @staticmethod
        def hpss(*args, **kwargs):
            """HPSS effect version"""
            if LIBROSA_AVAILABLE:
                return real_librosa.effects.hpss(*args, **kwargs)
            else:
                # Convert STFT result to time domain
                if AUBIO_LIB_AVAILABLE:
                    try:
                        # For effects.hpss, we need to work in time domain
                        y = args[0]
                        stft_result = aubio_lib.stft(y)
                        h_stft, p_stft = aubio_lib.hpss(stft_result)
                        
                        # Convert back to time domain (simplified)
                        h_y = np.real(np.fft.irfft(h_stft, axis=0)).flatten()[:len(y)]
                        p_y = np.real(np.fft.irfft(p_stft, axis=0)).flatten()[:len(y)]
                        
                        return h_y, p_y
                    except Exception as e:
                        warnings.warn(f"Aubio HPSS failed: {e}")
                        # Return original split in half
                        y = args[0]
                        return y * 0.5, y * 0.5
                else:
                    y = args[0]
                    return y * 0.5, y * 0.5
        
        @staticmethod
        def preemphasis(y, coef=0.95, return_zf=False):
            """Preemphasis filter - simple implementation"""
            if return_zf:
                if LIBROSA_AVAILABLE:
                    return real_librosa.effects.preemphasis(y, coef, return_zf)
                else:
                    filtered = np.append(y[0], y[1:] - coef * y[:-1])
                    return filtered, np.array([y[-1]])
            else:
                return np.append(y[0], y[1:] - coef * y[:-1])
    
    @staticmethod
    def stft(*args, **kwargs):
        """Short-time Fourier Transform"""
        return _with_fallback('stft', real_librosa.stft if LIBROSA_AVAILABLE else None, *args, **kwargs)
    
    @staticmethod
    def frames_to_time(*args, **kwargs):
        """Convert frames to time"""
        return _with_fallback('frames_to_time', real_librosa.frames_to_time if LIBROSA_AVAILABLE else None, *args, **kwargs)


# Create the compatibility module instance
librosa_compat = MockLibrosaModule()


# Additional utility functions for monitoring
def print_usage_report():
    """Print a report of aubio vs librosa usage"""
    stats = get_usage_stats()
    
    print("\n📊 Librosa Migration Usage Report")
    print("=" * 50)
    print(f"Total function calls: {stats['total_calls']}")
    print(f"Aubio success rate: {stats['aubio_success_rate']:.1%}")
    print(f"Librosa fallback rate: {stats['fallback_rate']:.1%}")
    
    if stats['functions_called']:
        print("\nMost used functions:")
        sorted_funcs = sorted(stats['functions_called'].items(), key=lambda x: x[1], reverse=True)
        for func_name, count in sorted_funcs[:10]:
            print(f"  {func_name}: {count} calls")
    
    if stats['aubio_success_rate'] >= 0.9:
        print("\n✅ Migration is successful! Ready to remove librosa dependency.")
    elif stats['aubio_success_rate'] >= 0.7:
        print("\n🟡 Migration is mostly successful. Some functions still use librosa fallback.")
    else:
        print("\n🔴 Migration needs work. Many functions are falling back to librosa.")


def test_compatibility():
    """Test basic compatibility functions"""
    print("🧪 Testing librosa compatibility layer...")
    
    # Test basic functions that don't require real audio
    try:
        # Test utility functions
        data = np.random.random(1000)
        normalized = librosa_compat.util.normalize(data)
        print("✅ util.normalize working")
        
        fixed_length = librosa_compat.util.fix_length(data, 1500)
        print("✅ util.fix_length working")
        
        # Test frames to time conversion
        frames = np.array([0, 10, 20, 30])
        times = librosa_compat.frames_to_time(frames)
        print("✅ frames_to_time working")
        
        print("✅ Basic compatibility tests passed!")
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        
    print_usage_report()


if __name__ == "__main__":
    test_compatibility()