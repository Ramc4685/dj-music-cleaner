"""
Audio Utilities

Provides common audio processing utilities used across the DJ Music Cleaner application.
Includes format detection, conversion helpers, sample rate utilities, and audio validation.
"""

import os
import struct
import wave
import contextlib
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import numpy as np

# Audio format constants
AUDIO_EXTENSIONS = {
    '.mp3': 'MP3',
    '.flac': 'FLAC', 
    '.wav': 'WAV',
    '.wave': 'WAV',
    '.m4a': 'M4A',
    '.mp4': 'MP4',
    '.aac': 'AAC',
    '.ogg': 'OGG',
    '.oga': 'OGG',
    '.aiff': 'AIFF',
    '.aif': 'AIFF',
    '.au': 'AU',
    '.wma': 'WMA',
    '.opus': 'OPUS'
}

LOSSLESS_FORMATS = {'.flac', '.wav', '.wave', '.aiff', '.aif', '.au'}
LOSSY_FORMATS = {'.mp3', '.m4a', '.mp4', '.aac', '.ogg', '.oga', '.wma', '.opus'}

# Common sample rates in Hz
COMMON_SAMPLE_RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000]

# Audio quality thresholds
QUALITY_THRESHOLDS = {
    'bitrate': {
        'poor': 128,      # kbps
        'acceptable': 192,
        'good': 256,
        'excellent': 320
    },
    'sample_rate': {
        'poor': 22050,    # Hz
        'acceptable': 44100,
        'good': 48000,
        'excellent': 96000
    },
    'dynamic_range': {
        'poor': 6,        # dB
        'acceptable': 12,
        'good': 20,
        'excellent': 25
    }
}


def is_audio_file(filepath: Union[str, Path]) -> bool:
    """
    Check if file is a supported audio format
    
    Args:
        filepath: Path to file to check
        
    Returns:
        True if file is a supported audio format
    """
    try:
        path = Path(filepath)
        return path.suffix.lower() in AUDIO_EXTENSIONS
    except Exception:
        return False


def get_audio_format(filepath: Union[str, Path]) -> Optional[str]:
    """
    Get audio format name from file extension
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Format name (e.g., 'MP3', 'FLAC') or None if not audio
    """
    try:
        path = Path(filepath)
        return AUDIO_EXTENSIONS.get(path.suffix.lower())
    except Exception:
        return None


def is_lossless_format(filepath: Union[str, Path]) -> bool:
    """
    Check if audio format is lossless
    
    Args:
        filepath: Path to audio file
        
    Returns:
        True if format is lossless
    """
    try:
        path = Path(filepath)
        return path.suffix.lower() in LOSSLESS_FORMATS
    except Exception:
        return False


def detect_audio_format_from_header(filepath: Union[str, Path]) -> Optional[str]:
    """
    Detect audio format from file header (magic bytes)
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Detected format or None if unknown
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
        
        if len(header) < 4:
            return None
        
        # Check magic bytes for various formats
        if header[:3] == b'ID3' or header[0:2] == b'\xff\xfb' or header[0:2] == b'\xff\xf3':
            return 'MP3'
        elif header[:4] == b'fLaC':
            return 'FLAC'
        elif header[:4] == b'RIFF' and header[8:12] == b'WAVE':
            return 'WAV'
        elif header[:4] == b'FORM' and header[8:12] == b'AIFF':
            return 'AIFF'
        elif header[:4] == b'OggS':
            return 'OGG'
        elif header[4:8] == b'ftyp':
            # Could be M4A, MP4, etc.
            return 'M4A'
        else:
            return None
            
    except Exception:
        return None


def validate_audio_file_header(filepath: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate audio file header for corruption
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        path = Path(filepath)
        
        # Check if file exists and has reasonable size
        if not path.exists():
            issues.append("File does not exist")
            return False, issues
        
        if path.stat().st_size < 100:
            issues.append("File too small to be valid audio")
            return False, issues
        
        # Detect format from header
        detected_format = detect_audio_format_from_header(filepath)
        extension_format = get_audio_format(filepath)
        
        if detected_format is None:
            issues.append("Unknown or corrupted audio format")
            return False, issues
        
        if extension_format and detected_format != extension_format:
            issues.append(f"File extension suggests {extension_format} but header indicates {detected_format}")
        
        # Format-specific validation
        if detected_format == 'WAV':
            wav_issues = _validate_wav_header(filepath)
            issues.extend(wav_issues)
        elif detected_format == 'MP3':
            mp3_issues = _validate_mp3_header(filepath)
            issues.extend(mp3_issues)
        elif detected_format == 'FLAC':
            flac_issues = _validate_flac_header(filepath)
            issues.extend(flac_issues)
        
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Validation error: {str(e)}")
        return False, issues


def _validate_wav_header(filepath: Union[str, Path]) -> List[str]:
    """Validate WAV file header"""
    issues = []
    
    try:
        with wave.open(str(filepath), 'rb') as wav_file:
            frames = wav_file.getnframes()
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            
            if frames <= 0:
                issues.append("WAV file contains no audio frames")
            if channels not in [1, 2]:
                issues.append(f"Unusual channel count: {channels}")
            if sample_rate not in COMMON_SAMPLE_RATES:
                issues.append(f"Unusual sample rate: {sample_rate} Hz")
            if sample_width not in [1, 2, 3, 4]:
                issues.append(f"Unusual sample width: {sample_width} bytes")
                
    except Exception as e:
        issues.append(f"WAV header validation failed: {str(e)}")
    
    return issues


def _validate_mp3_header(filepath: Union[str, Path]) -> List[str]:
    """Validate MP3 file header"""
    issues = []
    
    try:
        with open(filepath, 'rb') as f:
            # Read first few bytes to check for ID3 or sync frame
            header = f.read(10)
            
            if len(header) < 10:
                issues.append("MP3 file too small")
                return issues
            
            # Check for ID3 tag
            if header[:3] == b'ID3':
                # Skip ID3 tag
                version = header[3:5]
                if version not in [b'\x02\x00', b'\x03\x00', b'\x04\x00']:
                    issues.append(f"Unsupported ID3 version: {version}")
                
                # Get tag size
                size_bytes = header[6:10]
                size = (size_bytes[0] << 21) | (size_bytes[1] << 14) | (size_bytes[2] << 7) | size_bytes[3]
                
                # Seek to audio data
                f.seek(size + 10)
                audio_header = f.read(4)
            else:
                # No ID3 tag, header should be audio frame
                audio_header = header[:4]
            
            # Validate MP3 frame sync
            if len(audio_header) >= 2:
                if not (audio_header[0] == 0xFF and (audio_header[1] & 0xE0) == 0xE0):
                    issues.append("MP3 frame sync not found")
                    
    except Exception as e:
        issues.append(f"MP3 header validation failed: {str(e)}")
    
    return issues


def _validate_flac_header(filepath: Union[str, Path]) -> List[str]:
    """Validate FLAC file header"""
    issues = []
    
    try:
        with open(filepath, 'rb') as f:
            # Check FLAC signature
            signature = f.read(4)
            if signature != b'fLaC':
                issues.append("Invalid FLAC signature")
                return issues
            
            # Read first metadata block header
            block_header = f.read(4)
            if len(block_header) < 4:
                issues.append("FLAC metadata block header too short")
                return issues
            
            # Check if it's a STREAMINFO block (required first block)
            is_last = bool(block_header[0] & 0x80)
            block_type = block_header[0] & 0x7F
            
            if block_type != 0:
                issues.append("FLAC file missing required STREAMINFO block")
                
    except Exception as e:
        issues.append(f"FLAC header validation failed: {str(e)}")
    
    return issues


def get_audio_info_from_header(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract basic audio information from file header
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Dictionary with audio info (channels, sample_rate, duration, etc.)
    """
    info = {
        'format': None,
        'channels': None,
        'sample_rate': None,
        'duration': None,
        'bitrate': None,
        'bits_per_sample': None,
        'file_size': 0
    }
    
    try:
        path = Path(filepath)
        info['file_size'] = path.stat().st_size
        info['format'] = detect_audio_format_from_header(filepath)
        
        if info['format'] == 'WAV':
            wav_info = _get_wav_info(filepath)
            info.update(wav_info)
        elif info['format'] == 'FLAC':
            flac_info = _get_flac_info(filepath)
            info.update(flac_info)
        # MP3 and other formats would require more complex parsing
        
    except Exception:
        pass
    
    return info


def _get_wav_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Get WAV file information"""
    info = {}
    
    try:
        with wave.open(str(filepath), 'rb') as wav_file:
            info['channels'] = wav_file.getnchannels()
            info['sample_rate'] = wav_file.getframerate()
            info['bits_per_sample'] = wav_file.getsampwidth() * 8
            
            frames = wav_file.getnframes()
            if frames > 0 and info['sample_rate'] > 0:
                info['duration'] = frames / info['sample_rate']
                
                # Calculate bitrate
                info['bitrate'] = (info['sample_rate'] * info['channels'] * 
                                 info['bits_per_sample']) // 1000
                
    except Exception:
        pass
    
    return info


def _get_flac_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Get basic FLAC file information from STREAMINFO block"""
    info = {}
    
    try:
        with open(filepath, 'rb') as f:
            # Skip FLAC signature
            f.seek(4)
            
            # Read STREAMINFO block
            block_header = f.read(4)
            if len(block_header) < 4:
                return info
            
            block_type = block_header[0] & 0x7F
            if block_type == 0:  # STREAMINFO block
                streaminfo = f.read(34)  # STREAMINFO is always 34 bytes
                
                if len(streaminfo) >= 34:
                    # Parse STREAMINFO
                    min_block_size = struct.unpack('>H', streaminfo[0:2])[0]
                    max_block_size = struct.unpack('>H', streaminfo[2:4])[0]
                    
                    # Sample rate, channels, bits per sample (packed in 8 bytes)
                    packed = struct.unpack('>Q', streaminfo[10:18])[0]
                    
                    info['sample_rate'] = (packed >> 44) & 0xFFFFF
                    info['channels'] = ((packed >> 41) & 0x7) + 1
                    info['bits_per_sample'] = ((packed >> 36) & 0x1F) + 1
                    
                    # Total samples
                    total_samples = packed & 0xFFFFFFFFF
                    
                    if total_samples > 0 and info['sample_rate'] > 0:
                        info['duration'] = total_samples / info['sample_rate']
                        
                        # FLAC bitrate varies, but we can estimate
                        file_size = Path(filepath).stat().st_size
                        info['bitrate'] = int((file_size * 8) / (info['duration'] * 1000))
                        
    except Exception:
        pass
    
    return info


def calculate_audio_quality_score(audio_info: Dict[str, Any]) -> float:
    """
    Calculate audio quality score from audio information
    
    Args:
        audio_info: Dictionary with audio information
        
    Returns:
        Quality score from 0.0 to 10.0
    """
    score = 0.0
    
    try:
        # Bitrate score (0-4 points)
        bitrate = audio_info.get('bitrate', 0)
        if bitrate >= QUALITY_THRESHOLDS['bitrate']['excellent']:
            score += 4.0
        elif bitrate >= QUALITY_THRESHOLDS['bitrate']['good']:
            score += 3.0
        elif bitrate >= QUALITY_THRESHOLDS['bitrate']['acceptable']:
            score += 2.0
        elif bitrate >= QUALITY_THRESHOLDS['bitrate']['poor']:
            score += 1.0
        
        # Sample rate score (0-3 points)
        sample_rate = audio_info.get('sample_rate', 0)
        if sample_rate >= QUALITY_THRESHOLDS['sample_rate']['excellent']:
            score += 3.0
        elif sample_rate >= QUALITY_THRESHOLDS['sample_rate']['good']:
            score += 2.5
        elif sample_rate >= QUALITY_THRESHOLDS['sample_rate']['acceptable']:
            score += 2.0
        elif sample_rate >= QUALITY_THRESHOLDS['sample_rate']['poor']:
            score += 1.0
        
        # Format bonus (0-2 points)
        format_name = audio_info.get('format', '')
        if format_name in ['FLAC', 'WAV', 'AIFF']:
            score += 2.0  # Lossless format bonus
        elif format_name in ['MP3', 'M4A', 'OGG']:
            score += 1.0  # Common lossy format
        
        # Bit depth bonus (0-1 point)
        bits_per_sample = audio_info.get('bits_per_sample', 0)
        if bits_per_sample >= 24:
            score += 1.0
        elif bits_per_sample >= 16:
            score += 0.5
        
    except Exception:
        pass
    
    return min(10.0, score)


def normalize_sample_rate(sample_rate: int) -> int:
    """
    Normalize sample rate to nearest common value
    
    Args:
        sample_rate: Input sample rate in Hz
        
    Returns:
        Normalized sample rate
    """
    if sample_rate in COMMON_SAMPLE_RATES:
        return sample_rate
    
    # Find closest common sample rate
    differences = [abs(sample_rate - sr) for sr in COMMON_SAMPLE_RATES]
    min_diff_idx = differences.index(min(differences))
    
    return COMMON_SAMPLE_RATES[min_diff_idx]


def convert_time_to_samples(time_seconds: float, sample_rate: int) -> int:
    """
    Convert time in seconds to sample count
    
    Args:
        time_seconds: Time in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Number of samples
    """
    return int(time_seconds * sample_rate)


def convert_samples_to_time(samples: int, sample_rate: int) -> float:
    """
    Convert sample count to time in seconds
    
    Args:
        samples: Number of samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Time in seconds
    """
    if sample_rate <= 0:
        return 0.0
    return samples / sample_rate


def calculate_file_duration_estimate(filepath: Union[str, Path]) -> Optional[float]:
    """
    Estimate file duration without full decode (rough estimate)
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Estimated duration in seconds or None if can't estimate
    """
    try:
        path = Path(filepath)
        file_size = path.stat().st_size
        format_name = get_audio_format(filepath)
        
        # Rough estimates based on common bitrates
        if format_name == 'MP3':
            # Assume average 192 kbps
            estimated_duration = (file_size * 8) / (192 * 1000)
        elif format_name == 'FLAC':
            # Assume average 800 kbps (varies widely)
            estimated_duration = (file_size * 8) / (800 * 1000)
        elif format_name == 'WAV':
            # Calculate based on uncompressed PCM
            # Assume 44.1kHz, 16-bit, stereo = 1411.2 kbps
            estimated_duration = (file_size * 8) / (1411.2 * 1000)
        else:
            # Generic estimate
            estimated_duration = (file_size * 8) / (256 * 1000)  # Assume 256 kbps
        
        return estimated_duration
        
    except Exception:
        return None


def find_audio_files(directory: Union[str, Path], recursive: bool = True) -> List[str]:
    """
    Find all audio files in a directory
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    
    try:
        path = Path(directory)
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and is_audio_file(file_path):
                audio_files.append(str(file_path))
                
    except Exception:
        pass
    
    return sorted(audio_files)


@contextlib.contextmanager
def audio_file_context(filepath: Union[str, Path]):
    """
    Context manager for safely working with audio files
    
    Args:
        filepath: Path to audio file
        
    Yields:
        File path if valid, None if invalid
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            yield None
            return
        
        if not is_audio_file(path):
            yield None
            return
        
        yield str(path)
        
    except Exception:
        yield None


__all__ = [
    # Format detection
    'is_audio_file',
    'get_audio_format', 
    'is_lossless_format',
    'detect_audio_format_from_header',
    
    # Validation
    'validate_audio_file_header',
    'get_audio_info_from_header',
    
    # Quality assessment
    'calculate_audio_quality_score',
    
    # Utilities
    'normalize_sample_rate',
    'convert_time_to_samples',
    'convert_samples_to_time',
    'calculate_file_duration_estimate',
    'find_audio_files',
    'audio_file_context',
    
    # Constants
    'AUDIO_EXTENSIONS',
    'LOSSLESS_FORMATS', 
    'LOSSY_FORMATS',
    'COMMON_SAMPLE_RATES',
    'QUALITY_THRESHOLDS'
]