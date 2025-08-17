"""File Validation Service for Quality Pre-filtering"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


class AudioFileValidator:
    """Fast audio file validation for quality pre-filtering"""
    
    SUPPORTED_EXTENSIONS = {'.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', '.wma', '.aiff'}
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_file_size_mb = self.config.get('min_file_size_mb', 1.0)
        self.min_bitrate_kbps = self.config.get('min_bitrate_kbps', 320)  # Default 320kbps
        self.check_bitrate = self.config.get('check_bitrate', True)
        self.validation_stats = {
            'total_checked': 0, 'passed': 0, 'failed_size': 0, 
            'failed_format': 0, 'failed_permissions': 0, 'failed_extension': 0,
            'failed_bitrate': 0
        }
    
    def validate_file(self, filepath: str) -> Tuple[bool, List[str]]:
        self.validation_stats['total_checked'] += 1
        errors = []
        
        if not os.path.exists(filepath):
            errors.append("File does not exist")
            return False, errors
        
        if not os.path.isfile(filepath):
            errors.append("Path is not a file")
            return False, errors
        
        # Extension check
        ext = Path(filepath).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            errors.append("Unsupported file extension")
            self.validation_stats['failed_extension'] += 1
        
        # Size check
        try:
            file_size = os.path.getsize(filepath)
            min_bytes = int(self.min_file_size_mb * 1024 * 1024)
            if file_size < min_bytes:
                errors.append(f"File too small ({file_size:,} bytes)")
                self.validation_stats['failed_size'] += 1
        except OSError:
            errors.append("Cannot access file")
            self.validation_stats['failed_permissions'] += 1
        
        # Permission check
        if not os.access(filepath, os.R_OK):
            errors.append("File not readable")
            self.validation_stats['failed_permissions'] += 1
        
        # Bitrate quality check (320kbps+)
        if self.check_bitrate and MUTAGEN_AVAILABLE:
            bitrate_errors = self._validate_bitrate(filepath)
            if bitrate_errors:
                errors.extend(bitrate_errors)
                self.validation_stats['failed_bitrate'] += 1
        
        if errors:
            return False, errors
        
        self.validation_stats['passed'] += 1
        return True, []
    
    def _validate_bitrate(self, filepath: str) -> List[str]:
        """Validate audio bitrate (320kbps+ for high quality)"""
        errors = []
        
        try:
            audio_file = MutagenFile(filepath)
            if audio_file is None:
                errors.append("Cannot read audio metadata")
                return errors
            
            # Get bitrate information
            bitrate = getattr(audio_file.info, 'bitrate', None)
            
            if bitrate is None:
                errors.append("Cannot determine bitrate")
                return errors
            
            # Convert to kbps and check against minimum
            bitrate_kbps = bitrate // 1000
            
            if bitrate_kbps < self.min_bitrate_kbps:
                errors.append(f"Low quality ({bitrate_kbps}kbps, minimum {self.min_bitrate_kbps}kbps)")
            
        except Exception as e:
            errors.append(f"Bitrate validation error: {str(e)}")
        
        return errors
    
    def get_valid_files(self, filepaths: List[str], verbose: bool = False) -> List[str]:
        valid_files = []
        for filepath in filepaths:
            is_valid, errors = self.validate_file(filepath)
            if is_valid:
                valid_files.append(filepath)
            elif verbose:
                print(f"   ⚠️ Skipping {os.path.basename(filepath)}: {'; '.join(errors)}")
        return valid_files
    
    def get_validation_stats(self) -> Dict[str, int]:
        return self.validation_stats.copy()


def create_quality_filter(min_size_mb: float = 1.0, min_bitrate_kbps: int = 320, check_bitrate: bool = True) -> AudioFileValidator:
    """Create a quality-focused file validator with 320kbps+ filtering by default"""
    return AudioFileValidator({
        'min_file_size_mb': min_size_mb,
        'min_bitrate_kbps': min_bitrate_kbps,
        'check_bitrate': check_bitrate
    })
