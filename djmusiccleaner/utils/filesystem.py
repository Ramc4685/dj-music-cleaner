"""
Filesystem utilities for DJ Music Cleaner

This module provides safe file operations, path handling, and file information utilities.
"""

import os
import shutil
import stat
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import time
import hashlib

from ..core.exceptions import FileOperationError


def ensure_directory(path: str, create: bool = True) -> bool:
    """
    Ensure a directory exists, optionally creating it
    
    Args:
        path: Directory path to check/create
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        True if directory exists or was created successfully
        
    Raises:
        FileOperationError: If directory creation fails
    """
    try:
        path_obj = Path(path)
        
        if path_obj.exists():
            if path_obj.is_dir():
                return True
            else:
                raise FileOperationError(
                    f"Path exists but is not a directory: {path}",
                    filepath=path
                )
        
        if create:
            path_obj.mkdir(parents=True, exist_ok=True)
            return True
        
        return False
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to ensure directory: {str(e)}",
            details=str(e),
            filepath=path
        )


def safe_move_file(src_path: str, dst_path: str, backup: bool = True, 
                   overwrite: bool = False) -> Tuple[bool, str]:
    """
    Safely move a file with backup and conflict resolution
    
    Args:
        src_path: Source file path
        dst_path: Destination file path
        backup: Create backup if destination exists
        overwrite: Allow overwriting existing files
        
    Returns:
        Tuple of (success, final_destination_path)
        
    Raises:
        FileOperationError: If move operation fails
    """
    try:
        src = Path(src_path)
        dst = Path(dst_path)
        
        # Validate source
        if not src.exists():
            raise FileOperationError(
                f"Source file does not exist: {src_path}",
                filepath=src_path
            )
        
        if not src.is_file():
            raise FileOperationError(
                f"Source is not a file: {src_path}",
                filepath=src_path
            )
        
        # Ensure destination directory exists
        ensure_directory(str(dst.parent))
        
        # Handle existing destination
        if dst.exists():
            if not overwrite:
                if backup:
                    # Create backup of existing file
                    backup_path = _create_backup_path(str(dst))
                    shutil.move(str(dst), backup_path)
                else:
                    # Find unique name
                    dst = _find_unique_path(dst)
            else:
                # Remove existing file
                dst.unlink()
        
        # Perform the move
        shutil.move(str(src), str(dst))
        
        return True, str(dst)
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to move file: {str(e)}",
            details=f"From: {src_path}, To: {dst_path}",
            filepath=src_path
        )


def safe_copy_file(src_path: str, dst_path: str, preserve_metadata: bool = True) -> bool:
    """
    Safely copy a file with metadata preservation
    
    Args:
        src_path: Source file path
        dst_path: Destination file path
        preserve_metadata: Whether to preserve file metadata
        
    Returns:
        True if copy was successful
        
    Raises:
        FileOperationError: If copy operation fails
    """
    try:
        src = Path(src_path)
        dst = Path(dst_path)
        
        if not src.exists() or not src.is_file():
            raise FileOperationError(
                f"Source file does not exist or is not a file: {src_path}",
                filepath=src_path
            )
        
        # Ensure destination directory exists
        ensure_directory(str(dst.parent))
        
        # Perform copy
        if preserve_metadata:
            shutil.copy2(str(src), str(dst))
        else:
            shutil.copy(str(src), str(dst))
        
        return True
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to copy file: {str(e)}",
            details=f"From: {src_path}, To: {dst_path}",
            filepath=src_path
        )


def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get comprehensive file information
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary with file information
        
    Raises:
        FileOperationError: If file access fails
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            raise FileOperationError(
                f"File does not exist: {filepath}",
                filepath=filepath
            )
        
        stat_info = path.stat()
        
        info = {
            'filepath': str(path.absolute()),
            'filename': path.name,
            'basename': path.stem,
            'extension': path.suffix.lstrip('.'),
            'parent_directory': str(path.parent),
            'size_bytes': stat_info.st_size,
            'size_mb': round(stat_info.st_size / (1024 * 1024), 2),
            'created_time': stat_info.st_ctime,
            'modified_time': stat_info.st_mtime,
            'accessed_time': stat_info.st_atime,
            'is_file': path.is_file(),
            'is_directory': path.is_dir(),
            'is_symlink': path.is_symlink(),
            'permissions': oct(stat_info.st_mode)[-3:],
            'readable': os.access(filepath, os.R_OK),
            'writable': os.access(filepath, os.W_OK),
            'executable': os.access(filepath, os.X_OK)
        }
        
        # Add file hash for uniqueness checking
        if path.is_file() and stat_info.st_size > 0:
            info['md5_hash'] = _calculate_file_hash(filepath, 'md5')
            info['sha256_hash'] = _calculate_file_hash(filepath, 'sha256')
        
        return info
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to get file info: {str(e)}",
            details=str(e),
            filepath=filepath
        )


def find_audio_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all audio files in a directory
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of audio file paths
        
    Raises:
        FileOperationError: If directory access fails
    """
    audio_extensions = {
        '.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', 
        '.wma', '.aiff', '.ape', '.mpc', '.opus'
    }
    
    try:
        path = Path(directory)
        
        if not path.exists() or not path.is_dir():
            raise FileOperationError(
                f"Directory does not exist or is not a directory: {directory}",
                filepath=directory
            )
        
        audio_files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path.absolute()))
        
        return sorted(audio_files)
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to find audio files: {str(e)}",
            details=str(e),
            filepath=directory
        )


def is_audio_file(filepath: str) -> bool:
    """
    Check if a file is an audio file based on extension
    
    Args:
        filepath: Path to check
        
    Returns:
        True if file appears to be an audio file
    """
    audio_extensions = {
        '.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg',
        '.wma', '.aiff', '.ape', '.mpc', '.opus'
    }
    
    try:
        path = Path(filepath)
        return path.suffix.lower() in audio_extensions
    except:
        return False


def get_available_space(path: str) -> int:
    """
    Get available disk space for a path in bytes
    
    Args:
        path: Path to check (file or directory)
        
    Returns:
        Available space in bytes
    """
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path),
                ctypes.pointer(free_bytes),
                None, None
            )
            return free_bytes.value
        else:  # Unix-like systems
            statvfs = os.statvfs(path)
            return statvfs.f_frsize * statvfs.f_bavail
    except:
        return 0


def create_backup_path(filepath: str, backup_dir: Optional[str] = None) -> str:
    """
    Create a backup file path
    
    Args:
        filepath: Original file path
        backup_dir: Optional backup directory (defaults to same directory)
        
    Returns:
        Backup file path
    """
    return _create_backup_path(filepath, backup_dir)


def cleanup_empty_directories(directory: str, recursive: bool = True) -> int:
    """
    Remove empty directories
    
    Args:
        directory: Starting directory
        recursive: Whether to check recursively
        
    Returns:
        Number of directories removed
    """
    try:
        path = Path(directory)
        removed_count = 0
        
        if recursive:
            # Walk from deepest to shallowest
            for current_path in sorted(path.rglob('*'), key=lambda x: len(x.parts), reverse=True):
                if current_path.is_dir():
                    try:
                        current_path.rmdir()  # Only removes if empty
                        removed_count += 1
                    except OSError:
                        pass  # Directory not empty or other error
        else:
            try:
                path.rmdir()
                removed_count = 1
            except OSError:
                pass
        
        return removed_count
        
    except Exception:
        return 0


# Private helper functions

def _create_backup_path(filepath: str, backup_dir: Optional[str] = None) -> str:
    """Create a backup file path with timestamp"""
    path = Path(filepath)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if backup_dir:
        backup_parent = Path(backup_dir)
        ensure_directory(str(backup_parent))
    else:
        backup_parent = path.parent
    
    backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
    return str(backup_parent / backup_name)


def _find_unique_path(path: Path) -> Path:
    """Find a unique file path by appending numbers"""
    if not path.exists():
        return path
    
    counter = 1
    while True:
        new_name = f"{path.stem}_{counter:03d}{path.suffix}"
        new_path = path.parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1
        
        # Safety limit
        if counter > 999:
            timestamp = int(time.time())
            new_name = f"{path.stem}_{timestamp}{path.suffix}"
            return path.parent / new_name


def _calculate_file_hash(filepath: str, algorithm: str = 'md5') -> str:
    """Calculate file hash"""
    try:
        if algorithm.lower() == 'md5':
            hasher = hashlib.md5()
        elif algorithm.lower() == 'sha256':
            hasher = hashlib.sha256()
        else:
            return ""
        
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()
        
    except Exception:
        return ""


# Export functions
__all__ = [
    'ensure_directory',
    'safe_move_file', 
    'safe_copy_file',
    'get_file_info',
    'find_audio_files',
    'is_audio_file',
    'get_available_space',
    'create_backup_path',
    'cleanup_empty_directories'
]