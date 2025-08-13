"""
Service layer for Rekordbox operations.

This module provides high-level services for working with Rekordbox data,
including track management, file operations, and round-trip preservation
of DJ-specific metadata.
"""

import os
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set
from .models import RekordboxTrack
from .xml_parser import RekordboxXMLParser


class RekordboxService:
    """Service for Rekordbox operations"""
    
    def __init__(self):
        self.parser = RekordboxXMLParser()
        self.track_map: Dict[str, RekordboxTrack] = {}  # Hash -> Track
        self.processed_files: Dict[str, str] = {}  # Original path -> New path
        self.logger = logging.getLogger(__name__)
        
    def import_collection(self, xml_path: str, base_dir: Optional[str] = None) -> Dict[str, RekordboxTrack]:
        """Import Rekordbox collection and normalize paths"""
        self.parser.parse(xml_path)
        self.parser.normalize_file_paths(base_dir)
        
        # Build lookup map by file hash for matching processed files
        self._build_track_hash_map()
                
        return self.parser.tracks
    
    def _build_track_hash_map(self) -> None:
        """Build a hash-based lookup map for track matching"""
        for track_id, track in self.parser.tracks.items():
            if track.file_path and os.path.exists(track.file_path):
                try:
                    file_hash = self._calculate_file_hash(track.file_path)
                    self.track_map[file_hash] = track
                except (IOError, OSError) as e:
                    self.logger.warning(f"Could not hash file {track.file_path}: {str(e)}")
    
    def get_track_by_path(self, file_path: str) -> Optional[RekordboxTrack]:
        """Get Rekordbox track data by file path"""
        if not file_path:
            return None
            
        # Try direct path match first
        track = self.parser.get_track_by_path(file_path)
        if track:
            return track
                
        # Then try hash-based lookup
        if os.path.exists(file_path):
            try:
                file_hash = self._calculate_file_hash(file_path)
                return self.track_map.get(file_hash)
            except (IOError, OSError) as e:
                self.logger.warning(f"Could not hash file for lookup: {str(e)}")
            
        return None
    
    def register_processed_file(self, original_path: str, new_path: str) -> None:
        """Register a processed file to maintain mapping"""
        if not original_path or not new_path:
            return
            
        self.processed_files[original_path] = new_path
        
        # Find the track in our collection and update its location
        track = self.get_track_by_path(original_path)
        if track:
            self.parser.update_track_location(track.track_id, new_path)
            self.logger.info(f"Updated track location for {track.track_id} from {original_path} to {new_path}")
    
    def export_collection(self, output_path: str) -> str:
        """Export updated collection preserving all DJ data"""
        # Update any remaining track locations based on processed files
        self._update_track_locations()
                
        return self.parser.export_xml(output_path)
    
    def _update_track_locations(self) -> None:
        """Update all track locations based on processed files"""
        # This handles cases where files were processed outside the main loop
        for original_path, new_path in self.processed_files.items():
            track = self.get_track_by_path(original_path)
            if track and track.file_path != new_path:
                self.parser.update_track_location(track.track_id, new_path)
    
    @staticmethod
    def _calculate_file_hash(file_path: str, block_size: int = 8192) -> str:
        """Calculate hash from file for reliable matching"""
        # Use first 1MB for quick but reliable fingerprinting
        hasher = hashlib.md5()
        limit = 1024 * 1024  # 1MB
        read = 0
        
        with open(file_path, 'rb') as f:
            while read < limit:
                data = f.read(min(block_size, limit - read))
                if not data:
                    break
                hasher.update(data)
                read += len(data)
                
        return hasher.hexdigest()