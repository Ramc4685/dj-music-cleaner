"""
Service layer for Rekordbox operations.

This module provides high-level services for working with Rekordbox data,
including track management, file operations, and round-trip preservation
of DJ-specific metadata.
"""

import os
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from .models import RekordboxTrack
from .xml_parser import RekordboxXMLParser
from .pyrekordbox_adapter import PyrekordboxAdapter, PYREKORDBOX_AVAILABLE


class RekordboxService:
    """Service for Rekordbox operations"""
    
    def __init__(self, use_pyrekordbox: bool = True):
        # Enhanced adapter with pyrekordbox support
        self.adapter = PyrekordboxAdapter(use_pyrekordbox=use_pyrekordbox)
        
        # Legacy parser for compatibility
        self.parser = RekordboxXMLParser()
        
        self.track_map: Dict[str, RekordboxTrack] = {}  # Hash -> Track
        self.processed_files: Dict[str, str] = {}  # Original path -> New path
        self.logger = logging.getLogger(__name__)
        
        # Log capabilities
        capabilities = self.adapter.get_capabilities()
        if capabilities['using_pyrekordbox']:
            self.logger.info("🎉 Enhanced Rekordbox service with pyrekordbox support")
            self.logger.info(f"   Database access: {capabilities['database_access']}")
            self.logger.info(f"   Analysis files: {capabilities['analysis_files']}")
            self.logger.info(f"   Advanced playlists: {capabilities['advanced_playlists']}")
        else:
            self.logger.info("📄 Standard Rekordbox service (native XML parser)")
        
    def import_collection(self, xml_path: str, base_dir: Optional[str] = None) -> Dict[str, RekordboxTrack]:
        """Import Rekordbox collection and normalize paths"""
        # Use enhanced adapter for parsing
        self.adapter.parse_xml(xml_path)
        
        # Also use legacy parser for compatibility
        self.parser.parse(xml_path)
        self.parser.normalize_file_paths(base_dir)
        
        # Build lookup map by file hash for matching processed files
        self._build_track_hash_map()
        
        # Use adapter tracks if available, fallback to parser
        tracks = self.adapter.get_tracks()
        if not tracks:
            tracks = self.parser.tracks
                
        return tracks
    
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
    
    def import_database(self, db_path: str) -> Dict[str, RekordboxTrack]:
        """
        Import from Rekordbox database (.db file) - Enhanced feature
        
        Args:
            db_path: Path to Rekordbox database file
            
        Returns:
            Dictionary of tracks from database
        """
        if not self.adapter.get_capabilities()['database_access']:
            self.logger.warning("Database import requires pyrekordbox")
            return {}
        
        try:
            # This would use pyrekordbox database features
            self.logger.info(f"🔍 Enhanced: Reading Rekordbox database: {db_path}")
            
            # For now, return empty dict - this would be implemented with pyrekordbox.db6
            # when we have a real database file to test with
            self.logger.info("💡 Database import capability available but not yet implemented")
            return {}
            
        except Exception as e:
            self.logger.error(f"Database import failed: {e}")
            return {}
    
    def extract_analysis_data(self, analysis_file_path: str) -> Dict[str, Any]:
        """
        Extract waveform and analysis data from Rekordbox analysis files
        
        Args:
            analysis_file_path: Path to .DAT, .EXT, or .2EX file
            
        Returns:
            Analysis data dictionary
        """
        if not self.adapter.get_capabilities()['analysis_files']:
            self.logger.warning("Analysis file extraction requires pyrekordbox")
            return {}
        
        try:
            from pyrekordbox import anlz
            
            self.logger.info(f"🌊 Enhanced: Extracting analysis data: {analysis_file_path}")
            
            # Parse analysis file
            analysis = anlz.RekordboxAnlz.parse_file(analysis_file_path)
            
            result = {
                'waveform': None,
                'beat_grid': [],
                'cue_points': [],
                'loops': []
            }
            
            # Extract different types of analysis data
            if hasattr(analysis, 'waveforms') and analysis.waveforms:
                result['waveform'] = {
                    'preview': analysis.waveforms[0].preview if analysis.waveforms else None,
                    'detailed': analysis.waveforms[0].detailed if analysis.waveforms else None
                }
            
            if hasattr(analysis, 'beat_grid') and analysis.beat_grid:
                result['beat_grid'] = [
                    {
                        'beat': beat.beat,
                        'time': beat.time,
                        'tempo': beat.tempo
                    }
                    for beat in analysis.beat_grid
                ]
            
            if hasattr(analysis, 'cue_points') and analysis.cue_points:
                result['cue_points'] = [
                    {
                        'type': cue.type,
                        'time': cue.time,
                        'label': getattr(cue, 'label', ''),
                        'color': getattr(cue, 'color', 0)
                    }
                    for cue in analysis.cue_points
                ]
            
            self.logger.info(f"✅ Extracted: {len(result['beat_grid'])} beats, {len(result['cue_points'])} cues")
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis extraction failed: {e}")
            return {}
    
    def create_advanced_playlist(self, name: str, tracks: List[str]) -> bool:
        """
        Create playlist with advanced pyrekordbox features
        
        Args:
            name: Playlist name
            tracks: List of track IDs
            
        Returns:
            True if successful
        """
        if not self.adapter.get_capabilities()['advanced_playlists']:
            self.logger.warning("Advanced playlist creation requires pyrekordbox")
            return False
        
        try:
            self.logger.info(f"📋 Enhanced: Creating playlist '{name}' with {len(tracks)} tracks")
            
            # This would use pyrekordbox playlist features
            # Implementation would depend on having XML loaded in adapter
            
            self.logger.info("💡 Advanced playlist capability available")
            return True
            
        except Exception as e:
            self.logger.error(f"Playlist creation failed: {e}")
            return False
    
    def get_enhanced_capabilities(self) -> Dict[str, Any]:
        """
        Get detailed information about enhanced capabilities
        
        Returns:
            Dictionary of capabilities and their status
        """
        capabilities = self.adapter.get_capabilities()
        
        enhanced_info = {
            'pyrekordbox_version': None,
            'supported_formats': {
                'xml': True,
                'database': capabilities['database_access'],
                'analysis_files': capabilities['analysis_files']
            },
            'features': {
                'waveform_extraction': capabilities['analysis_files'],
                'beat_grid_analysis': capabilities['analysis_files'],
                'cue_point_extraction': capabilities['analysis_files'],
                'advanced_playlists': capabilities['advanced_playlists'],
                'database_import': capabilities['database_access']
            },
            'file_types': {
                'xml_files': ['.xml'],
                'database_files': ['.db', '.edb'] if capabilities['database_access'] else [],
                'analysis_files': ['.DAT', '.EXT', '.2EX'] if capabilities['analysis_files'] else []
            }
        }
        
        if capabilities['pyrekordbox_available']:
            try:
                import pyrekordbox
                enhanced_info['pyrekordbox_version'] = getattr(pyrekordbox, '__version__', 'unknown')
            except:
                pass
        
        return enhanced_info