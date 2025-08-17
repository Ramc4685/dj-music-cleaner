"""
PyRekordbox Integration Adapter

Provides optional pyrekordbox integration while maintaining compatibility
with the existing native XML parser. Allows for gradual migration and
testing of enhanced capabilities.
"""

import logging
from typing import Dict, List, Optional, Any
from .xml_parser import RekordboxXMLParser
from .models import RekordboxTrack

# Optional pyrekordbox import
try:
    from pyrekordbox.rbxml import RekordboxXml
    PYREKORDBOX_AVAILABLE = True
except ImportError:
    PYREKORDBOX_AVAILABLE = False
    RekordboxXml = None


class PyrekordboxAdapter:
    """
    Adapter that can use either pyrekordbox or native implementation
    
    Benefits of pyrekordbox integration:
    - More robust XML parsing
    - Access to database files (.db)
    - Analysis file support (.DAT, .EXT, .2EX)
    - Advanced playlist management
    - Active maintenance and updates
    
    Fallback to native implementation ensures compatibility
    """
    
    def __init__(self, use_pyrekordbox: bool = True, fallback_to_native: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_pyrekordbox = use_pyrekordbox and PYREKORDBOX_AVAILABLE
        self.fallback_to_native = fallback_to_native
        
        # Always keep native parser available
        self.native_parser = RekordboxXMLParser()
        self.pyrekordbox_xml = None
        
        if self.use_pyrekordbox:
            self.logger.info("PyRekordbox adapter initialized with pyrekordbox support")
        else:
            reason = "not available" if not PYREKORDBOX_AVAILABLE else "disabled"
            self.logger.info(f"PyRekordbox adapter using native parser (pyrekordbox {reason})")
    
    def parse_xml(self, xml_path: str) -> 'PyrekordboxAdapter':
        """
        Parse XML using pyrekordbox or native implementation
        
        Args:
            xml_path: Path to Rekordbox XML file
            
        Returns:
            Self for method chaining
        """
        if self.use_pyrekordbox:
            try:
                self.logger.debug(f"Attempting to parse XML with pyrekordbox: {xml_path}")
                self.pyrekordbox_xml = RekordboxXml(xml_path)
                self.logger.info(f"Successfully parsed XML with pyrekordbox: {len(self.pyrekordbox_xml.get_tracks())} tracks")
                return self
            except Exception as e:
                self.logger.warning(f"pyrekordbox parsing failed: {e}")
                if not self.fallback_to_native:
                    raise
                self.logger.info("Falling back to native XML parser")
                self.use_pyrekordbox = False  # Disable for this session
        
        # Use native parser
        self.native_parser.parse(xml_path)
        self.logger.info(f"Parsed XML with native parser: {len(self.native_parser.tracks)} tracks")
        return self
    
    def get_tracks(self) -> Dict[str, RekordboxTrack]:
        """
        Get all tracks, converting from pyrekordbox format if needed
        
        Returns:
            Dictionary of track_id -> RekordboxTrack
        """
        if self.use_pyrekordbox and self.pyrekordbox_xml:
            return self._convert_pyrekordbox_tracks()
        return self.native_parser.tracks
    
    def get_track_by_path(self, file_path: str) -> Optional[RekordboxTrack]:
        """
        Get track by file path using appropriate parser
        
        Args:
            file_path: Path to audio file
            
        Returns:
            RekordboxTrack or None if not found
        """
        if self.use_pyrekordbox and self.pyrekordbox_xml:
            return self._find_track_by_path_pyrekordbox(file_path)
        return self.native_parser.get_track_by_path(file_path)
    
    def update_track_location(self, track_id: str, new_location: str) -> bool:
        """
        Update track location using appropriate parser
        
        Args:
            track_id: Track ID to update
            new_location: New file path
            
        Returns:
            True if successful
        """
        if self.use_pyrekordbox and self.pyrekordbox_xml:
            return self._update_location_pyrekordbox(track_id, new_location)
        return self.native_parser.update_track_location(track_id, new_location)
    
    def export_xml(self, output_path: str) -> str:
        """
        Export XML using appropriate method
        
        Args:
            output_path: Path for output XML file
            
        Returns:
            Path to exported file
        """
        if self.use_pyrekordbox and self.pyrekordbox_xml:
            self.pyrekordbox_xml.write_xml(output_path)
            self.logger.info(f"Exported XML with pyrekordbox to: {output_path}")
            return output_path
        return self.native_parser.export_xml(output_path)
    
    def _convert_pyrekordbox_tracks(self) -> Dict[str, RekordboxTrack]:
        """Convert pyrekordbox tracks to native RekordboxTrack format"""
        converted_tracks = {}
        
        try:
            pyrekordbox_tracks = self.pyrekordbox_xml.get_tracks()
            
            for track in pyrekordbox_tracks:
                # Create native track object
                native_track = RekordboxTrack()
                
                # Map basic fields
                native_track.track_id = str(track.get("TrackID", ""))
                native_track.title = track.get("Name", "")
                native_track.artist = track.get("Artist", "")
                native_track.album = track.get("Album", "")
                native_track.genre = track.get("Genre", "")
                native_track.comment = track.get("Comments", "")
                native_track.year = track.get("Year", "")
                native_track.location = track.get("Location", "")
                
                # Map DJ-specific fields
                native_track.key = track.get("Tonality", "")
                native_track.rating = int(track.get("Rating", "0"))
                native_track.color = track.get("Colour", "")  # Note: UK spelling in Rekordbox
                native_track.play_count = int(track.get("PlayCount", "0"))
                native_track.mix_name = track.get("Mix", "")
                native_track.grouping = track.get("Grouping", "")
                native_track.label = track.get("Label", "")
                
                # Audio properties
                try:
                    native_track.bitrate = int(track.get("BitRate", "0"))
                    native_track.sample_rate = int(track.get("SampleRate", "0"))
                except ValueError:
                    pass
                
                converted_tracks[native_track.track_id] = native_track
                
        except Exception as e:
            self.logger.error(f"Error converting pyrekordbox tracks: {e}")
            return {}
        
        return converted_tracks
    
    def _find_track_by_path_pyrekordbox(self, file_path: str) -> Optional[RekordboxTrack]:
        """Find track by path in pyrekordbox data"""
        try:
            tracks = self._convert_pyrekordbox_tracks()
            for track in tracks.values():
                if track.file_path == file_path or track.location.endswith(file_path):
                    return track
        except Exception as e:
            self.logger.error(f"Error finding track by path: {e}")
        return None
    
    def _update_location_pyrekordbox(self, track_id: str, new_location: str) -> bool:
        """Update track location in pyrekordbox"""
        try:
            track = self.pyrekordbox_xml.get_track(int(track_id))
            if track:
                track["Location"] = f"file://{new_location}"
                return True
        except Exception as e:
            self.logger.error(f"Error updating track location: {e}")
        return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get available capabilities
        
        Returns:
            Dictionary of capability -> availability
        """
        return {
            'pyrekordbox_available': PYREKORDBOX_AVAILABLE,
            'using_pyrekordbox': self.use_pyrekordbox,
            'database_access': self.use_pyrekordbox,  # pyrekordbox can read .db files
            'analysis_files': self.use_pyrekordbox,   # pyrekordbox can read .DAT, .EXT, .2EX
            'advanced_playlists': self.use_pyrekordbox,
            'native_fallback': self.fallback_to_native
        }


def create_rekordbox_adapter(prefer_pyrekordbox: bool = True) -> PyrekordboxAdapter:
    """
    Factory function to create appropriate Rekordbox adapter
    
    Args:
        prefer_pyrekordbox: Whether to prefer pyrekordbox over native implementation
        
    Returns:
        Configured PyrekordboxAdapter
    """
    return PyrekordboxAdapter(
        use_pyrekordbox=prefer_pyrekordbox,
        fallback_to_native=True
    )


__all__ = ['PyrekordboxAdapter', 'create_rekordbox_adapter', 'PYREKORDBOX_AVAILABLE']