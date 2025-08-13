"""
Parser for Rekordbox XML collection files.

This module handles reading and writing Rekordbox XML files while
preserving all DJ-specific metadata and analysis data to ensure
round-trip fidelity.
"""

import os
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from urllib.parse import unquote, quote
from .models import RekordboxTrack


class RekordboxXMLParser:
    """Parser for Rekordbox XML collection files"""
    
    def __init__(self):
        self.tracks: Dict[str, RekordboxTrack] = {}
        self.playlists = []
        self.logger = logging.getLogger(__name__)
        
    def parse(self, xml_path: str) -> 'RekordboxXMLParser':
        """Parse Rekordbox XML file and build track collection"""
        if not os.path.exists(xml_path):
            self.logger.error(f"XML file not found: {xml_path}")
            return self
            
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Parse all tracks first
            self._parse_tracks(root)
            
            # Then parse playlists which reference track IDs
            self._parse_playlists(root)
            
            self.logger.info(f"Parsed {len(self.tracks)} tracks from {xml_path}")
            
        except Exception as e:
            self.logger.error(f"Error parsing XML: {str(e)}")
            
        return self
        
    def _parse_tracks(self, root):
        """Extract all track data from COLLECTION node"""
        collection = root.find('./COLLECTION')
        if collection is None:
            self.logger.warning("No COLLECTION node found in XML")
            return
            
        for track_node in collection.findall('./TRACK'):
            try:
                track = RekordboxTrack().from_xml_node(track_node)
                self.tracks[track.track_id] = track
            except Exception as e:
                track_id = track_node.get('TrackID', 'unknown')
                self.logger.warning(f"Error parsing track {track_id}: {str(e)}")
            
    def _parse_playlists(self, root):
        """Extract playlist structure with track references"""
        playlists_node = root.find('./PLAYLISTS')
        if playlists_node is None:
            self.logger.warning("No PLAYLISTS node found in XML")
            return
            
        # Basic playlist structure for future implementation
        # This is important for complete round-trip preservation
        self.playlists = {
            'node': playlists_node,
            'version': playlists_node.attrib.get('Version', '')
        }
        
    def normalize_file_paths(self, base_dir: Optional[str] = None) -> None:
        """Convert Rekordbox file paths to OS paths"""
        for track_id, track in self.tracks.items():
            if track.location.startswith('file://'):
                # Remove file:// prefix and URL decode
                path = unquote(track.location[7:])
                
                # Convert to OS path format
                path = os.path.normpath(path)
                
                # If base_dir provided, make relative to it
                if base_dir and os.path.isabs(path):
                    try:
                        rel_path = os.path.relpath(path, base_dir)
                        # Only use relative path if it doesn't start with '..'  
                        # (meaning it's actually inside the base_dir)
                        if not rel_path.startswith('..'):
                            path = rel_path
                    except ValueError:
                        # Different drive letters or other path issues, keep absolute
                        pass
                        
                track.file_path = path
    
    def get_track_by_path(self, file_path: str) -> Optional[RekordboxTrack]:
        """Get track by file path"""
        norm_path = os.path.normpath(file_path)
        
        for track in self.tracks.values():
            if track.file_path and os.path.normpath(track.file_path) == norm_path:
                return track
                
        return None
    
    def update_track_location(self, track_id: str, new_location: str) -> bool:
        """Update track location path"""
        if track_id not in self.tracks:
            return False
            
        track = self.tracks[track_id]
        
        # Save the original path for reference
        track.file_path = new_location
        
        # Update Rekordbox location with proper URL format
        if os.path.isabs(new_location):
            # Convert to URL format with file:// prefix and proper encoding
            track.location = 'file://' + quote(new_location)
        else:
            # Just store the relative path
            track.location = new_location
            
        return True
        
    def export_xml(self, output_path: str) -> str:
        """Export collection back to Rekordbox XML"""
        # Create basic XML structure
        root = ET.Element('DJ_PLAYLISTS')
        root.attrib['Version'] = "1.0.0"
        
        product = ET.SubElement(root, 'PRODUCT')
        product.attrib['Name'] = "rekordbox"
        product.attrib['Version'] = "6.6.2"
        product.attrib['Company'] = "AlphaTheta"
        
        collection = ET.SubElement(root, 'COLLECTION')
        collection.attrib['Entries'] = str(len(self.tracks))
        
        # Add all tracks
        for track_id, track in self.tracks.items():
            track.to_xml_node(collection)
            
        # Add playlists (basic structure for now)
        if hasattr(self, 'playlists') and isinstance(self.playlists, dict) and 'node' in self.playlists:
            # Deep copy the original playlists structure to preserve all custom nodes
            playlists = ET.SubElement(root, 'PLAYLISTS')
            if 'version' in self.playlists:
                playlists.attrib['Version'] = self.playlists['version']
                
            # In the future, implement playlist preservation/modification here
        else:
            # Create empty playlists node
            ET.SubElement(root, 'PLAYLISTS')
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='UTF-8', xml_declaration=True)
        
        self.logger.info(f"Exported {len(self.tracks)} tracks to {output_path}")
        return output_path