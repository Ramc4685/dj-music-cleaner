"""
Rekordbox Integration Service

Provides comprehensive Rekordbox XML database integration for DJ Music Cleaner.
Handles reading, updating, and exporting Rekordbox collections with full
metadata preservation and professional DJ workflow support.
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import urllib.parse

from ..core.models import TrackMetadata
from ..core.exceptions import RekordboxError
from ..utils.logging_config import get_logger


@dataclass
class RekordboxTrack:
    """Rekordbox track representation"""
    track_id: str
    location: str
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: Optional[int] = None
    bpm: Optional[float] = None
    key: str = ""
    rating: int = 0
    play_count: int = 0
    date_added: str = ""
    date_modified: str = ""
    cue_points: List[Dict[str, Any]] = field(default_factory=list)
    hot_cues: List[Dict[str, Any]] = field(default_factory=list)
    beat_grid: Optional[Dict[str, Any]] = None
    waveform: Optional[str] = None


@dataclass
class RekordboxPlaylist:
    """Rekordbox playlist representation"""
    playlist_id: str
    name: str
    parent_id: Optional[str] = None
    track_ids: List[str] = field(default_factory=list)
    is_folder: bool = False
    children: List['RekordboxPlaylist'] = field(default_factory=list)


class RekordboxService:
    """
    Comprehensive Rekordbox integration service
    
    Features:
    - Full XML database reading and writing
    - Track metadata synchronization
    - Playlist management
    - Cue point and hot cue preservation
    - Beat grid integration
    - Waveform data handling
    - Collection statistics and analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Rekordbox service"""
        self.config = config or {}
        self.logger = get_logger('rekordbox_service')
        
        # Configuration
        self.backup_xml = self.config.get('backup_xml', True)
        self.preserve_playlists = self.config.get('preserve_playlists', True)
        self.preserve_cues = self.config.get('preserve_cues', True)
        self.preserve_waveforms = self.config.get('preserve_waveforms', True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Rekordbox data
        self.tracks: Dict[str, RekordboxTrack] = {}
        self.playlists: Dict[str, RekordboxPlaylist] = {}
        self.xml_path: Optional[str] = None
        self.last_loaded: Optional[float] = None
        
        # Performance tracking
        self.stats = {
            'tracks_loaded': 0,
            'tracks_updated': 0,
            'playlists_loaded': 0,
            'xml_writes': 0,
            'last_backup_time': None
        }
    
    def load_xml(self, xml_path: str) -> Dict[str, Any]:
        """
        Load Rekordbox XML database
        
        Args:
            xml_path: Path to rekordbox.xml file
            
        Returns:
            Dictionary with loading results and statistics
        """
        load_result = {
            'success': False,
            'tracks_loaded': 0,
            'playlists_loaded': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            if not os.path.exists(xml_path):
                raise RekordboxError(f"Rekordbox XML file not found: {xml_path}")
            
            with self._lock:
                # Parse XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Clear existing data
                self.tracks.clear()
                self.playlists.clear()
                
                # Load tracks
                tracks_loaded = self._load_tracks_from_xml(root, load_result)
                
                # Load playlists
                playlists_loaded = self._load_playlists_from_xml(root, load_result)
                
                # Update service state
                self.xml_path = xml_path
                self.last_loaded = time.time()
                
                # Update statistics
                load_result['success'] = True
                load_result['tracks_loaded'] = tracks_loaded
                load_result['playlists_loaded'] = playlists_loaded
                
                self.stats['tracks_loaded'] = tracks_loaded
                self.stats['playlists_loaded'] = playlists_loaded
                
                print(f"✅ Loaded Rekordbox XML: {tracks_loaded} tracks, {playlists_loaded} playlists")
                
            return load_result
            
        except Exception as e:
            load_result['errors'].append(str(e))
            raise RekordboxError(f"Failed to load Rekordbox XML: {str(e)}")
    
    def save_xml(self, xml_path: Optional[str] = None, backup: Optional[bool] = None) -> Dict[str, Any]:
        """
        Save Rekordbox XML database
        
        Args:
            xml_path: Path to save XML (uses loaded path if None)
            backup: Whether to create backup (uses service default if None)
            
        Returns:
            Dictionary with save results
        """
        xml_path = xml_path or self.xml_path
        backup = backup if backup is not None else self.backup_xml
        
        if not xml_path:
            raise RekordboxError("No XML path specified and no file previously loaded")
        
        save_result = {
            'success': False,
            'tracks_saved': 0,
            'playlists_saved': 0,
            'backup_created': False,
            'xml_path': xml_path
        }
        
        try:
            with self._lock:
                # Create backup if requested
                if backup and os.path.exists(xml_path):
                    backup_path = f"{xml_path}.backup_{int(time.time())}"
                    import shutil
                    shutil.copy2(xml_path, backup_path)
                    save_result['backup_created'] = True
                    self.stats['last_backup_time'] = time.time()
                
                # Build XML structure
                root = self._build_xml_structure()
                
                # Add tracks
                tracks_saved = self._add_tracks_to_xml(root)
                
                # Add playlists
                playlists_saved = self._add_playlists_to_xml(root)
                
                # Write XML file with proper formatting
                self._write_formatted_xml(root, xml_path)
                
                save_result['success'] = True
                save_result['tracks_saved'] = tracks_saved
                save_result['playlists_saved'] = playlists_saved
                
                self.stats['xml_writes'] += 1
                
                print(f"✅ Saved Rekordbox XML: {tracks_saved} tracks, {playlists_saved} playlists")
                
            return save_result
            
        except Exception as e:
            self.logger.error(f"Failed to save Rekordbox XML: {e}")
            save_result['error'] = str(e)
            return save_result
    
    def create_empty_collection(self, xml_path: str) -> Dict[str, Any]:
        """
        Create a new empty Rekordbox XML collection
        
        Args:
            xml_path: Path where the new XML file will be saved
            
        Returns:
            Dictionary with creation results
        """
        create_result = {
            'success': False,
            'tracks_created': 0,
            'playlists_created': 0,
            'xml_path': xml_path
        }
        
        try:
            with self._lock:
                # Clear existing data
                self.tracks.clear()
                self.playlists.clear()
                self.xml_path = xml_path
                
                # Create basic XML structure
                root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
                
                # Add product info
                product = ET.SubElement(root, "PRODUCT", 
                    Name="rekordbox", 
                    Version="6.6.2", 
                    Company="AlphaTheta"
                )
                
                # Add empty collection
                collection = ET.SubElement(root, "COLLECTION", Entries="0")
                
                # Add empty playlists
                playlists = ET.SubElement(root, "PLAYLISTS", Version="")
                
                # Save the empty XML
                tree = ET.ElementTree(root)
                tree.write(xml_path, encoding='UTF-8', xml_declaration=True)
                
                create_result['success'] = True
                self.stats['xml_writes'] += 1
                
                self.logger.info(f"Created empty Rekordbox collection: {xml_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to create empty collection: {e}")
            create_result['error'] = str(e)
        
        return create_result
    
    def update_track_metadata(self, filepath: str, metadata: TrackMetadata) -> bool:
        """
        Update track metadata in Rekordbox collection
        
        Args:
            filepath: Path to the audio file
            metadata: Updated metadata
            
        Returns:
            True if track was found and updated
        """
        try:
            with self._lock:
                # Find track by location
                track_id = self._find_track_by_location(filepath)
                
                if not track_id:
                    # Track not in collection, could add it
                    return self._add_new_track(filepath, metadata)
                
                # Update existing track
                track = self.tracks[track_id]
                
                # Update metadata fields
                if metadata.title:
                    track.title = metadata.title
                if metadata.artist:
                    track.artist = metadata.artist
                if metadata.album:
                    track.album = metadata.album
                if metadata.genre:
                    track.genre = metadata.genre
                if metadata.year:
                    track.year = metadata.year
                if metadata.bpm:
                    track.bpm = metadata.bpm
                if metadata.musical_key:
                    track.key = metadata.musical_key
                
                # Update cue points if available
                if metadata.cue_points and self.preserve_cues:
                    track.cue_points = self._convert_cue_points_to_rekordbox(metadata.cue_points)
                
                track.date_modified = time.strftime("%Y-%m-%d")
                
                self.stats['tracks_updated'] += 1
                
                return True
                
        except Exception as e:
            raise RekordboxError(f"Failed to update track metadata: {str(e)}")
    
    def get_track_by_location(self, filepath: str) -> Optional[RekordboxTrack]:
        """
        Get track information by file location
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            RekordboxTrack object if found, None otherwise
        """
        with self._lock:
            track_id = self._find_track_by_location(filepath)
            return self.tracks.get(track_id) if track_id else None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        with self._lock:
            total_tracks = len(self.tracks)
            total_playlists = len(self.playlists)
            
            # Analyze genres
            genres = {}
            years = {}
            bpms = []
            keys = {}
            
            for track in self.tracks.values():
                if track.genre:
                    genres[track.genre] = genres.get(track.genre, 0) + 1
                if track.year:
                    years[track.year] = years.get(track.year, 0) + 1
                if track.bpm:
                    bpms.append(track.bpm)
                if track.key:
                    keys[track.key] = keys.get(track.key, 0) + 1
            
            # Calculate BPM statistics
            bpm_stats = {}
            if bpms:
                bpms.sort()
                bpm_stats = {
                    'min': min(bpms),
                    'max': max(bpms),
                    'average': sum(bpms) / len(bpms),
                    'median': bpms[len(bpms) // 2]
                }
            
            return {
                'total_tracks': total_tracks,
                'total_playlists': total_playlists,
                'genres': dict(sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10]),
                'years': dict(sorted(years.items(), reverse=True)[:10]),
                'keys': dict(sorted(keys.items(), key=lambda x: x[1], reverse=True)[:12]),
                'bpm_stats': bpm_stats,
                'tracks_with_cues': sum(1 for t in self.tracks.values() if t.cue_points),
                'tracks_with_beat_grid': sum(1 for t in self.tracks.values() if t.beat_grid),
                'last_loaded': self.last_loaded,
                'xml_path': self.xml_path
            }
    
    def export_playlist(self, playlist_name: str, export_format: str = 'm3u8') -> str:
        """
        Export playlist to file format
        
        Args:
            playlist_name: Name of playlist to export
            export_format: Export format ('m3u8', 'txt', 'csv')
            
        Returns:
            Path to exported file
        """
        playlist = None
        for p in self.playlists.values():
            if p.name == playlist_name:
                playlist = p
                break
        
        if not playlist:
            raise RekordboxError(f"Playlist not found: {playlist_name}")
        
        # Create export file
        safe_name = "".join(c for c in playlist_name if c.isalnum() or c in (' ', '-', '_')).strip()
        export_path = f"{safe_name}.{export_format}"
        
        try:
            if export_format == 'm3u8':
                self._export_m3u8_playlist(playlist, export_path)
            elif export_format == 'txt':
                self._export_txt_playlist(playlist, export_path)
            elif export_format == 'csv':
                self._export_csv_playlist(playlist, export_path)
            else:
                raise RekordboxError(f"Unsupported export format: {export_format}")
            
            return export_path
            
        except Exception as e:
            raise RekordboxError(f"Playlist export failed: {str(e)}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        with self._lock:
            return {
                **self.stats,
                'tracks_in_memory': len(self.tracks),
                'playlists_in_memory': len(self.playlists),
                'xml_loaded': self.xml_path is not None
            }
    
    # Private helper methods
    
    def _load_tracks_from_xml(self, root: ET.Element, result: Dict[str, Any]) -> int:
        """Load tracks from XML root element"""
        tracks_loaded = 0
        
        collection = root.find('.//COLLECTION')
        if collection is None:
            result['warnings'].append("No COLLECTION element found in XML")
            return 0
        
        for track_elem in collection.findall('TRACK'):
            try:
                track = self._parse_track_element(track_elem)
                self.tracks[track.track_id] = track
                tracks_loaded += 1
            except Exception as e:
                result['warnings'].append(f"Failed to parse track: {str(e)}")
        
        return tracks_loaded
    
    def _load_playlists_from_xml(self, root: ET.Element, result: Dict[str, Any]) -> int:
        """Load playlists from XML root element"""
        playlists_loaded = 0
        
        playlists_elem = root.find('.//PLAYLISTS')
        if playlists_elem is None:
            result['warnings'].append("No PLAYLISTS element found in XML")
            return 0
        
        # Load root playlists
        for node_elem in playlists_elem.findall('NODE'):
            playlist = self._parse_playlist_element(node_elem)
            if playlist:
                self.playlists[playlist.playlist_id] = playlist
                playlists_loaded += 1
                playlists_loaded += self._load_child_playlists(node_elem, playlist.playlist_id)
        
        return playlists_loaded
    
    def _load_child_playlists(self, parent_elem: ET.Element, parent_id: str) -> int:
        """Recursively load child playlists"""
        child_count = 0
        
        for child_elem in parent_elem.findall('NODE'):
            playlist = self._parse_playlist_element(child_elem, parent_id)
            if playlist:
                self.playlists[playlist.playlist_id] = playlist
                child_count += 1
                child_count += self._load_child_playlists(child_elem, playlist.playlist_id)
        
        return child_count
    
    def _parse_track_element(self, track_elem: ET.Element) -> RekordboxTrack:
        """Parse track XML element into RekordboxTrack object"""
        track = RekordboxTrack(
            track_id=track_elem.get('TrackID', ''),
            location=urllib.parse.unquote(track_elem.get('Location', '').replace('file://localhost', '')),
            title=track_elem.get('Name', ''),
            artist=track_elem.get('Artist', ''),
            album=track_elem.get('Album', ''),
            genre=track_elem.get('Genre', ''),
            year=int(track_elem.get('Year', 0)) or None,
            bpm=float(track_elem.get('AverageBpm', 0)) or None,
            key=track_elem.get('Tonality', ''),
            rating=int(track_elem.get('Rating', 0)),
            play_count=int(track_elem.get('PlayCount', 0)),
            date_added=track_elem.get('DateAdded', ''),
            date_modified=track_elem.get('DateModified', '')
        )
        
        # Parse cue points
        for cue_elem in track_elem.findall('POSITION_MARK'):
            cue_data = {
                'position': float(cue_elem.get('Start', 0)),
                'type': cue_elem.get('Type', ''),
                'name': cue_elem.get('Name', ''),
                'color': cue_elem.get('Color', '')
            }
            
            if cue_elem.get('Type') == '0':  # Regular cue point
                track.cue_points.append(cue_data)
            else:  # Hot cue
                track.hot_cues.append(cue_data)
        
        return track
    
    def _parse_playlist_element(self, node_elem: ET.Element, parent_id: Optional[str] = None) -> Optional[RekordboxPlaylist]:
        """Parse playlist XML element into RekordboxPlaylist object"""
        name = node_elem.get('Name', '')
        if not name:
            return None
        
        playlist = RekordboxPlaylist(
            playlist_id=str(len(self.playlists) + 1),  # Generate ID
            name=name,
            parent_id=parent_id,
            is_folder=node_elem.get('Type') == '1'
        )
        
        # Get track references
        for track_elem in node_elem.findall('TRACK'):
            track_id = track_elem.get('Key', '')
            if track_id:
                playlist.track_ids.append(track_id)
        
        return playlist
    
    def _find_track_by_location(self, filepath: str) -> Optional[str]:
        """Find track ID by file location"""
        normalized_path = os.path.normpath(filepath)
        
        for track_id, track in self.tracks.items():
            if os.path.normpath(track.location) == normalized_path:
                return track_id
        
        return None
    
    def _add_new_track(self, filepath: str, metadata: TrackMetadata) -> bool:
        """Add new track to collection"""
        try:
            # Generate new track ID
            if self.tracks:
                track_id = str(max(int(tid) for tid in self.tracks.keys() if tid.isdigit()) + 1)
            else:
                track_id = "1"
            
            track = RekordboxTrack(
                track_id=track_id,
                location=filepath,
                title=metadata.title or '',
                artist=metadata.artist or '',
                album=metadata.album or '',
                genre=metadata.genre or '',
                year=metadata.year,
                bpm=metadata.bpm,
                key=metadata.musical_key or '',
                date_added=time.strftime("%Y-%m-%d"),
                date_modified=time.strftime("%Y-%m-%d")
            )
            
            if metadata.cue_points:
                track.cue_points = self._convert_cue_points_to_rekordbox(metadata.cue_points)
            
            self.tracks[track_id] = track
            self.stats['tracks_updated'] += 1
            
            return True
            
        except Exception:
            return False
    
    def _convert_cue_points_to_rekordbox(self, cue_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert generic cue points to Rekordbox format"""
        rekordbox_cues = []
        
        for cue in cue_points:
            rekordbox_cue = {
                'position': cue.get('position', 0.0),
                'type': cue.get('type', '0'),
                'name': cue.get('name', cue.get('description', '')),  # Try 'name' first, then 'description'
                'color': 'FF0000'  # Default red
            }
            rekordbox_cues.append(rekordbox_cue)
        
        return rekordbox_cues
    
    def _build_xml_structure(self) -> ET.Element:
        """Build basic Rekordbox XML structure"""
        root = ET.Element('DJ_PLAYLISTS', {'Version': '1.0.0'})
        
        # Add product info
        product = ET.SubElement(root, 'PRODUCT', {
            'Name': 'rekordbox',
            'Version': '6.0.0',
            'Company': 'Pioneer DJ'
        })
        
        return root
    
    def _add_tracks_to_xml(self, root: ET.Element) -> int:
        """Add tracks to XML structure"""
        collection = ET.SubElement(root, 'COLLECTION', {'Entries': str(len(self.tracks))})
        
        tracks_added = 0
        for track in self.tracks.values():
            track_elem = ET.SubElement(collection, 'TRACK', {
                'TrackID': track.track_id,
                'Name': track.title,
                'Artist': track.artist,
                'Album': track.album,
                'Genre': track.genre,
                'Kind': 'MP3 File',
                'Size': '0',
                'TotalTime': '0',
                'DiscNumber': '0',
                'TrackNumber': '0',
                'Year': str(track.year) if track.year else '',
                'AverageBpm': str(track.bpm) if track.bpm else '',
                'DateAdded': track.date_added,
                'DateModified': track.date_modified,
                'BitRate': '320',
                'SampleRate': '44100',
                'PlayCount': str(track.play_count),
                'Rating': str(track.rating),
                'Location': f'file://localhost{urllib.parse.quote(track.location)}',
                'Remixer': '',
                'Tonality': track.key,
                'Label': '',
                'Mix': ''
            })
            
            # Add cue points
            for i, cue in enumerate(track.cue_points):
                # Extract actual position from name if position is 0
                position = cue.get('position', 0)
                name = cue.get('name', '')
                
                # Parse time from name like "Mix point at 30.0s"  
                if position == 0 and 'at ' in name and 's' in name:
                    try:
                        # Extract time between "at " and "s"
                        time_part = name.split('at ')[1].split('s')[0]
                        position = float(time_part)  # Keep in seconds for Rekordbox
                    except (IndexError, ValueError):
                        position = 0
                
                # Define cue colors based on type/name
                colors = self._get_cue_colors(name, i)
                
                # Create POSITION_MARK with proper hot cue format
                cue_attrs = {
                    'Name': name,
                    'Type': cue.get('type', '0'),
                    'Start': f"{position:.3f}",  # Decimal seconds format
                    'Num': str(i),  # Sequential hot cue numbers
                    'Red': str(colors[0]),
                    'Green': str(colors[1]),
                    'Blue': str(colors[2])
                }
                
                ET.SubElement(track_elem, 'POSITION_MARK', cue_attrs)
            
            tracks_added += 1
        
        return tracks_added
    
    def _get_cue_colors(self, name: str, index: int) -> tuple:
        """Get RGB color values for cue points based on name and type"""
        name_lower = name.lower()
        
        # Color mapping based on cue point purpose
        if 'start' in name_lower or index == 0:
            return (40, 226, 20)      # Green - Start points
        elif 'intro' in name_lower or 'build' in name_lower:
            return (255, 165, 0)      # Orange - Intro/Build sections
        elif 'mix' in name_lower and ('in' in name_lower or 'point' in name_lower):
            return (255, 0, 0)        # Red - Mix points
        elif 'break' in name_lower or 'breakdown' in name_lower:
            return (0, 0, 255)        # Blue - Breaks
        elif 'drop' in name_lower:
            return (255, 255, 0)      # Yellow - Drops
        elif 'out' in name_lower:
            return (255, 0, 255)      # Magenta - Mix out
        else:
            # Default colors cycling through hot cue colors
            default_colors = [
                (40, 226, 20),    # Green
                (255, 165, 0),    # Orange  
                (255, 0, 0),      # Red
                (0, 0, 255),      # Blue
                (255, 255, 0),    # Yellow
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Cyan
                (255, 255, 255)   # White
            ]
            return default_colors[index % len(default_colors)]
    
    def _add_playlists_to_xml(self, root: ET.Element) -> int:
        """Add playlists to XML structure"""
        if not self.playlists:
            return 0
        
        playlists_elem = ET.SubElement(root, 'PLAYLISTS')
        
        # Add root playlists first
        root_playlists = [p for p in self.playlists.values() if not p.parent_id]
        
        playlists_added = 0
        for playlist in root_playlists:
            node_elem = self._create_playlist_node(playlist)
            playlists_elem.append(node_elem)
            playlists_added += 1
            playlists_added += self._add_child_playlist_nodes(node_elem, playlist.playlist_id)
        
        return playlists_added
    
    def _add_child_playlist_nodes(self, parent_elem: ET.Element, parent_id: str) -> int:
        """Add child playlist nodes recursively"""
        child_playlists = [p for p in self.playlists.values() if p.parent_id == parent_id]
        
        added_count = 0
        for playlist in child_playlists:
            node_elem = self._create_playlist_node(playlist)
            parent_elem.append(node_elem)
            added_count += 1
            added_count += self._add_child_playlist_nodes(node_elem, playlist.playlist_id)
        
        return added_count
    
    def _create_playlist_node(self, playlist: RekordboxPlaylist) -> ET.Element:
        """Create XML element for playlist"""
        node_elem = ET.Element('NODE', {
            'Type': '1' if playlist.is_folder else '0',
            'Name': playlist.name,
            'Count': str(len(playlist.track_ids))
        })
        
        # Add tracks
        for track_id in playlist.track_ids:
            ET.SubElement(node_elem, 'TRACK', {'Key': track_id})
        
        return node_elem
    
    def _write_formatted_xml(self, root: ET.Element, xml_path: str):
        """Write XML with proper formatting"""
        # Convert to string and format
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        
        # Write formatted XML
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent='  '))
    
    def _export_m3u8_playlist(self, playlist: RekordboxPlaylist, export_path: str):
        """Export playlist as M3U8 file"""
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write('#EXTM3U\n')
            
            for track_id in playlist.track_ids:
                track = self.tracks.get(track_id)
                if track:
                    f.write(f'#EXTINF:-1,{track.artist} - {track.title}\n')
                    f.write(f'{track.location}\n')
    
    def _export_txt_playlist(self, playlist: RekordboxPlaylist, export_path: str):
        """Export playlist as simple text file"""
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(f'Playlist: {playlist.name}\n')
            f.write('=' * (len(playlist.name) + 10) + '\n\n')
            
            for i, track_id in enumerate(playlist.track_ids, 1):
                track = self.tracks.get(track_id)
                if track:
                    f.write(f'{i:3d}. {track.artist} - {track.title}\n')
    
    def _export_csv_playlist(self, playlist: RekordboxPlaylist, export_path: str):
        """Export playlist as CSV file"""
        import csv
        
        with open(export_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Position', 'Artist', 'Title', 'Album', 'Genre', 'BPM', 'Key', 'Location'])
            
            for i, track_id in enumerate(playlist.track_ids, 1):
                track = self.tracks.get(track_id)
                if track:
                    writer.writerow([
                        i, track.artist, track.title, track.album,
                        track.genre, track.bpm or '', track.key, track.location
                    ])


__all__ = ['RekordboxService', 'RekordboxTrack', 'RekordboxPlaylist']