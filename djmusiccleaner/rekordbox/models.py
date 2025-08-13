"""
Models for Rekordbox metadata and analysis data.

This module contains data classes that represent Rekordbox-specific metadata,
including beat grid information, cue points, and other DJ-oriented data that
needs to be preserved during round-trip operations.
"""

import copy
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TempoData:
    """Beat grid segment with timing data"""
    inizio: float = 0.0  # Start time in seconds
    bpm: float = 0.0
    metro: str = "4/4"   # Time signature
    battito: int = 1     # Beat number


@dataclass
class PositionMark:
    """Cue point or memory marker"""
    type: int = 0        # 0 = memory, 1 = hot cue
    position: float = 0.0
    name: str = ""
    num: int = 0         # Hot cue number
    color_id: int = 0


@dataclass
class RekordboxTrack:
    """Complete track representation preserving all Rekordbox data"""
    # Core identifiers
    track_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    location: str = ""   # File path
    file_path: str = ""  # Absolute path for reference
    
    # Basic metadata (already handled by main app)
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    comment: str = ""
    year: str = ""
    
    # DJ-specific metadata
    key: str = ""        # Musical key notation
    rating: int = 0      # Star rating (0-5)
    color: str = ""      # Color coding
    play_count: int = 0
    mix_name: str = ""   # Remix/version info
    label: str = ""      # Record label
    grouping: str = ""   # DJ grouping
    
    # Analysis data
    tempo_data: List[TempoData] = field(default_factory=list)
    position_marks: List[PositionMark] = field(default_factory=list)
    beat_grid_adjusted: bool = False
    
    # Audio properties
    bitrate: int = 0
    sample_rate: int = 0
    auto_gain: float = 0.0
    peak_db: float = 0.0
    perceived_loudness: float = 0.0
    
    # Internal fields for round-trip preservation
    _original_attributes: Dict[str, str] = field(default_factory=dict)
    _original_children: List[Any] = field(default_factory=list)
    
    def from_xml_node(self, node):
        """Parse all data from Rekordbox XML TRACK node"""
        # Store ALL attributes for preservation
        self._original_attributes = dict(node.attrib)
        
        # Map basic fields
        self.track_id = node.attrib.get('TrackID', str(uuid.uuid4()))
        self.location = node.attrib.get('Location', '')
        self.title = node.attrib.get('Name', '')
        self.artist = node.attrib.get('Artist', '')
        self.album = node.attrib.get('Album', '')
        self.genre = node.attrib.get('Genre', '')
        self.comment = node.attrib.get('Comments', '')
        self.year = node.attrib.get('Year', '')
        
        # Audio properties
        try:
            self.bitrate = int(node.attrib.get('BitRate', '0'))
            self.sample_rate = int(node.attrib.get('SampleRate', '0'))
        except ValueError:
            self.bitrate = 0
            self.sample_rate = 0
        
        # Map DJ-specific fields
        self.key = node.attrib.get('Tonality', '')
        self.rating = int(node.attrib.get('Rating', '0'))
        self.color = node.attrib.get('Color', '')
        self.play_count = int(node.attrib.get('PlayCount', '0'))
        self.mix_name = node.attrib.get('Mix', '')
        self.grouping = node.attrib.get('Grouping', '')
        self.label = node.attrib.get('Label', '')
        
        # Store all child nodes (deep copy)
        self._original_children = [copy.deepcopy(child) for child in node]
        
        # Extract specific analysis data
        self._parse_tempo_data(node)
        self._parse_position_marks(node)
        
        return self
    
    def _parse_tempo_data(self, node):
        """Extract beat grid information"""
        self.tempo_data = []
        for tempo in node.findall(".//TEMPO"):
            try:
                self.tempo_data.append(TempoData(
                    inizio=float(tempo.attrib.get('Inizio', '0')),
                    bpm=float(tempo.attrib.get('Bpm', '0')),
                    metro=tempo.attrib.get('Metro', '4/4'),
                    battito=int(tempo.attrib.get('Battito', '1'))
                ))
            except (ValueError, TypeError):
                # Skip malformed entries
                pass
    
    def _parse_position_marks(self, node):
        """Extract cue points and memory markers"""
        self.position_marks = []
        for mark in node.findall(".//POSITION_MARK"):
            try:
                self.position_marks.append(PositionMark(
                    type=int(mark.attrib.get('Type', '0')),
                    position=float(mark.attrib.get('Start', '0')),
                    name=mark.attrib.get('Name', ''),
                    num=int(mark.attrib.get('Num', '0')),
                    color_id=int(mark.attrib.get('ColorID', '0'))
                ))
            except (ValueError, TypeError):
                # Skip malformed entries
                pass
    
    def to_xml_node(self, parent_node=None):
        """Generate XML node with all preserved data"""
        track = ET.Element('TRACK')
        
        # First apply all original attributes (complete preservation)
        for key, value in self._original_attributes.items():
            track.attrib[key] = value
        
        # Then override with potentially modified basic fields
        track.attrib['Name'] = self.title
        track.attrib['Artist'] = self.artist
        track.attrib['Album'] = self.album
        
        # Add/update DJ metadata that may have changed
        if self.key:
            track.attrib['Tonality'] = self.key
        if self.color:
            track.attrib['Color'] = self.color
        if self.rating > 0:
            track.attrib['Rating'] = str(self.rating)
        if self.play_count > 0:
            track.attrib['PlayCount'] = str(self.play_count)
        if self.mix_name:
            track.attrib['Mix'] = self.mix_name
        if self.grouping:
            track.attrib['Grouping'] = self.grouping
        if self.label:
            track.attrib['Label'] = self.label
            
        # Re-add all child nodes to preserve structure
        for child in self._original_children:
            track.append(copy.deepcopy(child))
            
        # If no original children but we have generated data, add it
        if not self._original_children:
            self._add_tempo_data(track)
            self._add_position_marks(track)
            
        # If parent provided, add as child, otherwise return node
        if parent_node is not None:
            parent_node.append(track)
        
        return track
        
    def _add_tempo_data(self, track_node):
        """Add tempo data to track if not already present"""
        if not track_node.findall(".//TEMPO") and self.tempo_data:
            for tempo in self.tempo_data:
                tempo_node = ET.SubElement(track_node, 'TEMPO')
                tempo_node.attrib['Inizio'] = str(tempo.inizio)
                tempo_node.attrib['Bpm'] = str(tempo.bpm)
                tempo_node.attrib['Metro'] = tempo.metro
                tempo_node.attrib['Battito'] = str(tempo.battito)

    def _add_position_marks(self, track_node):
        """Add position marks to track if not already present"""
        if not track_node.findall(".//POSITION_MARK") and self.position_marks:
            for mark in self.position_marks:
                mark_node = ET.SubElement(track_node, 'POSITION_MARK')
                mark_node.attrib['Type'] = str(mark.type)
                mark_node.attrib['Start'] = str(mark.position)
                mark_node.attrib['Name'] = mark.name
                mark_node.attrib['Num'] = str(mark.num)
                mark_node.attrib['ColorID'] = str(mark.color_id)