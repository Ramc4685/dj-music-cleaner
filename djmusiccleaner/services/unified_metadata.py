"""
Unified Metadata Service

Consolidates metadata functionality from both implementations:
1. Enhanced metadata extraction with validation
2. Online metadata enhancement (MusicBrainz, AcoustID)
3. Professional metadata cleaning and normalization  
4. Format-specific handling (ID3, FLAC, etc.)
"""

import os
import sys
import json
import time
import threading
from typing import Dict, Any, Optional, List, Set, Tuple
from pathlib import Path
import re

# Metadata library imports
try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3NoHeaderError, ID3
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.mp4 import MP4
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# Online enhancement imports
try:
    import acoustid
    import musicbrainzngs
    ACOUSTID_AVAILABLE = True
except ImportError:
    ACOUSTID_AVAILABLE = False

import requests
from ..core.models import TrackMetadata
from ..core.exceptions import MetadataError
from ..utils.text import sanitize_tag_value, clean_text, extract_remix_info, detect_featuring, standardize_genre


class UnifiedMetadataService:
    """
    Unified metadata service combining all metadata functionality
    
    Features:
    - Comprehensive metadata extraction from all audio formats
    - Online metadata enhancement via MusicBrainz/AcoustID
    - Professional metadata cleaning and normalization
    - Format-specific optimizations
    - Validation and error reporting
    - Performance tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified metadata service"""
        self.config = config or {}
        
        # Configuration
        self.enable_online_lookup = self.config.get('enable_online_lookup', True)
        self.online_timeout = self.config.get('online_timeout', 10)
        self.cleanup_metadata = self.config.get('cleanup_metadata', True)
        self.aggressive_cleanup = self.config.get('aggressive_cleanup', False)
        
        # API configuration
        self.acoustid_api_key = self.config.get('acoustid_api_key', 'default_key')
        self.musicbrainz_app_name = self.config.get('musicbrainz_app_name', 'DJ Music Cleaner')
        self.musicbrainz_app_version = self.config.get('musicbrainz_app_version', '2.0')
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            'extractions': 0,
            'online_lookups': 0,
            'successful_lookups': 0,
            'cleanup_operations': 0,
            'validation_errors': 0,
            'total_processing_time': 0.0
        }
        
        # Initialize services
        self._init_online_services()
        
        # Validate dependencies
        if not MUTAGEN_AVAILABLE:
            raise MetadataError("Mutagen library not available - required for metadata extraction")
    
    def _init_online_services(self):
        """Initialize online metadata services"""
        if ACOUSTID_AVAILABLE and self.enable_online_lookup:
            try:
                # Configure MusicBrainz
                musicbrainzngs.set_useragent(
                    self.musicbrainz_app_name,
                    self.musicbrainz_app_version
                )
                print(f"   🌐 Online metadata services initialized")
            except Exception as e:
                print(f"   ⚠️ Online services initialization warning: {e}")
    
    def extract_metadata(self, filepath: str, enhance_online: bool = True, acoustid_api_key: Optional[str] = None) -> TrackMetadata:
        """
        Extract comprehensive metadata from audio file
        
        Args:
            filepath: Path to audio file
            enhance_online: Whether to perform online metadata enhancement
            acoustid_api_key: AcoustID API key for fingerprinting (overrides config)
            
        Returns:
            TrackMetadata object with extracted information
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self.stats['extractions'] += 1
            
            # Initialize metadata object
            metadata = TrackMetadata()
            metadata.filepath = filepath
            metadata.filename = os.path.basename(filepath)
            
            # Get file information
            self._extract_file_info(filepath, metadata)
            
            # Extract basic metadata using mutagen
            self._extract_basic_metadata(filepath, metadata)
            
            # Clean and normalize metadata
            if self.cleanup_metadata:
                self._clean_metadata(metadata)
            
            # Online enhancement
            if enhance_online and self.enable_online_lookup:
                self._enhance_metadata_online(filepath, metadata, acoustid_api_key)
            
            # Validate metadata
            validation_errors = self._validate_metadata(metadata)
            metadata.validation_errors = validation_errors
            
            # Update performance stats
            processing_time = time.time() - start_time
            metadata.processing_time = processing_time
            
            with self._lock:
                self.stats['total_processing_time'] += processing_time
                if validation_errors:
                    self.stats['validation_errors'] += 1
            
            return metadata
            
        except Exception as e:
            raise MetadataError(
                f"Metadata extraction failed: {str(e)}",
                filepath=filepath
            )
    
    def _extract_file_info(self, filepath: str, metadata: TrackMetadata):
        """Extract basic file information"""
        try:
            stat_info = os.stat(filepath)
            metadata.filesize = stat_info.st_size
            metadata.last_modified = stat_info.st_mtime
            
            # Determine format from extension
            ext = Path(filepath).suffix.lower()
            format_map = {
                '.mp3': 'MP3',
                '.flac': 'FLAC', 
                '.m4a': 'M4A',
                '.mp4': 'MP4',
                '.wav': 'WAV',
                '.aiff': 'AIFF',
                '.ogg': 'OGG',
                '.wma': 'WMA'
            }
            metadata.format = format_map.get(ext, ext.lstrip('.').upper())
            
        except Exception as e:
            metadata.validation_errors.append(f"File info extraction failed: {e}")
    
    def _extract_basic_metadata(self, filepath: str, metadata: TrackMetadata):
        """Extract metadata using mutagen"""
        try:
            audio_file = MutagenFile(filepath)
            if not audio_file:
                raise MetadataError("Could not read audio file")
            
            # Get audio properties
            if hasattr(audio_file, 'info'):
                info = audio_file.info
                metadata.duration = getattr(info, 'length', 0.0)
                metadata.bitrate = getattr(info, 'bitrate', 0)
            
            # Extract tags based on file format
            if hasattr(audio_file, 'tags') and audio_file.tags:
                if isinstance(audio_file, MP3):
                    self._extract_id3_tags(audio_file.tags, metadata)
                elif isinstance(audio_file, FLAC):
                    self._extract_vorbis_tags(audio_file.tags, metadata)
                elif isinstance(audio_file, MP4):
                    self._extract_mp4_tags(audio_file.tags, metadata)
                else:
                    self._extract_generic_tags(audio_file.tags, metadata)
            
        except Exception as e:
            metadata.validation_errors.append(f"Basic metadata extraction failed: {e}")
    
    def _extract_id3_tags(self, tags, metadata: TrackMetadata):
        """Extract ID3 tags from MP3 files"""
        try:
            # Basic tags
            metadata.title = self._get_tag_value(tags, ['TIT2', 'TITLE'])
            metadata.artist = self._get_tag_value(tags, ['TPE1', 'ARTIST'])
            metadata.album = self._get_tag_value(tags, ['TALB', 'ALBUM'])
            metadata.album_artist = self._get_tag_value(tags, ['TPE2', 'ALBUMARTIST'])
            metadata.genre = self._get_tag_value(tags, ['TCON', 'GENRE'])
            
            # Year handling
            year_str = self._get_tag_value(tags, ['TDRC', 'TYER', 'DATE', 'YEAR'])
            if year_str:
                metadata.year = self._parse_year(year_str)
            
            # Track number
            track_str = self._get_tag_value(tags, ['TRCK', 'TRACKNUMBER'])
            if track_str:
                metadata.track_number = self._parse_track_number(track_str)
            
            # Extended tags
            metadata.composer = self._get_tag_value(tags, ['TCOM', 'COMPOSER'])
            metadata.publisher = self._get_tag_value(tags, ['TPUB', 'PUBLISHER'])
            metadata.isrc = self._get_tag_value(tags, ['TSRC', 'ISRC'])
            metadata.label = self._get_tag_value(tags, ['TPUB', 'LABEL'])
            
            # Custom DJ-related tags
            metadata.bpm = self._parse_bpm(self._get_tag_value(tags, ['TBPM', 'BPM']))
            metadata.musical_key = self._get_tag_value(tags, ['TKEY', 'KEY', 'INITIALKEY'])
            
        except Exception as e:
            metadata.validation_errors.append(f"ID3 tag extraction failed: {e}")
    
    def _extract_vorbis_tags(self, tags, metadata: TrackMetadata):
        """Extract Vorbis comments from FLAC files"""
        try:
            # Basic tags
            metadata.title = self._get_vorbis_tag(tags, ['TITLE'])
            metadata.artist = self._get_vorbis_tag(tags, ['ARTIST'])
            metadata.album = self._get_vorbis_tag(tags, ['ALBUM'])
            metadata.album_artist = self._get_vorbis_tag(tags, ['ALBUMARTIST', 'ALBUM_ARTIST'])
            metadata.genre = self._get_vorbis_tag(tags, ['GENRE'])
            
            # Year
            year_str = self._get_vorbis_tag(tags, ['DATE', 'YEAR'])
            if year_str:
                metadata.year = self._parse_year(year_str)
            
            # Track number
            track_str = self._get_vorbis_tag(tags, ['TRACKNUMBER', 'TRACK'])
            if track_str:
                metadata.track_number = self._parse_track_number(track_str)
            
            # Extended tags
            metadata.composer = self._get_vorbis_tag(tags, ['COMPOSER'])
            metadata.publisher = self._get_vorbis_tag(tags, ['PUBLISHER'])
            metadata.isrc = self._get_vorbis_tag(tags, ['ISRC'])
            metadata.label = self._get_vorbis_tag(tags, ['LABEL'])
            
            # DJ-related
            metadata.bpm = self._parse_bpm(self._get_vorbis_tag(tags, ['BPM']))
            metadata.musical_key = self._get_vorbis_tag(tags, ['KEY', 'INITIALKEY'])
            
        except Exception as e:
            metadata.validation_errors.append(f"Vorbis tag extraction failed: {e}")
    
    def _extract_mp4_tags(self, tags, metadata: TrackMetadata):
        """Extract MP4 tags"""
        try:
            # Basic tags
            metadata.title = self._get_mp4_tag(tags, ['\xa9nam'])
            metadata.artist = self._get_mp4_tag(tags, ['\xa9ART'])
            metadata.album = self._get_mp4_tag(tags, ['\xa9alb'])
            metadata.album_artist = self._get_mp4_tag(tags, ['aART'])
            metadata.genre = self._get_mp4_tag(tags, ['\xa9gen'])
            
            # Year
            year_str = self._get_mp4_tag(tags, ['\xa9day'])
            if year_str:
                metadata.year = self._parse_year(year_str)
            
            # Track number
            track_data = tags.get('trkn')
            if track_data:
                metadata.track_number = track_data[0][0]
            
            # Extended tags
            metadata.composer = self._get_mp4_tag(tags, ['\xa9wrt'])
            
        except Exception as e:
            metadata.validation_errors.append(f"MP4 tag extraction failed: {e}")
    
    def _extract_generic_tags(self, tags, metadata: TrackMetadata):
        """Extract tags from other formats"""
        try:
            # Try common tag names
            tag_mapping = {
                'title': ['title', 'TITLE'],
                'artist': ['artist', 'ARTIST'], 
                'album': ['album', 'ALBUM'],
                'genre': ['genre', 'GENRE'],
                'date': ['date', 'DATE', 'year', 'YEAR'],
                'tracknumber': ['tracknumber', 'TRACKNUMBER', 'track', 'TRACK']
            }
            
            for attr, tag_names in tag_mapping.items():
                value = None
                for tag_name in tag_names:
                    if tag_name in tags:
                        value = tags[tag_name]
                        if isinstance(value, list) and value:
                            value = str(value[0])
                        else:
                            value = str(value)
                        break
                
                if value:
                    if attr == 'date':
                        metadata.year = self._parse_year(value)
                    elif attr == 'tracknumber':
                        metadata.track_number = self._parse_track_number(value)
                    else:
                        setattr(metadata, attr, value)
                        
        except Exception as e:
            metadata.validation_errors.append(f"Generic tag extraction failed: {e}")
    
    def _clean_metadata(self, metadata: TrackMetadata):
        """Clean and normalize metadata"""
        try:
            with self._lock:
                self.stats['cleanup_operations'] += 1
            
            # Clean text fields
            text_fields = [
                'title', 'artist', 'album', 'album_artist', 'genre',
                'composer', 'publisher', 'label', 'musical_key'
            ]
            
            for field in text_fields:
                value = getattr(metadata, field, '')
                if value:
                    if self.aggressive_cleanup:
                        cleaned = sanitize_tag_value(value)
                    else:
                        cleaned = clean_text(value)
                    setattr(metadata, field, cleaned)
            
            # Extract remix information
            if metadata.title:
                clean_title, remix_info = extract_remix_info(metadata.title)
                if remix_info:
                    metadata.title = clean_title
                    # Store remix info in a custom field if needed
            
            # Extract featuring artists
            if metadata.artist:
                clean_artist, featuring = detect_featuring(metadata.artist)
                if featuring:
                    metadata.artist = clean_artist
                    # Could store featuring artists separately
            
            # Standardize genre
            if metadata.genre:
                metadata.genre = standardize_genre(metadata.genre)
            
            # Normalize keys
            if metadata.musical_key:
                metadata.musical_key = self._normalize_key(metadata.musical_key)
            
        except Exception as e:
            metadata.validation_errors.append(f"Metadata cleanup failed: {e}")
    
    def _enhance_metadata_online(self, filepath: str, metadata: TrackMetadata, acoustid_api_key: Optional[str] = None):
        """Enhance metadata using online services"""
        if not ACOUSTID_AVAILABLE:
            return
        
        try:
            with self._lock:
                self.stats['online_lookups'] += 1
            
            # Use AcoustID for fingerprinting
            try:
                duration, fingerprint = acoustid.fingerprint_file(filepath)
                
                # Use provided API key or fall back to config
                api_key = acoustid_api_key or self.acoustid_api_key or os.environ.get('ACOUSTID_API_KEY')
                
                if not api_key or api_key == 'default_key':
                    self.logger.warning("No valid AcoustID API key available. Skipping fingerprint lookup.")
                    return
                
                # Look up in AcoustID database
                results = acoustid.lookup(
                    api_key,
                    fingerprint,
                    duration,
                    meta='recordings+releasegroups+compress'
                )
                
                # Process results
                for score, recording_id, title, artist in results['results']:
                    if score > 0.8:  # High confidence match
                        # Enhance metadata with online data
                        if not metadata.title and title:
                            metadata.title = clean_text(title)
                        if not metadata.artist and artist:
                            metadata.artist = clean_text(artist)
                        
                        metadata.online_enhanced = True
                        metadata.acoustid_id = recording_id
                        
                        with self._lock:
                            self.stats['successful_lookups'] += 1
                        break
                        
            except Exception as e:
                metadata.validation_errors.append(f"Online enhancement failed: {e}")
                
        except Exception as e:
            metadata.validation_errors.append(f"Online lookup failed: {e}")
    
    def _validate_metadata(self, metadata: TrackMetadata) -> List[str]:
        """Validate extracted metadata and return list of issues"""
        issues = []
        
        try:
            # Check for essential fields
            if not metadata.title:
                issues.append("Missing title")
            if not metadata.artist:
                issues.append("Missing artist")
            
            # Validate data types and ranges
            if metadata.year and (metadata.year < 1900 or metadata.year > 2030):
                issues.append(f"Invalid year: {metadata.year}")
            
            if metadata.track_number and (metadata.track_number < 1 or metadata.track_number > 999):
                issues.append(f"Invalid track number: {metadata.track_number}")
            
            if metadata.bpm and (metadata.bpm < 20 or metadata.bpm > 300):
                issues.append(f"Invalid BPM: {metadata.bpm}")
            
            if metadata.duration and metadata.duration < 1:
                issues.append("Invalid duration")
            
            # Check for suspicious content
            if metadata.title and len(metadata.title) > 200:
                issues.append("Title suspiciously long")
            
            if metadata.artist and len(metadata.artist) > 100:
                issues.append("Artist name suspiciously long")
            
            # Check for pollution in text fields
            pollution_patterns = [
                r'www\.',
                r'\.com',
                r'free\s+download',
                r'\d{3,4}\s*kbps',
                r'promo\s+only'
            ]
            
            for field_name in ['title', 'artist', 'album']:
                value = getattr(metadata, field_name, '')
                if value:
                    for pattern in pollution_patterns:
                        if re.search(pattern, value.lower()):
                            issues.append(f"Possible pollution in {field_name}: {pattern}")
                            break
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return issues
    
    def update_metadata(self, filepath: str, metadata: TrackMetadata, 
                       backup: bool = True) -> bool:
        """
        Update metadata tags in audio file
        
        Args:
            filepath: Path to audio file
            metadata: TrackMetadata object with new values
            backup: Whether to create backup before updating
            
        Returns:
            True if update was successful
        """
        try:
            # Create backup if requested
            if backup:
                backup_path = f"{filepath}.backup"
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(filepath, backup_path)
            
            # Open file for writing
            audio_file = MutagenFile(filepath)
            if not audio_file:
                raise MetadataError("Could not open file for writing")
            
            # Update tags based on format
            if isinstance(audio_file, MP3):
                self._update_id3_tags(audio_file, metadata)
            elif isinstance(audio_file, FLAC):
                self._update_vorbis_tags(audio_file, metadata)
            elif isinstance(audio_file, MP4):
                self._update_mp4_tags(audio_file, metadata)
            else:
                self._update_generic_tags(audio_file, metadata)
            
            # Save changes
            audio_file.save()
            
            return True
            
        except Exception as e:
            raise MetadataError(
                f"Failed to update metadata: {str(e)}",
                filepath=filepath
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            avg_time = (self.stats['total_processing_time'] / self.stats['extractions'] 
                       if self.stats['extractions'] > 0 else 0)
            
            lookup_success_rate = (self.stats['successful_lookups'] / self.stats['online_lookups'] * 100
                                 if self.stats['online_lookups'] > 0 else 0)
            
            return {
                'total_extractions': self.stats['extractions'],
                'online_lookups': self.stats['online_lookups'],
                'successful_lookups': self.stats['successful_lookups'],
                'lookup_success_rate': round(lookup_success_rate, 1),
                'cleanup_operations': self.stats['cleanup_operations'],
                'validation_errors': self.stats['validation_errors'],
                'average_processing_time': round(avg_time, 3),
                'total_processing_time': round(self.stats['total_processing_time'], 2),
                'services_available': {
                    'mutagen': MUTAGEN_AVAILABLE,
                    'acoustid': ACOUSTID_AVAILABLE,
                    'online_lookup_enabled': self.enable_online_lookup
                }
            }
    
    # Private helper methods
    
    def _get_tag_value(self, tags, tag_names: List[str]) -> str:
        """Get tag value from ID3 tags"""
        for tag_name in tag_names:
            if tag_name in tags:
                value = tags[tag_name]
                if hasattr(value, 'text'):
                    return str(value.text[0]) if value.text else ''
                return str(value)
        return ''
    
    def _get_vorbis_tag(self, tags, tag_names: List[str]) -> str:
        """Get tag value from Vorbis comments"""
        for tag_name in tag_names:
            if tag_name in tags:
                value = tags[tag_name]
                return str(value[0]) if isinstance(value, list) and value else str(value)
        return ''
    
    def _get_mp4_tag(self, tags, tag_names: List[str]) -> str:
        """Get tag value from MP4 tags"""
        for tag_name in tag_names:
            if tag_name in tags:
                value = tags[tag_name]
                return str(value[0]) if isinstance(value, list) and value else str(value)
        return ''
    
    def _parse_year(self, year_str: str) -> Optional[int]:
        """Parse year from string"""
        if not year_str:
            return None
        
        # Extract 4-digit year
        year_match = re.search(r'(\d{4})', str(year_str))
        if year_match:
            year = int(year_match.group(1))
            if 1900 <= year <= 2030:
                return year
        return None
    
    def _parse_track_number(self, track_str: str) -> Optional[int]:
        """Parse track number from string"""
        if not track_str:
            return None
        
        # Handle "track/total" format
        track_match = re.search(r'^(\d+)', str(track_str))
        if track_match:
            track_num = int(track_match.group(1))
            if 1 <= track_num <= 999:
                return track_num
        return None
    
    def _parse_bpm(self, bpm_str: str) -> Optional[float]:
        """Parse BPM from string"""
        if not bpm_str:
            return None
        
        try:
            bpm = float(str(bpm_str))
            if 20 <= bpm <= 300:
                return round(bpm, 1)
        except:
            pass
        return None
    
    def _normalize_key(self, key_str: str) -> str:
        """Normalize musical key notation"""
        if not key_str:
            return ''
        
        # Common key normalizations
        key_map = {
            'ab': 'Ab', 'a#': 'A#', 'bb': 'Bb', 'c#': 'C#', 
            'db': 'Db', 'd#': 'D#', 'eb': 'Eb', 'f#': 'F#',
            'gb': 'Gb', 'g#': 'G#'
        }
        
        cleaned = key_str.strip()
        
        # Handle minor keys
        if cleaned.lower().endswith('m'):
            base = cleaned[:-1].lower()
            if base in key_map:
                return key_map[base] + 'm'
            else:
                return cleaned[:-1].capitalize() + 'm'
        else:
            lower_key = cleaned.lower()
            if lower_key in key_map:
                return key_map[lower_key]
            else:
                return cleaned.capitalize()
    
    def _update_id3_tags(self, audio_file, metadata: TrackMetadata):
        """Update ID3 tags"""
        # Implementation for updating ID3 tags
        # This would be a comprehensive method to write back all metadata
        pass
    
    def _update_vorbis_tags(self, audio_file, metadata: TrackMetadata):
        """Update Vorbis comment tags"""
        # Implementation for updating Vorbis tags
        pass
    
    def _update_mp4_tags(self, audio_file, metadata: TrackMetadata):
        """Update MP4 tags"""
        # Implementation for updating MP4 tags
        pass
    
    def _update_generic_tags(self, audio_file, metadata: TrackMetadata):
        """Update generic format tags"""
        # Implementation for updating other format tags
        pass


__all__ = ['UnifiedMetadataService']