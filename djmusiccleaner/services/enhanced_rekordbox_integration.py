"""
Enhanced Rekordbox Integration Service

This service ensures that all processed metadata, analysis results, and enhanced
track information are properly integrated into Rekordbox XML with complete
DJ-specific data preservation and enhancement.
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
from xml.dom import minidom
import urllib.parse
import hashlib

from ..core.models import TrackMetadata, ProcessingResult
from ..core.exceptions import RekordboxError
from ..utils.logging_config import get_logger, get_app_logger
from ..services.rekordbox import RekordboxService, RekordboxTrack


@dataclass 
class EnhancedRekordboxTrack:
    """Enhanced track representation with all DJ Music Cleaner data"""
    # Core track info
    track_id: str
    location: str
    original_location: Optional[str] = None  # If file was moved/renamed
    
    # Basic metadata (enhanced by cleaner)
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: Optional[int] = None
    comment: str = ""
    
    # Audio analysis results
    bpm: Optional[float] = None
    musical_key: str = ""
    energy_level: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None
    
    # Advanced analysis
    cue_points: List[Dict[str, Any]] = None
    beat_grid: Optional[Dict[str, Any]] = None
    energy_profile: Optional[Dict[str, Any]] = None
    
    # DJ metadata
    rating: int = 0
    color: str = ""
    play_count: int = 0
    grouping: str = ""
    label: str = ""
    mix_name: str = ""
    
    # Quality and technical info
    bitrate: int = 0
    sample_rate: int = 0
    is_high_quality: bool = False
    audio_format: str = ""
    
    # Processing metadata
    processed_by_cleaner: bool = False
    processing_date: str = ""
    cleanup_applied: bool = False
    online_enhanced: bool = False
    
    def __post_init__(self):
        if self.cue_points is None:
            self.cue_points = []


class EnhancedRekordboxIntegration:
    """
    Enhanced Rekordbox integration ensuring all cleaned metadata is properly imported
    
    Features:
    - Comprehensive metadata mapping from cleaner results to Rekordbox
    - Advanced analysis data integration (cue points, beat grids, energy profiles)
    - Smart file location tracking for renamed/moved files
    - Backup and rollback capabilities
    - Detailed import/export reporting
    - Playlist preservation with enhanced metadata
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced Rekordbox integration"""
        self.config = config or {}
        self.logger = get_logger('rekordbox_enhanced')
        self.app_logger = get_app_logger()
        
        # Base Rekordbox service
        self.rekordbox_service = RekordboxService(config)
        
        # Enhanced tracking
        self.enhanced_tracks: Dict[str, EnhancedRekordboxTrack] = {}
        self.file_mappings: Dict[str, str] = {}  # original -> new path
        self.processing_results: Dict[str, ProcessingResult] = {}
        
        # Configuration
        self.preserve_existing_data = self.config.get('preserve_existing_data', True)
        self.backup_before_update = self.config.get('backup_before_update', True)
        self.add_processing_metadata = self.config.get('add_processing_metadata', True)
        self.sync_advanced_analysis = self.config.get('sync_advanced_analysis', True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'tracks_loaded': 0,
            'tracks_enhanced': 0,
            'tracks_updated': 0,
            'tracks_added': 0,
            'cue_points_synced': 0,
            'beat_grids_synced': 0,
            'playlists_preserved': 0,
            'backups_created': 0
        }
    
    def load_rekordbox_collection(self, xml_path: str) -> Dict[str, Any]:
        """
        Load Rekordbox collection and prepare for enhancement
        
        Args:
            xml_path: Path to rekordbox.xml file (will create new if doesn't exist)
            
        Returns:
            Loading results with enhancement preparation status
        """
        self.logger.info(f"Loading Rekordbox collection: {xml_path}")
        
        try:
            # Check if file exists
            if os.path.exists(xml_path):
                # Load existing collection
                load_result = self.rekordbox_service.load_xml(xml_path)
                
                if not load_result['success']:
                    return load_result
            else:
                # Create new empty collection
                self.logger.info(f"Creating new Rekordbox collection (file doesn't exist): {xml_path}")
                create_result = self.rekordbox_service.create_empty_collection(xml_path)
                
                if create_result['success']:
                    load_result = {
                        'success': True,
                        'tracks_loaded': 0,
                        'playlists_loaded': 0,
                        'errors': [],
                        'warnings': ['Created new empty collection']
                    }
                else:
                    load_result = {
                        'success': False,
                        'tracks_loaded': 0,
                        'playlists_loaded': 0,
                        'errors': [f"Failed to create empty collection: {create_result.get('error', 'Unknown error')}"],
                        'warnings': []
                    }
            
            # Convert to enhanced tracks
            with self._lock:
                self.enhanced_tracks.clear()
                
                for track_id, track in self.rekordbox_service.tracks.items():
                    enhanced_track = self._convert_to_enhanced_track(track)
                    self.enhanced_tracks[track_id] = enhanced_track
                
                self.stats['tracks_loaded'] = len(self.enhanced_tracks)
            
            self.logger.info(f"Loaded {len(self.enhanced_tracks)} tracks for enhancement")
            
            # Add enhancement info to result
            load_result['enhanced_tracks_prepared'] = len(self.enhanced_tracks)
            load_result['enhancement_ready'] = True
            
            return load_result
            
        except Exception as e:
            self.logger.error(f"Failed to load Rekordbox collection: {str(e)}", exc_info=True)
            raise RekordboxError(f"Enhanced Rekordbox loading failed: {str(e)}")
    
    def integrate_processing_results(self, processing_results: List[ProcessingResult]) -> Dict[str, Any]:
        """
        Integrate DJ Music Cleaner processing results into Rekordbox collection
        
        Args:
            processing_results: List of processing results from DJ Music Cleaner
            
        Returns:
            Integration results and statistics
        """
        self.logger.info(f"Integrating {len(processing_results)} processing results into Rekordbox")
        
        integration_result = {
            'success': False,
            'tracks_updated': 0,
            'tracks_added': 0,
            'cue_points_added': 0,
            'beat_grids_added': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            with self._lock:
                for result in processing_results:
                    try:
                        if result.success and result.metadata:
                            # Track file mapping if renamed
                            if result.file_renamed and result.new_path:
                                self.file_mappings[result.filepath] = result.new_path
                            
                            # Store processing result
                            self.processing_results[result.filepath] = result
                            
                            # Find or create track in collection
                            track_updated = self._integrate_single_result(result, integration_result)
                            
                            if track_updated:
                                integration_result['tracks_updated'] += 1
                                self.stats['tracks_enhanced'] += 1
                                
                                self.logger.debug(f"Integrated metadata for: {os.path.basename(result.filepath)}")
                    
                    except Exception as e:
                        error_msg = f"Failed to integrate {os.path.basename(result.filepath)}: {str(e)}"
                        integration_result['errors'].append(error_msg)
                        self.logger.error(error_msg)
                
                integration_result['success'] = True
                self.logger.info(f"Integration complete: {integration_result['tracks_updated']} tracks updated")
            
            return integration_result
            
        except Exception as e:
            self.logger.error(f"Processing results integration failed: {str(e)}", exc_info=True)
            raise RekordboxError(f"Failed to integrate processing results: {str(e)}")
    
    def save_enhanced_collection(self, xml_path: Optional[str] = None, 
                                create_backup: Optional[bool] = None) -> Dict[str, Any]:
        """
        Save enhanced Rekordbox collection with all integrated metadata
        
        Args:
            xml_path: Output path for XML (uses original if None)
            create_backup: Whether to backup original (uses config default if None)
            
        Returns:
            Save results and statistics
        """
        xml_path = xml_path or self.rekordbox_service.xml_path
        create_backup = create_backup if create_backup is not None else self.backup_before_update
        
        if not xml_path:
            raise RekordboxError("No XML path specified")
        
        self.logger.info(f"Saving enhanced Rekordbox collection: {xml_path}")
        
        save_result = {
            'success': False,
            'xml_path': xml_path,
            'backup_created': False,
            'tracks_saved': 0,
            'playlists_saved': 0,
            'enhancements_applied': 0
        }
        
        try:
            with self._lock:
                # Create backup if requested
                if create_backup and os.path.exists(xml_path):
                    backup_path = f"{xml_path}.enhanced_backup_{int(time.time())}"
                    import shutil
                    shutil.copy2(xml_path, backup_path)
                    save_result['backup_created'] = True
                    self.stats['backups_created'] += 1
                    self.logger.info(f"Created backup: {backup_path}")
                
                # Update base service with enhanced data
                self._sync_enhanced_to_base_service()
                
                # Save using base service
                base_result = self.rekordbox_service.save_xml(xml_path, backup=False)  # We handled backup
                
                # Merge results
                save_result.update(base_result)
                save_result['enhancements_applied'] = self.stats['tracks_enhanced']
                
                self.logger.info(f"Enhanced collection saved successfully")
                self.app_logger.log_service_operation('rekordbox', 'save_enhanced_collection', 
                                                     0, True, {'tracks_enhanced': self.stats['tracks_enhanced']})
            
            return save_result
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced collection: {str(e)}", exc_info=True)
            raise RekordboxError(f"Enhanced collection save failed: {str(e)}")
    
    def validate_integration(self) -> Dict[str, Any]:
        """
        Validate the integration results to ensure data integrity
        
        Returns:
            Validation results and recommendations
        """
        self.logger.info("Validating Rekordbox integration")
        
        validation_result = {
            'validation_passed': False,
            'total_tracks': 0,
            'enhanced_tracks': 0,
            'missing_metadata': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            with self._lock:
                validation_result['total_tracks'] = len(self.enhanced_tracks)
                
                enhanced_count = 0
                missing_metadata = []
                
                for track_id, track in self.enhanced_tracks.items():
                    if track.processed_by_cleaner:
                        enhanced_count += 1
                    
                    # Check for missing essential metadata
                    missing_fields = []
                    if not track.title:
                        missing_fields.append('title')
                    if not track.artist:
                        missing_fields.append('artist')
                    if not track.bpm:
                        missing_fields.append('bpm')
                    
                    if missing_fields:
                        missing_metadata.append({
                            'track_id': track_id,
                            'location': os.path.basename(track.location),
                            'missing_fields': missing_fields
                        })
                
                validation_result['enhanced_tracks'] = enhanced_count
                validation_result['missing_metadata'] = missing_metadata[:10]  # Limit to first 10
                
                # Generate recommendations
                if enhanced_count == 0:
                    validation_result['recommendations'].append(
                        "No tracks were enhanced - check processing results integration"
                    )
                
                if len(missing_metadata) > 0:
                    validation_result['recommendations'].append(
                        f"{len(missing_metadata)} tracks have missing essential metadata"
                    )
                
                enhancement_rate = (enhanced_count / len(self.enhanced_tracks) * 100) if self.enhanced_tracks else 0
                if enhancement_rate < 50:
                    validation_result['recommendations'].append(
                        f"Low enhancement rate ({enhancement_rate:.1f}%) - consider reprocessing with different settings"
                    )
                
                validation_result['validation_passed'] = len(validation_result['recommendations']) == 0
                
                self.logger.info(f"Validation complete: {enhanced_count}/{len(self.enhanced_tracks)} tracks enhanced")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Integration validation failed: {str(e)}", exc_info=True)
            return {
                'validation_passed': False,
                'error': str(e)
            }
    
    def export_integration_report(self, output_path: str) -> str:
        """
        Export detailed integration report for review
        
        Args:
            output_path: Path for report file
            
        Returns:
            Path to generated report
        """
        self.logger.info(f"Generating integration report: {output_path}")
        
        try:
            report_data = {
                'integration_summary': {
                    'total_tracks': len(self.enhanced_tracks),
                    'enhanced_tracks': self.stats['tracks_enhanced'],
                    'tracks_updated': self.stats['tracks_updated'], 
                    'tracks_added': self.stats['tracks_added'],
                    'cue_points_synced': self.stats['cue_points_synced'],
                    'beat_grids_synced': self.stats['beat_grids_synced'],
                    'enhancement_rate': (self.stats['tracks_enhanced'] / len(self.enhanced_tracks) * 100) if self.enhanced_tracks else 0
                },
                'enhanced_tracks': [],
                'file_mappings': self.file_mappings,
                'statistics': self.stats,
                'validation': self.validate_integration()
            }
            
            # Add enhanced track details
            for track_id, track in self.enhanced_tracks.items():
                if track.processed_by_cleaner:
                    track_data = asdict(track)
                    # Remove sensitive paths for report
                    track_data['location'] = os.path.basename(track.location)
                    report_data['enhanced_tracks'].append(track_data)
            
            # Write report
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Integration report saved: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate integration report: {str(e)}", exc_info=True)
            raise RekordboxError(f"Report generation failed: {str(e)}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        with self._lock:
            base_stats = self.rekordbox_service.get_service_stats()
            
            return {
                **base_stats,
                'enhanced_integration': self.stats,
                'enhancement_rate': (self.stats['tracks_enhanced'] / len(self.enhanced_tracks) * 100) if self.enhanced_tracks else 0,
                'file_mappings_count': len(self.file_mappings),
                'processing_results_count': len(self.processing_results)
            }
    
    # Private helper methods
    
    def _convert_to_enhanced_track(self, base_track: RekordboxTrack) -> EnhancedRekordboxTrack:
        """Convert base Rekordbox track to enhanced version"""
        return EnhancedRekordboxTrack(
            track_id=base_track.track_id,
            location=base_track.location,
            title=base_track.title,
            artist=base_track.artist,
            album=base_track.album,
            genre=base_track.genre,
            year=base_track.year,
            bpm=base_track.bpm,
            musical_key=base_track.key,
            rating=base_track.rating,
            play_count=base_track.play_count,
            cue_points=base_track.cue_points.copy() if base_track.cue_points else [],
            beat_grid=base_track.beat_grid
        )
    
    def _integrate_single_result(self, result: ProcessingResult, integration_result: Dict[str, Any]) -> bool:
        """Integrate a single processing result into the collection"""
        try:
            # Find track by location (considering file mappings)
            location_to_find = result.new_path if result.new_path else result.filepath
            enhanced_track = self._find_track_by_location(location_to_find)
            
            if not enhanced_track:
                # Create new track
                enhanced_track = self._create_enhanced_track_from_result(result)
                track_id = str(max([int(tid) for tid in self.enhanced_tracks.keys() if tid.isdigit()] + [0]) + 1)
                enhanced_track.track_id = track_id
                self.enhanced_tracks[track_id] = enhanced_track
                integration_result['tracks_added'] += 1
                self.stats['tracks_added'] += 1
            
            # Update with processing results
            self._apply_processing_result_to_track(enhanced_track, result)
            
            # Sync advanced analysis
            if self.sync_advanced_analysis:
                self._sync_advanced_analysis(enhanced_track, result)
            
            # Mark as processed
            enhanced_track.processed_by_cleaner = True
            enhanced_track.processing_date = time.strftime("%Y-%m-%d %H:%M:%S")
            enhanced_track.online_enhanced = result.online_enhancement
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate single result: {str(e)}")
            return False
    
    def _find_track_by_location(self, location: str) -> Optional[EnhancedRekordboxTrack]:
        """Find enhanced track by file location"""
        normalized_location = os.path.normpath(location)
        
        for track in self.enhanced_tracks.values():
            if os.path.normpath(track.location) == normalized_location:
                return track
            
            # Check original location if file was moved
            if track.original_location and os.path.normpath(track.original_location) == normalized_location:
                return track
        
        return None
    
    def _create_enhanced_track_from_result(self, result: ProcessingResult) -> EnhancedRekordboxTrack:
        """Create new enhanced track from processing result"""
        metadata = result.metadata
        
        return EnhancedRekordboxTrack(
            track_id="",  # Will be set by caller
            location=result.new_path or result.filepath,
            original_location=result.filepath if result.new_path else None,
            title=metadata.title or "",
            artist=metadata.artist or "",
            album=metadata.album or "",
            genre=metadata.genre or "",
            year=metadata.year,
            bpm=metadata.bpm,
            musical_key=metadata.musical_key or "",
            energy_level=metadata.energy_level,
            danceability=getattr(metadata, 'danceability', None),
            valence=getattr(metadata, 'valence', None),
            bitrate=getattr(metadata, 'bitrate', 0),
            sample_rate=getattr(metadata, 'sample_rate', 0),
            is_high_quality=getattr(metadata, 'is_high_quality', False),
            audio_format=getattr(metadata, 'format', ""),
            cleanup_applied=True
        )
    
    def _apply_processing_result_to_track(self, track: EnhancedRekordboxTrack, result: ProcessingResult):
        """Apply processing result metadata to enhanced track"""
        metadata = result.metadata
        
        # Update basic metadata (preserve existing if cleaner didn't enhance)
        if metadata.title and (not track.title or self.config.get('overwrite_existing', False)):
            track.title = metadata.title
        if metadata.artist and (not track.artist or self.config.get('overwrite_existing', False)):
            track.artist = metadata.artist
        if metadata.album and (not track.album or self.config.get('overwrite_existing', False)):
            track.album = metadata.album
        if metadata.genre and (not track.genre or self.config.get('overwrite_existing', False)):
            track.genre = metadata.genre
        if metadata.year and (not track.year or self.config.get('overwrite_existing', False)):
            track.year = metadata.year
        
        # Always update analysis results
        if metadata.bpm:
            track.bpm = metadata.bpm
        if metadata.musical_key:
            track.musical_key = metadata.musical_key
        if metadata.energy_level is not None:
            track.energy_level = metadata.energy_level
        if hasattr(metadata, 'danceability') and metadata.danceability is not None:
            track.danceability = metadata.danceability
        if hasattr(metadata, 'valence') and metadata.valence is not None:
            track.valence = metadata.valence
        
        # Update technical info
        if hasattr(metadata, 'bitrate') and metadata.bitrate:
            track.bitrate = metadata.bitrate
        if hasattr(metadata, 'sample_rate') and metadata.sample_rate:
            track.sample_rate = metadata.sample_rate
        if hasattr(metadata, 'is_high_quality') and metadata.is_high_quality is not None:
            track.is_high_quality = metadata.is_high_quality
        if hasattr(metadata, 'format') and metadata.format:
            track.audio_format = metadata.format
    
    def _sync_advanced_analysis(self, track: EnhancedRekordboxTrack, result: ProcessingResult):
        """Sync advanced analysis data (cue points, beat grids, energy profiles)"""
        metadata = result.metadata
        
        # Sync cue points
        if hasattr(metadata, 'cue_points') and metadata.cue_points:
            track.cue_points = self._convert_cue_points_for_rekordbox(metadata.cue_points)
            self.stats['cue_points_synced'] += len(metadata.cue_points)
        
        # Sync beat grid
        if hasattr(metadata, 'beat_grid') and metadata.beat_grid:
            track.beat_grid = metadata.beat_grid
            self.stats['beat_grids_synced'] += 1
        
        # Sync energy profile
        if hasattr(metadata, 'energy_profile') and metadata.energy_profile:
            track.energy_profile = metadata.energy_profile
    
    def _convert_cue_points_for_rekordbox(self, cue_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert generic cue points to Rekordbox format"""
        rekordbox_cues = []
        
        for i, cue in enumerate(cue_points):
            rekordbox_cue = {
                'position': cue.get('position_seconds', 0.0),
                'type': '0',  # Regular cue point
                'name': cue.get('description', f'Cue {i+1}'),
                'color': 'FF0000',  # Red
                'num': str(i)
            }
            rekordbox_cues.append(rekordbox_cue)
        
        return rekordbox_cues
    
    def _sync_enhanced_to_base_service(self):
        """Sync enhanced tracks back to base Rekordbox service for saving"""
        self.rekordbox_service.tracks.clear()
        
        for track_id, enhanced_track in self.enhanced_tracks.items():
            # Convert back to base RekordboxTrack
            base_track = RekordboxTrack(
                track_id=enhanced_track.track_id,
                location=enhanced_track.location,
                title=enhanced_track.title,
                artist=enhanced_track.artist,
                album=enhanced_track.album,
                genre=enhanced_track.genre,
                year=enhanced_track.year,
                bpm=enhanced_track.bpm,
                key=enhanced_track.musical_key,
                rating=enhanced_track.rating,
                play_count=enhanced_track.play_count,
                cue_points=enhanced_track.cue_points,
                beat_grid=enhanced_track.beat_grid,
                date_modified=time.strftime("%Y-%m-%d")
            )
            
            self.rekordbox_service.tracks[track_id] = base_track


__all__ = ['EnhancedRekordboxIntegration', 'EnhancedRekordboxTrack']