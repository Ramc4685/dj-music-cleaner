"""
Data models for DJ Music Cleaner

This module defines all data structures used throughout the application
for configuration, metadata, and processing results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import time


@dataclass
class ProcessingOptions:
    """Configuration options for processing operations"""
    
    # Core processing options
    enhance_online: bool = True
    acoustid_api_key: Optional[str] = None
    skip_existing: bool = False
    allow_rename: bool = True
    include_year_in_filename: bool = False
    workers: int = 1
    dry_run: bool = False
    
    # Audio analysis options
    analyze_bpm: bool = True
    analyze_key: bool = True
    analyze_energy: bool = True
    analyze_quality: bool = True
    
    # Advanced features
    enable_advanced_cues: bool = False
    enable_advanced_beatgrid: bool = False
    enable_calibrated_energy: bool = False
    enable_professional_reporting: bool = True
    
    # Caching options
    use_cache: bool = True
    cache_timeout_days: int = 30
    force_refresh: bool = False
    
    # Rekordbox integration
    rekordbox_xml_path: Optional[str] = None
    update_rekordbox: bool = False
    rekordbox_backup: bool = True
    preserve_rekordbox_data: bool = True
    rekordbox_report_path: Optional[str] = None
    
    # Reporting options
    generate_report: bool = True
    report_format: str = "html"  # html, json, csv
    report_path: Optional[str] = None
    
    # File organization
    organize_files: bool = False
    output_directory: Optional[str] = None
    
    # Quality and filtering
    min_quality_score: float = 0.0
    skip_duplicates: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingOptions':
        """Create from dictionary"""
        # Filter only valid field names
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class TrackMetadata:
    """Complete metadata for a music track"""
    
    # File information
    filepath: str = ""
    filename: str = ""
    filesize: int = 0
    format: str = ""
    bitrate: int = 0
    duration: float = 0.0
    
    # Basic metadata
    title: str = ""
    artist: str = ""
    album: str = ""
    year: Optional[int] = None
    genre: str = ""
    track_number: Optional[int] = None
    
    # Extended metadata
    album_artist: str = ""
    composer: str = ""
    publisher: str = ""
    isrc: str = ""
    catalog_number: str = ""
    label: str = ""
    
    # Audio analysis results
    bpm: Optional[float] = None
    bpm_confidence: float = 0.0
    musical_key: str = ""
    key_confidence: float = 0.0
    camelot_key: str = ""
    energy_level: Optional[float] = None
    
    # Quality metrics
    quality_score: float = 0.0
    dynamic_range: Optional[float] = None
    peak_level: Optional[float] = None
    rms_level: Optional[float] = None
    
    # DJ-specific features
    cue_points: List[Dict[str, Any]] = field(default_factory=list)
    beat_grid: Optional[Dict[str, Any]] = None
    loop_points: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    cache_hit: bool = False
    online_enhanced: bool = False
    validation_errors: List[str] = field(default_factory=list)
    analysis_errors: List[str] = field(default_factory=list)
    analysis_status: str = "pending"  # pending, success, failed_*, error
    
    # Source information
    musicbrainz_id: str = ""
    acoustid_id: str = ""
    last_modified: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod  
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackMetadata':
        """Create from dictionary"""
        # Handle missing fields gracefully
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def is_complete(self) -> bool:
        """Check if metadata has essential information"""
        return bool(self.title and self.artist and self.bpm)
    
    def get_clean_filename(self, include_year: bool = False) -> str:
        """Generate clean filename from metadata"""
        from ..utils.text import generate_clean_filename
        return generate_clean_filename(self.artist, self.title, self.year if include_year else None)


@dataclass
class ProcessingResult:
    """Result of processing a single track"""
    
    # Processing info
    filepath: str = ""
    success: bool = False
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    # Results
    metadata: Optional[TrackMetadata] = None
    original_metadata: Optional[TrackMetadata] = None
    
    # Processing details
    operations_performed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # File operations
    file_renamed: bool = False
    old_filename: str = ""
    new_filename: str = ""
    file_moved: bool = False
    old_path: str = ""
    new_path: str = ""
    
    # Analysis results
    cache_hit: bool = False
    online_enhancement: bool = False
    validation_performed: bool = False
    repair_performed: bool = False
    
    # Shared audio analysis for advanced services
    audio_analysis_result: Optional[TrackMetadata] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            if hasattr(value, 'to_dict'):
                result[field.name] = value.to_dict()
            else:
                result[field.name] = value
        return result
    
    def add_operation(self, operation: str):
        """Add an operation to the list"""
        if operation not in self.operations_performed:
            self.operations_performed.append(operation)
    
    def add_warning(self, warning: str):
        """Add a warning message"""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def add_error(self, error: str):
        """Add an error message"""
        if error not in self.errors:
            self.errors.append(error)


@dataclass
class BatchProcessingResult:
    """Result of batch processing operation"""
    
    # Overall stats
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_time: float = 0.0
    
    # Results
    results: List[ProcessingResult] = field(default_factory=list)
    
    # Summary stats
    files_renamed: int = 0
    files_moved: int = 0
    cache_hits: int = 0
    online_enhancements: int = 0
    
    # Analytics
    avg_processing_time: float = 0.0
    bpm_distribution: Dict[str, int] = field(default_factory=dict)
    key_distribution: Dict[str, int] = field(default_factory=dict)
    genre_distribution: Dict[str, int] = field(default_factory=dict)
    
    def add_result(self, result: ProcessingResult):
        """Add a processing result to the batch"""
        self.results.append(result)
        
        if result.success:
            self.successful += 1
        else:
            self.failed += 1
            
        if result.file_renamed:
            self.files_renamed += 1
            
        if result.file_moved:
            self.files_moved += 1
            
        if result.cache_hit:
            self.cache_hits += 1
            
        if result.online_enhancement:
            self.online_enhancements += 1
    
    def finalize(self):
        """Finalize batch processing and calculate summary stats"""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.total_files = len(self.results)
        
        # Calculate averages
        if self.successful > 0:
            total_processing_time = sum(r.processing_time for r in self.results if r.success)
            self.avg_processing_time = total_processing_time / self.successful
        
        # Calculate distributions
        for result in self.results:
            if result.success and result.metadata:
                # BPM distribution
                if result.metadata.bpm:
                    bpm_range = f"{int(result.metadata.bpm // 10) * 10}-{int(result.metadata.bpm // 10) * 10 + 9}"
                    self.bpm_distribution[bpm_range] = self.bpm_distribution.get(bpm_range, 0) + 1
                
                # Key distribution
                if result.metadata.musical_key:
                    self.key_distribution[result.metadata.musical_key] = \
                        self.key_distribution.get(result.metadata.musical_key, 0) + 1
                
                # Genre distribution
                if result.metadata.genre:
                    self.genre_distribution[result.metadata.genre] = \
                        self.genre_distribution.get(result.metadata.genre, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_files': self.total_files,
            'successful': self.successful,
            'failed': self.failed,
            'skipped': self.skipped,
            'total_time': self.total_time,
            'avg_processing_time': self.avg_processing_time,
            'files_renamed': self.files_renamed,
            'files_moved': self.files_moved,
            'cache_hits': self.cache_hits,
            'online_enhancements': self.online_enhancements,
            'bpm_distribution': self.bpm_distribution,
            'key_distribution': self.key_distribution,
            'genre_distribution': self.genre_distribution,
            'results': [r.to_dict() for r in self.results]
        }


@dataclass
class ServiceConfiguration:
    """Configuration for individual services"""
    
    # Cache service
    cache_enabled: bool = True
    cache_database: str = "dj_music_cleaner.db"
    cache_timeout_days: int = 30
    
    # Audio analysis service
    audio_analysis_timeout: int = 30
    prefer_aubio: bool = True
    fallback_to_librosa: bool = False
    
    # Metadata service
    online_lookup_enabled: bool = True
    online_timeout: int = 10
    metadata_cleanup_enabled: bool = True
    
    # File operations service
    backup_enabled: bool = True
    validation_enabled: bool = True
    auto_repair_enabled: bool = False
    
    # Analytics service
    analytics_enabled: bool = True
    detailed_analytics: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }