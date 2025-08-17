"""
DJ Music Cleaner - Core Processing Engine

This module contains the core processing engine extracted from the monolithic
implementation. It provides the fundamental processing logic that can be used
by different orchestrators while maintaining clean separation of concerns.
"""

import os
import time
import json
import threading
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union, Any

from ..core.exceptions import ProcessingError
from ..utils.logging_config import get_logger, get_app_logger

from .models import ProcessingOptions, TrackMetadata, ProcessingResult
from .exceptions import ProcessingError, DJMusicCleanerError
from ..utils.text import generate_clean_filename
from ..utils.filesystem import get_file_info, safe_move_file


class ProcessingEngine:
    """
    Core processing engine for DJ Music Cleaner
    
    This engine provides the fundamental processing logic extracted from
    the monolithic implementation. It's designed to be used by different
    orchestrators (CLI, GUI, API) while maintaining consistent behavior.
    
    Features:
    - File validation and preprocessing
    - Metadata extraction coordination  
    - Audio analysis coordination
    - File operations (rename, organize)
    - Result compilation and validation
    - Error handling and recovery
    """
    
    def __init__(self, audio_service=None, metadata_service=None, cache_service=None):
        """
        Initialize the processing engine
        
        Args:
            audio_service: Audio analysis service instance
            metadata_service: Metadata extraction service instance  
            cache_service: Caching service instance
        """
        self.audio_service = audio_service
        self.metadata_service = metadata_service
        self.cache_service = cache_service
        self.logger = get_logger('engine')
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Log initialization
        self.logger.debug("Processing engine initialized")
        self.logger.debug(f"Audio service: {type(self.audio_service).__name__ if self.audio_service else 'None'}")
        self.logger.debug(f"Metadata service: {type(self.metadata_service).__name__ if self.metadata_service else 'None'}")
        self.logger.debug(f"Cache service: {type(self.cache_service).__name__ if self.cache_service else 'None'}")
        
        # Processing state
        self._processing_count = 0
        self._error_count = 0
        
        # Performance tracking
        self.engine_stats = {
            'files_processed': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'metadata_extractions': 0,
            'audio_analyses': 0,
            'file_operations': 0
        }
    
    def process_file(self, filepath: str, options: ProcessingOptions) -> ProcessingResult:
        """
        Process a single audio file through the complete pipeline
        
        Args:
            filepath: Path to the audio file to process
            options: Processing options and configuration
            
        Returns:
            ProcessingResult with complete processing information
            
        Raises:
            ProcessingError: If processing fails critically
        """
        start_time = time.time()
        
        # Initialize result
        result = ProcessingResult()
        result.filepath = filepath
        result.timestamp = time.time()
        
        with self._lock:
            self._processing_count += 1
            self.engine_stats['files_processed'] += 1
        
        try:
            # Phase 1: File Validation and Preprocessing
            self._validate_file(filepath, result, options)
            
            # Phase 2: Cache Check
            cached_metadata = self._check_cache(filepath, result, options)
            
            # Phase 3: Metadata Extraction (if not cached)
            if not result.cache_hit:
                metadata = self._extract_metadata(filepath, result, options)
                result.metadata = metadata
            else:
                result.metadata = cached_metadata
            
            # Phase 4: Audio Analysis (if not cached and enabled)
            if not result.cache_hit and self._should_analyze_audio(options):
                self._perform_audio_analysis(filepath, result, options)
            
            # Phase 5: Cache Storage (if not from cache)
            if not result.cache_hit and options.use_cache:
                self._store_in_cache(filepath, result, options)
            
            # Phase 6: File Operations
            if self._should_perform_file_operations(options):
                self._perform_file_operations(filepath, result, options)
            
            # Phase 7: Result Validation and Finalization
            self._finalize_result(result, start_time)
            
            with self._lock:
                self.engine_stats['successful_processes'] += 1
            
            return result
            
        except Exception as e:
            # Handle processing errors
            result.success = False
            result.processing_time = time.time() - start_time
            result.add_error(f"Processing failed: {str(e)}")
            
            with self._lock:
                self._error_count += 1
                self.engine_stats['failed_processes'] += 1
            
            if isinstance(e, ProcessingError):
                return result
            else:
                raise ProcessingError(f"Unexpected error processing {filepath}: {str(e)}")
    
    def _validate_file(self, filepath: str, result: ProcessingResult, options: ProcessingOptions):
        """Phase 1: Validate file exists and is processable"""
        try:
            if not os.path.exists(filepath):
                raise ProcessingError(f"File not found: {filepath}")
            
            if not os.access(filepath, os.R_OK):
                raise ProcessingError(f"File not readable: {filepath}")
            
            # Get basic file info
            file_info = get_file_info(filepath)
            if not file_info['is_file']:
                raise ProcessingError(f"Path is not a file: {filepath}")
            
            # Check file size (avoid processing empty files)
            if file_info['size_bytes'] < 1024:  # Less than 1KB
                raise ProcessingError(f"File too small to be valid audio: {filepath}")
            
            result.add_operation("file_validation")
            
        except Exception as e:
            raise ProcessingError(f"File validation failed: {str(e)}")
    
    def _check_cache(self, filepath: str, result: ProcessingResult, options: ProcessingOptions) -> Optional[TrackMetadata]:
        """Phase 2: Check if file is already cached"""
        if not options.use_cache or options.force_refresh or not self.cache_service:
            return None
        
        try:
            cached_metadata = self.cache_service.get_metadata(
                filepath,
                max_age_days=options.cache_timeout_days
            )
            
            if cached_metadata:
                result.cache_hit = True
                result.add_operation("cache_hit")
                
                with self._lock:
                    self.engine_stats['cache_hits'] += 1
                
                return cached_metadata
            
            return None
            
        except Exception as e:
            # Cache errors shouldn't fail the whole process
            result.add_warning(f"Cache check failed: {str(e)}")
            return None
    
    def _extract_metadata(self, filepath: str, result: ProcessingResult, options: ProcessingOptions) -> TrackMetadata:
        """Phase 3: Extract metadata from file"""
        if not self.metadata_service:
            raise ProcessingError("Metadata service not available")
        
        try:
            metadata = self.metadata_service.extract_metadata(
                filepath,
                enhance_online=options.enhance_online,
                acoustid_api_key=options.acoustid_api_key
            )
            
            # Store original metadata for comparison
            result.original_metadata = TrackMetadata.from_dict(metadata.to_dict())
            result.add_operation("metadata_extraction")
            
            if metadata.online_enhanced:
                result.online_enhancement = True
            
            with self._lock:
                self.engine_stats['metadata_extractions'] += 1
            
            return metadata
            
        except Exception as e:
            raise ProcessingError(f"Metadata extraction failed: {str(e)}")
    
    def _should_analyze_audio(self, options: ProcessingOptions) -> bool:
        """Determine if audio analysis should be performed"""
        return (options.analyze_bpm or 
                options.analyze_key or 
                options.analyze_energy or 
                options.analyze_quality or
                options.enable_advanced_cues)
    
    def _perform_audio_analysis(self, filepath: str, result: ProcessingResult, options: ProcessingOptions):
        """Phase 4: Perform audio analysis"""
        if not self.audio_service:
            result.add_warning("Audio service not available - skipping analysis")
            return
        
        try:
            # Prepare analysis options
            analysis_options = {
                'bpm': options.analyze_bpm,
                'key': options.analyze_key,
                'energy': options.analyze_energy,
                'quality': options.analyze_quality,
                'cue_points': options.enable_advanced_cues
            }
            
            # Perform analysis
            audio_metadata = self.audio_service.analyze_track(filepath, analysis_options)
            
            # Check if analysis was successful
            if audio_metadata.analysis_status == "success":
                # Store full analysis result for advanced services
                result.audio_analysis_result = audio_metadata
                
                # Merge audio analysis results into existing metadata
                if result.metadata:
                    self._merge_audio_metadata(result.metadata, audio_metadata)
                else:
                    result.metadata = audio_metadata
                
                result.add_operation("audio_analysis")
            else:
                # Audio analysis failed - add as warning but continue processing
                error_details = "; ".join(audio_metadata.analysis_errors) if audio_metadata.analysis_errors else "Unknown error"
                result.add_warning(f"Audio analysis failed ({audio_metadata.analysis_status}): {error_details}")
                
                # Still merge what we can (like filepath, filename)
                if result.metadata:
                    result.metadata.filepath = audio_metadata.filepath
                    result.metadata.filename = audio_metadata.filename
                else:
                    result.metadata = audio_metadata
            
            with self._lock:
                self.engine_stats['audio_analyses'] += 1
            
        except Exception as e:
            # Audio analysis errors are warnings, not failures
            result.add_warning(f"Audio analysis failed: {str(e)}")
    
    def _merge_audio_metadata(self, base_metadata: TrackMetadata, audio_metadata: TrackMetadata):
        """Merge audio analysis results into base metadata"""
        # Update with audio analysis results, preserving existing values where appropriate
        if audio_metadata.bpm and (not base_metadata.bpm or audio_metadata.bpm_confidence > 0.5):
            base_metadata.bpm = audio_metadata.bpm
            base_metadata.bpm_confidence = audio_metadata.bpm_confidence
        
        if audio_metadata.musical_key and not base_metadata.musical_key:
            base_metadata.musical_key = audio_metadata.musical_key
            base_metadata.key_confidence = audio_metadata.key_confidence
            base_metadata.camelot_key = audio_metadata.camelot_key
        
        if audio_metadata.energy_level is not None:
            base_metadata.energy_level = audio_metadata.energy_level
        
        if audio_metadata.quality_score > 0:
            base_metadata.quality_score = audio_metadata.quality_score
            base_metadata.dynamic_range = audio_metadata.dynamic_range
            base_metadata.peak_level = audio_metadata.peak_level
            base_metadata.rms_level = audio_metadata.rms_level
        
        if audio_metadata.cue_points:
            base_metadata.cue_points = audio_metadata.cue_points
    
    def _store_in_cache(self, filepath: str, result: ProcessingResult, options: ProcessingOptions):
        """Phase 5: Store results in cache"""
        if not self.cache_service or not result.metadata:
            return
        
        try:
            success = self.cache_service.set_metadata(
                filepath,
                result.metadata,
                timeout_days=options.cache_timeout_days
            )
            
            if success:
                result.add_operation("cache_store")
            else:
                result.add_warning("Cache storage failed")
                
        except Exception as e:
            result.add_warning(f"Cache storage error: {str(e)}")
    
    def _should_perform_file_operations(self, options: ProcessingOptions) -> bool:
        """Determine if file operations should be performed"""
        return (options.allow_rename or 
                options.organize_files or
                options.output_directory is not None)
    
    def _perform_file_operations(self, filepath: str, result: ProcessingResult, options: ProcessingOptions):
        """Phase 6: Perform file operations (rename, organize)"""
        try:
            # File renaming
            if options.allow_rename and result.metadata and result.metadata.artist and result.metadata.title:
                self._handle_file_rename(filepath, result, options)
            
            # File organization  
            if options.organize_files and options.output_directory:
                self._handle_file_organization(filepath, result, options)
            
            if result.file_renamed or result.file_moved:
                with self._lock:
                    self.engine_stats['file_operations'] += 1
            
        except Exception as e:
            result.add_error(f"File operations failed: {str(e)}")
    
    def _handle_file_rename(self, filepath: str, result: ProcessingResult, options: ProcessingOptions):
        """Handle file renaming operation"""
        try:
            new_filename = generate_clean_filename(
                result.metadata.artist,
                result.metadata.title,
                result.metadata.year if options.include_year_in_filename else None
            )
            
            current_filename = os.path.basename(filepath)
            
            if new_filename != current_filename:
                if not options.dry_run:
                    new_path = os.path.join(os.path.dirname(filepath), new_filename)
                    
                    success, final_path = safe_move_file(filepath, new_path, backup=True)
                    
                    if success:
                        result.file_renamed = True
                        result.old_filename = current_filename
                        result.new_filename = os.path.basename(final_path)
                        result.old_path = filepath
                        result.new_path = final_path
                        result.add_operation("file_rename")
                else:
                    # Dry run simulation
                    result.add_operation("file_rename_simulation")
                    result.old_filename = current_filename
                    result.new_filename = new_filename
                    
        except Exception as e:
            result.add_warning(f"File rename failed: {str(e)}")
    
    def _handle_file_organization(self, filepath: str, result: ProcessingResult, options: ProcessingOptions):
        """Handle file organization into folder structure"""
        try:
            if not result.metadata:
                return
            
            # Create organized path structure
            # Example: /output/Artist/Album/Track.mp3
            artist_folder = result.metadata.artist or "Unknown Artist"
            album_folder = result.metadata.album or "Unknown Album"
            
            # Sanitize folder names
            from ..utils.text import sanitize_tag_value
            artist_folder = sanitize_tag_value(artist_folder)
            album_folder = sanitize_tag_value(album_folder)
            
            organized_path = os.path.join(
                options.output_directory,
                artist_folder,
                album_folder,
                result.new_filename if result.file_renamed else os.path.basename(filepath)
            )
            
            if not options.dry_run:
                # Use copy2 instead of move to preserve input file
                from shutil import copy2
                
                # Ensure target directory exists
                os.makedirs(os.path.dirname(organized_path), exist_ok=True)
                
                # Copy the file instead of moving it (preserves input file)
                source_path = result.new_path if result.file_renamed else filepath
                copy2(source_path, organized_path)
                success = True
                final_path = organized_path
                
                if success:
                    result.file_moved = True
                    result.new_path = final_path
                    result.add_operation("file_organize")
            else:
                # Dry run simulation
                result.add_operation("file_organize_simulation")
                result.file_moved = True
                result.new_path = organized_path
                
        except Exception as e:
            result.add_warning(f"File organization failed: {str(e)}")
    
    def _finalize_result(self, result: ProcessingResult, start_time: float):
        """Phase 7: Finalize processing result"""
        result.processing_time = time.time() - start_time
        result.success = True
        
        # Update engine statistics
        with self._lock:
            self.engine_stats['total_processing_time'] += result.processing_time
            if self.engine_stats['files_processed'] > 0:
                self.engine_stats['average_processing_time'] = (
                    self.engine_stats['total_processing_time'] / 
                    self.engine_stats['files_processed']
                )
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get processing engine statistics"""
        with self._lock:
            stats = self.engine_stats.copy()
            stats['current_processing_count'] = self._processing_count
            stats['current_error_count'] = self._error_count
            
            # Calculate rates
            if stats['files_processed'] > 0:
                stats['success_rate'] = (stats['successful_processes'] / 
                                       stats['files_processed'] * 100)
                stats['cache_hit_rate'] = (stats['cache_hits'] / 
                                         stats['files_processed'] * 100)
            else:
                stats['success_rate'] = 0.0
                stats['cache_hit_rate'] = 0.0
            
            return stats
    
    def reset_stats(self):
        """Reset processing statistics"""
        with self._lock:
            self.engine_stats = {
                'files_processed': 0,
                'successful_processes': 0,
                'failed_processes': 0,
                'total_processing_time': 0.0,
                'average_processing_time': 0.0,
                'cache_hits': 0,
                'metadata_extractions': 0,
                'audio_analyses': 0,
                'file_operations': 0
            }
            self._processing_count = 0
            self._error_count = 0
    
    def validate_services(self) -> Dict[str, bool]:
        """Validate that required services are available"""
        return {
            'audio_service': self.audio_service is not None,
            'metadata_service': self.metadata_service is not None,
            'cache_service': self.cache_service is not None
        }


__all__ = ['ProcessingEngine']