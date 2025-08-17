"""
DJ Music Cleaner - Unified Application

This is the main application class that orchestrates all services in the
new service-oriented architecture. It consolidates functionality from:

1. Original monolithic DJMusicCleaner
2. Enhanced DJMusicCleanerEvolved  
3. Process-isolated audio analysis service

Provides a clean, maintainable, and extensible architecture while
preserving all existing functionality.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import asdict

# Core imports
from .core.models import ProcessingOptions, TrackMetadata, ProcessingResult, BatchProcessingResult, ServiceConfiguration
from .core.exceptions import DJMusicCleanerError, ProcessingError, ServiceError

# Service imports
from .services.unified_audio_analysis import UnifiedAudioAnalysisService
from .services.unified_cache import UnifiedCacheService  
from .services.unified_metadata import UnifiedMetadataService
from .services.analytics import AnalyticsService
from .services.file_operations import FileOperationsService
from .services.rekordbox import RekordboxService
from .services.enhanced_rekordbox_integration import EnhancedRekordboxIntegration
from .services.advanced.cue_detection import AdvancedCueDetectionService
from .services.advanced.beatgrid import BeatGridService
from .services.advanced.energy_calibration import EnergyCalibrationService
from .services.advanced.export_services import ExportService
from .core.engine import ProcessingEngine

# Utility imports
from .utils.text import generate_clean_filename
from .utils.filesystem import find_audio_files, ensure_directory, safe_move_file, get_file_info
from .utils.audio import is_audio_file, get_audio_format, validate_audio_file_header
from .utils.logging_config import setup_logging, get_logger, get_app_logger


class DJMusicCleanerUnified:
    """
    Unified DJ Music Cleaner Application
    
    Service-oriented architecture providing:
    - Comprehensive metadata extraction and enhancement
    - Stable audio analysis with multiprocessing safety
    - Professional caching and performance optimization  
    - Advanced file operations and validation
    - Professional analytics and reporting
    - Advanced cue detection and beat grid analysis
    - Energy calibration and dynamics analysis
    - Rekordbox XML integration
    - Multi-format export capabilities
    - Full backward compatibility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified DJ Music Cleaner
        
        Args:
            config: Configuration dictionary for services
        """
        self.config = config or {}
        
        # Setup comprehensive logging
        log_config = self.config.get('logging', {})
        self.app_logger = setup_logging(
            log_dir=log_config.get('log_dir'),
            console_level=log_config.get('console_level', 'INFO'),
            file_level=log_config.get('file_level', 'DEBUG'),
            enable_console=log_config.get('enable_console', True)
        )
        self.logger = self.app_logger.get_logger('main')
        
        # Service configuration
        self.service_config = ServiceConfiguration()
        if 'service_config' in self.config:
            # Update with user-provided config
            for key, value in self.config['service_config'].items():
                if hasattr(self.service_config, key):
                    setattr(self.service_config, key, value)
        
        # Application state
        self._initialized = False
        self._processing_lock = threading.Lock()
        
        # Performance tracking
        self.session_stats = {
            'files_processed': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'cache_hits': 0,
            'online_enhancements': 0,
            'total_processing_time': 0.0,
            'session_start_time': time.time()
        }
        
        # Initialize services
        self._init_services()
        
        # Log initialization
        self.logger.info("DJ Music Cleaner Unified - Initialization Started")
        self.logger.info("Core Services: Audio Analysis, Metadata, Cache, Analytics")
        self.logger.info("Advanced Services: Cue Detection, Beat Grid, Energy Calibration")
        self.logger.info("Integration: Rekordbox, Export Services, File Operations")
        self.logger.info("Architecture: Service-Oriented with Processing Engine")
        self.logger.info("Features: Professional Analytics, Multiprocessing-Safe")
        
        # Console output for user feedback
        print("🎵 DJ Music Cleaner Unified - Initialized Successfully")
        print(f"   Core Services: Audio Analysis, Metadata, Cache, Analytics")
        print(f"   Advanced Services: Cue Detection, Beat Grid, Energy Calibration")
        print(f"   Integration: Rekordbox, Export Services, File Operations")
        print(f"   Architecture: Service-Oriented with Processing Engine")
        print(f"   Features: Professional Analytics, Multiprocessing-Safe")
        print(f"   Logs: {self.app_logger.log_dir}")
    
    def _init_services(self):
        """Initialize all services with proper configuration"""
        try:
            # Core services
            audio_config = {
                'prefer_aubio': self.service_config.prefer_aubio,
                'fallback_to_librosa': self.service_config.fallback_to_librosa,
                'analysis_timeout': self.service_config.audio_analysis_timeout
            }
            self.audio_service = UnifiedAudioAnalysisService(audio_config)
            
            metadata_config = {
                'enable_online_lookup': self.service_config.online_lookup_enabled,
                'online_timeout': self.service_config.online_timeout,
                'cleanup_metadata': self.service_config.metadata_cleanup_enabled,
                'aggressive_cleanup': True  # Enable aggressive tag cleaning to remove pollution like '.com'
            }
            self.metadata_service = UnifiedMetadataService(metadata_config)
            
            cache_config = {
                'cache_dir': os.path.expanduser('~/.dj_music_cleaner_unified'),
                'db_name': self.service_config.cache_database,
                'default_timeout_days': self.service_config.cache_timeout_days,
                'enable_compression': True
            }
            self.cache_service = UnifiedCacheService(cache_config)
            
            # Professional services
            self.analytics_service = AnalyticsService(self.config.get('analytics', {}))
            self.file_operations_service = FileOperationsService(self.config.get('file_operations', {}))
            self.rekordbox_service = RekordboxService(self.config.get('rekordbox', {}))
            self.enhanced_rekordbox = EnhancedRekordboxIntegration(self.config.get('rekordbox', {}))
            self.export_service = ExportService(self.config.get('export', {}))
            
            # Advanced services (initialized on demand)
            self.cue_detection_service = None
            self.beatgrid_service = None
            self.energy_calibration_service = None
            
            # Processing engine
            self.processing_engine = ProcessingEngine(
                audio_service=self.audio_service,
                metadata_service=self.metadata_service,
                cache_service=self.cache_service
            )
            
            self._initialized = True
            self.logger.info("All services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {str(e)}", exc_info=True)
            raise DJMusicCleanerError(f"Service initialization failed: {str(e)}")
    
    def process_single_file(self, filepath: str, options: Optional[ProcessingOptions] = None) -> ProcessingResult:
        """
        Process a single audio file
        
        Args:
            filepath: Path to audio file
            options: Processing options
            
        Returns:
            ProcessingResult with processing details
        """
        if not self._initialized:
            raise DJMusicCleanerError("Services not initialized")
        
        start_time = time.time()
        options = options or ProcessingOptions()
        
        # Initialize result
        result = ProcessingResult()
        result.filepath = filepath
        result.timestamp = time.time()
        
        try:
            print(f"   🎵 Processing: {os.path.basename(filepath)}")
            
            # Validate file exists and is readable
            if not os.path.exists(filepath):
                raise ProcessingError(f"File not found: {filepath}")
            
            if not os.access(filepath, os.R_OK):
                raise ProcessingError(f"File not readable: {filepath}")
                
            result.add_operation("file_validation")
            
            # Check cache first
            if options.use_cache and not options.force_refresh:
                cached_metadata = self.cache_service.get_metadata(
                    filepath, 
                    max_age_days=options.cache_timeout_days
                )
                
                if cached_metadata:
                    result.metadata = cached_metadata
                    result.cache_hit = True
                    result.success = True  # Explicitly set success flag for cache hits
                    result.add_operation("cache_hit")
                    
                    with self._processing_lock:
                        self.session_stats['cache_hits'] += 1
                    
                    print(f"   ✅ Cache hit - skipping analysis")
            
            # Use processing engine for core processing
            if not result.cache_hit:
                self.app_logger.log_cache_operation('get', filepath, False)
                self.logger.debug(f"Cache miss - proceeding with analysis: {os.path.basename(filepath)}")
                
                engine_result = self.processing_engine.process_file(filepath, options)
                # Merge engine results
                result.success = engine_result.success
                result.metadata = engine_result.metadata
                result.operations_performed.extend(engine_result.operations_performed)
                result.errors.extend(engine_result.errors)
                result.warnings.extend(engine_result.warnings)
                result.online_enhancement = engine_result.online_enhancement
                
                # Copy file operation results
                result.file_renamed = engine_result.file_renamed
                result.old_filename = engine_result.old_filename
                result.new_filename = engine_result.new_filename
                result.old_path = engine_result.old_path
                result.new_path = engine_result.new_path
                result.file_moved = engine_result.file_moved
                
                if result.online_enhancement:
                    self.logger.debug(f"Online enhancement applied: {os.path.basename(filepath)}")
                
                # Determine current filepath for advanced services
                # If file was renamed/moved during processing, use the new path
                current_filepath = result.new_path if result.file_renamed else filepath
                
                # Advanced processing if enabled
                if options.enable_advanced_cues and result.metadata:
                    cue_start = time.time()
                    self.logger.debug(f"Starting advanced cue detection: {os.path.basename(current_filepath)}")
                    # Pass pre-computed audio analysis to avoid redundant computation
                    cue_result = self._get_cue_detection_service().detect_cue_points(
                        current_filepath, 
                        audio_result=result.audio_analysis_result
                    )
                    cue_duration = time.time() - cue_start
                    
                    if cue_result.success:
                        result.metadata.cue_points = [{
                            'position_seconds': cp.position_seconds,
                            'cue_type': cp.cue_type,
                            'description': cp.description,
                            'confidence': cp.confidence
                        } for cp in cue_result.cue_points]
                        result.add_operation("advanced_cue_detection")
                        self.app_logger.log_service_operation('cue_detection', 'detect_cue_points', 
                                                             cue_duration, True, 
                                                             {'cue_points': len(cue_result.cue_points)})
                    else:
                        self.app_logger.log_service_operation('cue_detection', 'detect_cue_points', 
                                                             cue_duration, False)
                
                if options.enable_advanced_beatgrid and result.metadata:
                    beatgrid_start = time.time()
                    self.logger.debug(f"Starting advanced beatgrid analysis: {os.path.basename(current_filepath)}")
                    # Pass pre-computed audio analysis to avoid redundant computation
                    beatgrid_result = self._get_beatgrid_service().generate_beat_grid(
                        current_filepath,
                        audio_result=result.audio_analysis_result
                    )
                    beatgrid_duration = time.time() - beatgrid_start
                    
                    if beatgrid_result.success:
                        result.metadata.beat_grid = {
                            'average_bpm': beatgrid_result.average_bpm,
                            'tempo_stability': beatgrid_result.tempo_stability,
                            'downbeats': beatgrid_result.downbeats,
                            'time_signature': beatgrid_result.time_signature
                        }
                        result.add_operation("advanced_beatgrid")
                        self.app_logger.log_service_operation('beatgrid', 'generate_beat_grid', 
                                                             beatgrid_duration, True, 
                                                             {'bpm': beatgrid_result.average_bpm})
                    else:
                        self.app_logger.log_service_operation('beatgrid', 'generate_beat_grid', 
                                                             beatgrid_duration, False)
                
                if options.enable_calibrated_energy and result.metadata:
                    energy_start = time.time()
                    self.logger.debug(f"Starting energy calibration analysis: {os.path.basename(current_filepath)}")
                    energy_profile = self._get_energy_calibration_service().analyze_track_energy(current_filepath)
                    energy_duration = time.time() - energy_start
                    
                    result.metadata.energy_profile = {
                        'overall_energy': energy_profile.overall_energy,
                        'energy_rating': energy_profile.energy_rating,
                        'dynamic_range': energy_profile.dynamic_range,
                        'recommended_gain': energy_profile.recommended_gain
                    }
                    result.add_operation("calibrated_energy_analysis")
                    self.app_logger.log_service_operation('energy_calibration', 'analyze_track_energy', 
                                                         energy_duration, True, 
                                                         {'energy_rating': energy_profile.energy_rating})
            
            # File operations - Generate new filename regardless of where it will be written
            if options.allow_rename and result.metadata and result.metadata.artist and result.metadata.title:
                new_filename = generate_clean_filename(
                    result.metadata.artist,
                    result.metadata.title, 
                    result.metadata.year if options.include_year_in_filename else None
                )
                
                # Track the new filename in the result object
                if new_filename != result.metadata.filename:
                    result.new_filename = new_filename
                    result.file_renamed = True
                    result.old_filename = result.metadata.filename
                    result.add_operation("file_rename")
                    self.app_logger.log_file_operation('rename_generated', result.metadata.filename, new_filename)
                    self.logger.debug(f"Generated new filename: {result.metadata.filename} -> {new_filename}")
                    print(f"   📝 Generated new name: {new_filename}")
            
            # Always write to output directory if specified, never modify input file
            if options.output_directory and not options.dry_run:
                try:
                    # Ensure output directory exists
                    ensure_directory(options.output_directory)
                    
                    # Use new filename if renamed, otherwise use original filename
                    output_filename = result.new_filename if result.file_renamed else os.path.basename(filepath)
                    output_path = os.path.join(options.output_directory, output_filename)
                    
                    # Copy file to output directory (never modify original)
                    from shutil import copy2
                    copy2(filepath, output_path)
                    
                    # Update result with new path
                    result.output_path = output_path
                    result.add_operation("output_file_saved")
                    print(f"   💾 Saved to output: {output_path}")
                except Exception as e:
                    result.add_warning(f"Output file save failed: {str(e)}")
                    
            # In dry run mode, just simulate the operation
            elif options.dry_run and options.output_directory:
                output_filename = result.new_filename if result.file_renamed else os.path.basename(filepath)
                output_path = os.path.join(options.output_directory, output_filename)
                result.add_operation("output_file_simulation")
                print(f"   💾 Would save to: {output_path}")

            # File organization
            if options.organize_files and options.output_directory:
                # Implementation for organizing files into folders
                # This would move files to organized directory structure
                result.add_operation("file_organization")
        except Exception as e:
            # Handle errors
            result.success = False
            result.add_error(str(e))
            
            with self._processing_lock:
                self.session_stats['failed_processes'] += 1
            
            print(f"   ❌ Failed: {str(e)}")
            
            if not isinstance(e, (ProcessingError, ServiceError)):
                print(f"   Unexpected error: {str(e)}")
        
        # Calculate processing time and update analytics (always executed)
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        self.analytics_service.record_processing_result(result)

        # Track detailed file metadata for advanced JSON reports
        if result.success and result.metadata and hasattr(self.analytics_service, 'track_file_details'):
            # Extract original and cleaned metadata
            original_metadata = {}
            cleaned_metadata = {}
            changes = []
            output_path = None
            bitrate = 0
            sample_rate = 0.0
            is_high_quality = False

            # Get original metadata if available
            if hasattr(result, 'original_metadata') and result.original_metadata:
                original_metadata = result.original_metadata
            else:
                # Create placeholder for missing original metadata
                original_metadata = result.metadata.to_dict() if hasattr(result.metadata, 'to_dict') else {}

            # Get cleaned metadata
            cleaned_metadata = result.metadata.to_dict() if hasattr(result.metadata, 'to_dict') else {}

            # Collect changes if available
            if hasattr(result, 'changes') and result.changes:
                changes = result.changes
            elif result.operations_performed:
                changes = [f"Operation performed: {op}" for op in result.operations_performed]

            # Get output path if file was saved
            if hasattr(result, 'output_path') and result.output_path:
                output_path = result.output_path
            elif options and options.output_directory:
                # Construct likely output path based on options
                basename = os.path.basename(filepath)
                output_path = os.path.join(options.output_directory, basename)

            # Get audio quality info if available
            if hasattr(result.metadata, 'bitrate') and result.metadata.bitrate:
                bitrate = result.metadata.bitrate
            if hasattr(result.metadata, 'sample_rate') and result.metadata.sample_rate:
                sample_rate = result.metadata.sample_rate
            if hasattr(result.metadata, 'is_high_quality') and result.metadata.is_high_quality:
                is_high_quality = result.metadata.is_high_quality

            # Track detailed file information for reports
            try:
                self.analytics_service.track_file_details(
                    file_path=filepath,
                    original_metadata=original_metadata,
                    cleaned_metadata=cleaned_metadata,
                    changes=changes,
                    is_high_quality=is_high_quality,
                    bitrate=bitrate,
                    sample_rate=sample_rate,
                    enhanced=result.online_enhancement,
                    output_path=output_path
                )
            except Exception as e:
                print(f"   ⚠️ Warning: Could not track detailed file information: {str(e)}")

        # Update session stats
        with self._processing_lock:
            self.session_stats['files_processed'] += 1
            if result.success:
                self.session_stats['successful_processes'] += 1
                print(f"   ✅ Completed in {result.processing_time:.2f}s")
        
        return result
    
    def process_folder(self, folder_path: str, options: Optional[ProcessingOptions] = None,
                      progress_callback: Optional[callable] = None) -> BatchProcessingResult:
        """
        Process all audio files in a folder
        
        Args:
            folder_path: Path to folder containing audio files
            options: Processing options
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchProcessingResult with batch processing details
        """
        if not self._initialized:
            raise DJMusicCleanerError("Services not initialized")
        
        options = options or ProcessingOptions()
        
        # Log batch processing start
        self.logger.info(f"Starting batch processing: {folder_path}")
        self.logger.debug(f"Batch options: {asdict(options)}")
        
        print(f"🎵 Processing folder: {folder_path}")
        print(f"   Workers: {options.workers}")
        print(f"   Dry run: {options.dry_run}")
        
        # Initialize batch result
        batch_result = BatchProcessingResult()
        batch_result.start_time = time.time()
        
        try:
            # Find audio files
            audio_files = find_audio_files(folder_path, recursive=True)
            
            if not audio_files:
                self.logger.warning(f"No audio files found in {folder_path}")
                print("   ⚠️ No audio files found")
                batch_result.finalize()
                return batch_result
            
            # Pre-filter files for quality (skip problematic files early)
            print(f"🔍 Pre-validating {len(audio_files)} files...")
            from .services.file_validator import create_quality_filter
            
            validator = create_quality_filter(min_size_mb=1.0, min_bitrate_kbps=320, check_bitrate=True)  # 320kbps+ only
            valid_files = validator.get_valid_files(audio_files, verbose=True)
            
            validation_stats = validator.get_validation_stats()
            skipped_count = validation_stats['total_checked'] - validation_stats['passed']
            
            if skipped_count > 0:
                print(f"   ⏭️ Skipped {skipped_count} files:")
                if validation_stats['failed_size'] > 0:
                    print(f"     - {validation_stats['failed_size']} too small (< 1MB)")
                if validation_stats['failed_bitrate'] > 0:
                    print(f"     - {validation_stats['failed_bitrate']} low quality (< 320kbps)")
                if validation_stats['failed_format'] > 0:
                    print(f"     - {validation_stats['failed_format']} corrupted/unsupported")
                if validation_stats['failed_extension'] > 0:
                    print(f"     - {validation_stats['failed_extension']} wrong file type")
                print(f"   ✅ {validation_stats['passed']} high-quality files (320kbps+) will be processed")
            
            audio_files = valid_files
            
            if not audio_files:
                self.logger.warning(f"No valid audio files found after pre-filtering in {folder_path}")
                print("   ⚠️ No valid audio files found after pre-filtering")
                batch_result.finalize()
                return batch_result
            
            # Log batch processing details
            self.app_logger.log_batch_start(folder_path, len(audio_files), options.workers, asdict(options))
            self.logger.info(f"Found {len(audio_files)} valid audio files to process")
            
            # Filter files if skip_existing enabled
            if options.skip_existing:
                filtered_files = []
                for filepath in audio_files:
                    cached = self.cache_service.get_metadata(filepath)
                    if not cached:
                        filtered_files.append(filepath)
                
                skipped = len(audio_files) - len(filtered_files)
                batch_result.skipped = skipped
                audio_files = filtered_files
                
                print(f"   Skipping {skipped} existing files")
                print(f"   ⏭️ Skipping {skipped} existing files")
            
            # Process files
            if options.workers > 1:
                # Parallel processing
                self._process_files_parallel(audio_files, options, batch_result, progress_callback)
            else:
                # Sequential processing  
                self._process_files_sequential(audio_files, options, batch_result, progress_callback)
            
            # Finalize results
            batch_result.finalize()
            
            # Log batch completion
            self.app_logger.log_batch_complete(
                folder_path, batch_result.total_files, batch_result.successful, 
                batch_result.failed, batch_result.total_time
            )
            
            # Print summary
            self._print_batch_summary(batch_result)
            
            # Auto-integrate with Rekordbox if configured
            if options.update_rekordbox and options.rekordbox_xml_path:
                try:
                    self.logger.info("Auto-integrating results with Rekordbox")
                    
                    # Collect successful processing results
                    processing_results = [result for result in batch_result.results if result.success]
                    
                    if processing_results:
                        # Load Rekordbox collection if not already loaded
                        if not hasattr(self.enhanced_rekordbox, 'enhanced_tracks') or not self.enhanced_rekordbox.enhanced_tracks:
                            self.load_rekordbox_xml(options.rekordbox_xml_path)
                        
                        # Save with integrated results
                        rekordbox_result = self.save_rekordbox_xml(options.rekordbox_xml_path, processing_results)
                        
                        print(f"\n🎛️ Rekordbox Integration Complete:")
                        print(f"   Enhanced tracks: {rekordbox_result.get('enhancements_applied', 0)}")
                        print(f"   Validation: {'✅ Passed' if rekordbox_result.get('validation', {}).get('validation_passed') else '⚠️ Issues detected'}")
                        
                        if rekordbox_result.get('backup_created'):
                            print(f"   Backup created: ✅")
                    else:
                        self.logger.warning("No successful processing results to integrate with Rekordbox")
                        
                except Exception as e:
                    self.logger.error(f"Rekordbox integration failed: {str(e)}")
                    print(f"\n⚠️ Rekordbox integration failed: {str(e)}")
            
            return batch_result
            
        except Exception as e:
            batch_result.finalize()
            self.logger.error(f"Batch processing failed for {folder_path}: {str(e)}", exc_info=True)
            raise ProcessingError(f"Batch processing failed: {str(e)}")
    
    def _process_files_sequential(self, files: List[str], options: ProcessingOptions,
                                 batch_result: BatchProcessingResult,
                                 progress_callback: Optional[callable] = None):
        """Process files sequentially"""
        for i, filepath in enumerate(files):
            try:
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(files), filepath)
                
                # Process file
                result = self.process_single_file(filepath, options)
                batch_result.add_result(result)
                
            except Exception as e:
                # Create error result
                error_result = ProcessingResult()
                error_result.filepath = filepath
                error_result.success = False
                error_result.add_error(str(e))
                batch_result.add_result(error_result)
    
    def _process_files_parallel(self, files: List[str], options: ProcessingOptions,
                               batch_result: BatchProcessingResult,
                               progress_callback: Optional[callable] = None):
        """Process files in parallel using ThreadPoolExecutor"""
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=options.workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, filepath, options): filepath
                for filepath in files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    batch_result.add_result(result)
                    
                except Exception as e:
                    # Create error result
                    error_result = ProcessingResult()
                    error_result.filepath = filepath
                    error_result.success = False
                    error_result.add_error(str(e))
                    batch_result.add_result(error_result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed_count, len(files), filepath)
    
    def _print_batch_summary(self, batch_result: BatchProcessingResult):
        """Print batch processing summary"""
        print(f"\n📊 Batch Processing Summary")
        print(f"   Total files: {batch_result.total_files}")
        print(f"   Successful: {batch_result.successful} ✅")
        print(f"   Failed: {batch_result.failed} ❌")
        print(f"   Skipped: {batch_result.skipped} ⏭️")
        print(f"   Processing time: {batch_result.total_time:.1f}s")
        print(f"   Average per file: {batch_result.avg_processing_time:.2f}s")
        
        if batch_result.files_renamed > 0:
            print(f"   Files renamed: {batch_result.files_renamed}")
        
        if batch_result.cache_hits > 0:
            print(f"   Cache hits: {batch_result.cache_hits}")
        
        if batch_result.online_enhancements > 0:
            print(f"   Online enhancements: {batch_result.online_enhancements}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        with self._processing_lock:
            session_duration = time.time() - self.session_stats['session_start_time']
            avg_time = (self.session_stats['total_processing_time'] / 
                       self.session_stats['files_processed'] 
                       if self.session_stats['files_processed'] > 0 else 0)
            
            return {
                'session': {
                    'duration_seconds': round(session_duration, 1),
                    'files_processed': self.session_stats['files_processed'],
                    'successful_processes': self.session_stats['successful_processes'],
                    'failed_processes': self.session_stats['failed_processes'],
                    'success_rate': round(
                        self.session_stats['successful_processes'] / 
                        self.session_stats['files_processed'] * 100
                        if self.session_stats['files_processed'] > 0 else 0, 1
                    ),
                    'cache_hits': self.session_stats['cache_hits'],
                    'online_enhancements': self.session_stats['online_enhancements'],
                    'average_processing_time': round(avg_time, 3)
                },
                'services': {
                    'audio_analysis': self.audio_service.get_performance_stats(),
                    'metadata': self.metadata_service.get_performance_stats(), 
                    'cache': self.cache_service.get_cache_stats(),
                    'analytics': self.analytics_service.get_stats(),
                    'file_operations': self.file_operations_service.get_service_stats(),
                    'rekordbox': self.rekordbox_service.get_service_stats(),
                    'export': self.export_service.get_stats()
                }
            }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize application performance"""
        print("🔧 Optimizing Performance...")
        
        results = {
            'cache_optimization': {},
            'service_optimization': {},
            'recommendations': []
        }
        
        try:
            # Optimize cache
            cache_results = self.cache_service.optimize_cache()
            results['cache_optimization'] = cache_results
            
            print(f"   🗑️ Cache cleanup: {cache_results['expired_removed']} expired, "
                  f"{cache_results['orphaned_removed']} orphaned")
            
            # Service recommendations
            stats = self.get_session_stats()
            
            # Cache hit rate recommendations
            cache_hit_rate = (stats['session']['cache_hits'] / 
                            stats['session']['files_processed'] * 100
                            if stats['session']['files_processed'] > 0 else 0)
            
            if cache_hit_rate < 20:
                results['recommendations'].append(
                    "Low cache hit rate - consider increasing cache timeout"
                )
            
            # Performance recommendations  
            avg_time = stats['session']['average_processing_time']
            if avg_time > 5.0:
                results['recommendations'].append(
                    "Slow processing - consider increasing worker count or disabling expensive analysis"
                )
            
            print("   ✅ Performance optimization completed")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"   ❌ Optimization failed: {e}")
        
        return results
    
    def _get_cue_detection_service(self) -> AdvancedCueDetectionService:
        """Get cue detection service (lazy initialization)"""
        if self.cue_detection_service is None:
            self.cue_detection_service = AdvancedCueDetectionService(
                self.config.get('cue_detection', {})
            )
        return self.cue_detection_service
    
    def _get_beatgrid_service(self) -> BeatGridService:
        """Get beat grid service (lazy initialization)"""
        if self.beatgrid_service is None:
            self.beatgrid_service = BeatGridService(
                self.config.get('beatgrid', {})
            )
        return self.beatgrid_service
    
    def _get_energy_calibration_service(self) -> EnergyCalibrationService:
        """Get energy calibration service (lazy initialization)"""
        if self.energy_calibration_service is None:
            self.energy_calibration_service = EnergyCalibrationService(
                self.config.get('energy_calibration', {})
            )
        return self.energy_calibration_service
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Get comprehensive analytics report"""
        return self.analytics_service.get_performance_report()
    
    def export_data(self, tracks: List[Dict[str, Any]], format_name: str, 
                   output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Export track data to various formats"""
        return self.export_service.export_collection(tracks, format_name, output_path, options)
    
    def load_rekordbox_xml(self, xml_path: str) -> Dict[str, Any]:
        """Load Rekordbox XML database with enhanced integration"""
        self.logger.info(f"Loading Rekordbox XML for enhanced integration: {xml_path}")
        return self.enhanced_rekordbox.load_rekordbox_collection(xml_path)
    
    def save_rekordbox_xml(self, xml_path: Optional[str] = None, processing_results: Optional[List[ProcessingResult]] = None) -> Dict[str, Any]:
        """Save Rekordbox XML database with enhanced metadata integration"""
        self.logger.info(f"Saving enhanced Rekordbox XML: {xml_path}")
        
        # If we have processing results, integrate them first
        if processing_results:
            integration_result = self.enhanced_rekordbox.integrate_processing_results(processing_results)
            self.logger.info(f"Integrated {integration_result['tracks_updated']} tracks with enhanced metadata")
        
        # Save the enhanced collection
        save_result = self.enhanced_rekordbox.save_enhanced_collection(xml_path)
        
        # Validate the integration
        validation_result = self.enhanced_rekordbox.validate_integration()
        save_result['validation'] = validation_result
        
        if not validation_result['validation_passed']:
            self.logger.warning("Rekordbox integration validation found issues")
            for warning in validation_result.get('warnings', []):
                self.logger.warning(f"  - {warning}")
            for recommendation in validation_result.get('recommendations', []):
                self.logger.info(f"  Recommendation: {recommendation}")
        
        return save_result
    
    def cleanup_resources(self):
        """Cleanup resources and shutdown services"""
        print("🧹 Cleaning up resources...")
        
        try:
            # Cache cleanup
            if hasattr(self, 'cache_service'):
                expired = self.cache_service.cleanup_expired()
                print(f"   🗑️ Removed {expired} expired cache entries")
            
            # Export service cleanup
            if hasattr(self, 'export_service'):
                self.export_service.cleanup_completed_jobs()
            
            print("   ✅ Cleanup completed")
            
        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup_resources()


# Backward compatibility classes
class DJMusicCleaner(DJMusicCleanerUnified):
    """Backward compatibility alias for original class"""
    pass


class DJMusicCleanerEvolved(DJMusicCleanerUnified):
    """Backward compatibility alias for evolved class"""
    pass


__all__ = [
    'DJMusicCleanerUnified',
    'DJMusicCleaner',
    'DJMusicCleanerEvolved'
]