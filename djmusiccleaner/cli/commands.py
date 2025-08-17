"""
CLI Commands Module

Provides command implementations for the DJ Music Cleaner CLI.
Separates command logic from argument parsing for better maintainability.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core.models import ProcessingOptions
from ..core.exceptions import DJMusicCleanerError
from ..dj_music_cleaner_unified import DJMusicCleanerUnified
from ..services.batch_processor import BatchProcessor
from ..services.advanced.export_services import ExportService
from .config import CLIConfig


class CLICommands:
    """
    CLI command implementations
    
    Handles the execution logic for different CLI commands while keeping
    the argument parsing separate for better maintainability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CLI commands"""
        self.config = config or {}
        self.app: Optional[DJMusicCleanerUnified] = None
        self.verbose = False
    
    def process_command(self, args) -> int:
        """
        Main processing command
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            self.verbose = getattr(args, 'verbose', False)
            
            # Initialize application
            self._print_header()
            self.app = DJMusicCleanerUnified(self.config)
            
            # Validate path
            if not os.path.exists(args.path):
                self._print_error(f"Path not found: {args.path}")
                return 1
            
            # Create processing options
            options = self._create_processing_options(args)
            
            # Print configuration
            self._print_config(args, options)
            
            # Execute processing
            start_time = time.time()
            
            if os.path.isfile(args.path):
                exit_code = self._process_single_file(args.path, options)
            else:
                exit_code = self._process_folder(args.path, options, args)
            
            # Print final statistics
            total_time = time.time() - start_time
            self._print_completion_stats(total_time)
            
            return exit_code
            
        except DJMusicCleanerError as e:
            self._print_error(f"Application Error: {e}")
            return 1
        except KeyboardInterrupt:
            self._print_warning("Processing interrupted by user")
            return 130
        except Exception as e:
            self._print_error(f"Unexpected Error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
        finally:
            if self.app:
                self.app.cleanup_resources()
    
    def stats_command(self, args) -> int:
        """Show performance statistics"""
        try:
            self.app = DJMusicCleanerUnified(self.config)
            stats = self.app.get_session_stats()
            
            print("📊 Performance Statistics:")
            print(json.dumps(stats, indent=2))
            return 0
            
        except Exception as e:
            self._print_error(f"Failed to get statistics: {e}")
            return 1
    
    def optimize_command(self, args) -> int:
        """Optimize performance settings"""
        try:
            self.app = DJMusicCleanerUnified(self.config)
            results = self.app.optimize_performance()
            
            print("🔧 Optimization Results:")
            print(json.dumps(results, indent=2))
            return 0
            
        except Exception as e:
            self._print_error(f"Optimization failed: {e}")
            return 1
    
    def batch_command(self, args) -> int:
        """Execute batch processing command"""
        try:
            self.verbose = getattr(args, 'verbose', False)
            
            # Load batch session or create new one
            batch_processor = BatchProcessor(
                max_workers=getattr(args, 'workers', 4),
                callback=self._batch_progress_callback if self.verbose else None
            )
            
            if getattr(args, 'list_sessions', False):
                return self._list_batch_sessions(batch_processor)
            
            if getattr(args, 'resume_session', None):
                return self._resume_batch_session(batch_processor, args.resume_session, args)
            
            # Create new batch session
            return self._create_batch_session(batch_processor, args)
            
        except Exception as e:
            self._print_error(f"Batch processing failed: {e}")
            return 1
    
    def export_command(self, args) -> int:
        """Execute export command"""
        try:
            export_service = ExportService(self.config)
            
            if getattr(args, 'list_formats', False):
                return self._list_export_formats(export_service)
            
            # Get tracks to export
            if not hasattr(args, 'source') or not args.source:
                self._print_error("No source specified for export")
                return 1
            
            tracks = self._load_tracks_for_export(args.source, args)
            if not tracks:
                self._print_error("No tracks found for export")
                return 1
            
            # Start export
            format_name = getattr(args, 'format', 'json')
            output_path = getattr(args, 'output', f'export.{format_name}')
            
            job_id = export_service.export_collection(tracks, format_name, output_path)
            
            print(f"🚀 Export started: {job_id}")
            print(f"   Format: {format_name}")
            print(f"   Output: {output_path}")
            print(f"   Tracks: {len(tracks)}")
            
            # Monitor progress if verbose
            if self.verbose:
                self._monitor_export_progress(export_service, job_id)
            
            return 0
            
        except Exception as e:
            self._print_error(f"Export failed: {e}")
            return 1
    
    def config_command(self, args) -> int:
        """Handle configuration commands"""
        try:
            cli_config = CLIConfig()
            
            if getattr(args, 'show_config', False):
                config = cli_config.load_config()
                print("⚙️ Current Configuration:")
                print(json.dumps(config, indent=2))
                return 0
            
            if getattr(args, 'reset_config', False):
                cli_config.reset_to_defaults()
                print("✅ Configuration reset to defaults")
                return 0
            
            if getattr(args, 'set_option', None):
                key, value = args.set_option.split('=', 1)
                cli_config.set_option(key, value)
                print(f"✅ Set {key} = {value}")
                return 0
            
            return 0
            
        except Exception as e:
            self._print_error(f"Configuration command failed: {e}")
            return 1
    
    # Private helper methods
    
    def _create_processing_options(self, args) -> ProcessingOptions:
        """Create ProcessingOptions from command line arguments"""
        return ProcessingOptions(
            # Core processing
            enhance_online=getattr(args, 'enhance_online', True),
            skip_existing=getattr(args, 'skip_existing', False),
            allow_rename=getattr(args, 'allow_rename', False),
            include_year_in_filename=getattr(args, 'include_year_in_filename', False),
            workers=getattr(args, 'workers', 1),
            dry_run=getattr(args, 'dry_run', False),
            
            # Audio analysis
            analyze_bmp=getattr(args, 'analyze_bpm', True),
            analyze_key=getattr(args, 'analyze_key', True),
            analyze_energy=getattr(args, 'analyze_energy', True),
            analyze_quality=getattr(args, 'analyze_quality', False),
            
            # Advanced features
            enable_advanced_cues=getattr(args, 'enable_advanced_cues', False),
            enable_advanced_beatgrid=getattr(args, 'enable_advanced_beatgrid', False),
            enable_calibrated_energy=getattr(args, 'enable_calibrated_energy', False),
            enable_professional_reporting=getattr(args, 'enable_professional_reporting', True),
            
            # Caching
            use_cache=getattr(args, 'use_cache', True),
            cache_timeout_days=getattr(args, 'cache_timeout_days', 30),
            force_refresh=getattr(args, 'force_refresh', False),
            
            # Rekordbox
            rekordbox_xml_path=getattr(args, 'rekordbox_xml_path', None),
            update_rekordbox=getattr(args, 'update_rekordbox', False),
            
            # Reporting
            generate_report=getattr(args, 'generate_report', True),
            report_format=getattr(args, 'report_format', 'html'),
            report_path=getattr(args, 'report_path', None),
            
            # File organization
            organize_files=getattr(args, 'organize_files', False),
            output_directory=getattr(args, 'output_directory', None),
            
            # Quality filtering
            min_quality_score=getattr(args, 'min_quality_score', 0.0),
            skip_duplicates=getattr(args, 'skip_duplicates', True)
        )
    
    def _process_single_file(self, filepath: str, options: ProcessingOptions) -> int:
        """Process a single file"""
        result = self.app.process_single_file(filepath, options)
        
        if result.success:
            print(f"\n✅ Processing completed successfully")
            if result.metadata:
                print(f"   Title: {result.metadata.title}")
                print(f"   Artist: {result.metadata.artist}")
                if result.metadata.bmp:
                    print(f"   BPM: {result.metadata.bmp}")
                if result.metadata.musical_key:
                    print(f"   Key: {result.metadata.musical_key}")
            return 0
        else:
            print(f"\n❌ Processing failed")
            for error in result.errors:
                print(f"   Error: {error}")
            return 1
    
    def _process_folder(self, folder_path: str, options: ProcessingOptions, args) -> int:
        """Process a folder of files"""
        progress_callback = self._print_progress if self.verbose or options.workers == 1 else None
        
        batch_result = self.app.process_folder(
            folder_path,
            options,
            progress_callback=progress_callback
        )
        
        if batch_result.successful > 0:
            print(f"\n✅ Batch processing completed")
            print(f"   Successful: {batch_result.successful}")
            print(f"   Failed: {batch_result.failed}")
            print(f"   Total time: {batch_result.total_time:.1f}s")
            return 0
        else:
            print(f"\n❌ Batch processing failed - no files processed successfully")
            return 1
    
    def _list_batch_sessions(self, batch_processor: BatchProcessor) -> int:
        """List available batch sessions"""
        sessions = batch_processor.list_sessions()
        
        if not sessions:
            print("📋 No batch sessions found")
            return 0
        
        print("📋 Available Batch Sessions:")
        for session in sessions:
            print(f"   {session['session_id']}")
            print(f"      Name: {session['name']}")
            print(f"      Items: {session['item_count']}")
            print(f"      Created: {time.ctime(session['created_at'])}")
            print()
        
        return 0
    
    def _resume_batch_session(self, batch_processor: BatchProcessor, session_id: str, args) -> int:
        """Resume a batch session"""
        print(f"🔄 Resuming batch session: {session_id}")
        
        # Load session
        if not batch_processor.load_session(session_id):
            self._print_error(f"Failed to load session: {session_id}")
            return 1
        
        # Process session (would need processor functions)
        # This is a simplified version - full implementation would need service integration
        results = batch_processor.process_session(session_id)
        
        if results.get('status') == 'completed':
            print("✅ Batch session completed")
            return 0
        else:
            self._print_error(f"Batch session failed: {results.get('error', 'Unknown error')}")
            return 1
    
    def _create_batch_session(self, batch_processor: BatchProcessor, args) -> int:
        """Create a new batch session"""
        # This would implement new batch session creation
        self._print_error("Batch session creation not yet implemented")
        return 1
    
    def _list_export_formats(self, export_service: ExportService) -> int:
        """List available export formats"""
        formats = export_service.get_supported_formats()
        
        print("📤 Supported Export Formats:")
        for fmt in formats:
            print(f"   {fmt.name} (.{fmt.extension})")
            print(f"      {fmt.description}")
            features = []
            if fmt.supports_metadata:
                features.append("metadata")
            if fmt.supports_cues:
                features.append("cues")
            if fmt.supports_beatgrid:
                features.append("beatgrid")
            if fmt.supports_playlists:
                features.append("playlists")
            print(f"      Features: {', '.join(features)}")
            print()
        
        return 0
    
    def _load_tracks_for_export(self, source: str, args) -> List[Dict[str, Any]]:
        """Load tracks for export from various sources"""
        # This would implement track loading from different sources
        # For now, return empty list
        return []
    
    def _monitor_export_progress(self, export_service: ExportService, job_id: str):
        """Monitor export progress"""
        while True:
            status = export_service.get_job_status(job_id)
            if not status:
                break
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
            
            print(f"   Progress: {status['progress']:.1f}%")
            time.sleep(1)
        
        if status:
            if status['status'] == 'completed':
                print("✅ Export completed successfully")
            else:
                print(f"❌ Export {status['status']}")
                for error in status.get('errors', []):
                    print(f"   Error: {error}")
    
    def _print_progress(self, current: int, total: int, filepath: str):
        """Print processing progress"""
        percentage = (current / total) * 100
        filename = os.path.basename(filepath)
        
        # Truncate filename if too long
        if len(filename) > 50:
            filename = filename[:47] + "..."
        
        print(f"   [{current:4d}/{total:4d}] ({percentage:5.1f}%) {filename}")
    
    def _batch_progress_callback(self, progress: Dict[str, Any]):
        """Callback for batch processing progress"""
        if self.verbose:
            print(f"   Batch Progress: {progress['progress_percent']:.1f}% - ETA: {progress['eta']}")
    
    def _print_header(self):
        """Print application header"""
        print("🚀 DJ Music Cleaner - Unified Edition")
        print("   Professional Audio Library Management System")
        print("   Service-Oriented Architecture | Multiprocessing-Safe")
        print()
    
    def _print_config(self, args, options: ProcessingOptions):
        """Print processing configuration"""
        print("⚙️ Processing Configuration:")
        if os.path.isfile(args.path):
            print(f"   Target: Single file - {os.path.basename(args.path)}")
        else:
            print(f"   Target: Folder - {args.path}")
        print(f"   Workers: {options.workers}")
        print(f"   Online enhancement: {options.enhance_online}")
        print(f"   Caching: {options.use_cache}")
        print(f"   Rename files: {options.allow_rename}")
        print(f"   Dry run: {options.dry_run}")
        print()
    
    def _print_completion_stats(self, total_time: float):
        """Print completion statistics"""
        print(f"\n📈 Session completed in {total_time:.1f}s")
        
        if self.verbose and self.app:
            stats = self.app.get_session_stats()
            print("\n📊 Final Statistics:")
            session_stats = stats.get('session', {})
            print(f"   Files processed: {session_stats.get('files_processed', 0)}")
            print(f"   Success rate: {session_stats.get('success_rate', 0)}%")
            print(f"   Cache hits: {session_stats.get('cache_hits', 0)}")
            print(f"   Average time per file: {session_stats.get('average_processing_time', 0):.2f}s")
    
    def _print_error(self, message: str):
        """Print error message"""
        print(f"❌ {message}")
    
    def _print_warning(self, message: str):
        """Print warning message"""
        print(f"⚠️ {message}")


__all__ = ['CLICommands']