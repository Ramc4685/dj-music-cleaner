"""
Unified CLI Interface for DJ Music Cleaner

Consolidates CLI functionality from both implementations providing:
- Single command with all options
- Backward compatibility
- Progressive complexity (simple to advanced options)
- Professional output formatting
"""

import argparse
import sys
import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

from ..core.models import ProcessingOptions
from ..core.exceptions import DJMusicCleanerError
from ..dj_music_cleaner_unified import DJMusicCleanerUnified
from .config_validator import validate_config


def create_parser() -> argparse.ArgumentParser:
    """Create unified argument parser with all options"""
    
    parser = argparse.ArgumentParser(
        description="DJ Music Cleaner - Professional Audio Library Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/music                    # Process folder with default settings
  %(prog)s /path/to/music --workers 4       # Use 4 parallel workers  
  %(prog)s /path/to/music --dry-run          # Preview changes without modifying
  %(prog)s song.mp3 --enhance-online         # Process single file with online lookup
  %(prog)s /music --rename --year-in-filename # Rename files with year
  %(prog)s /music --no-cache --force-refresh  # Skip cache and force reanalysis

Advanced:
  %(prog)s /music --advanced-cues --professional-reporting
  %(prog)s /music --organize --output-dir /organized --workers 8
        """
    )
    
    # Positional argument
    parser.add_argument('path', 
                       help='Path to audio file or folder to process')
    
    # Basic options
    basic_group = parser.add_argument_group('Basic Options')
    basic_group.add_argument('--dry-run', action='store_true',
                            help='Preview changes without modifying files')
    basic_group.add_argument('--workers', type=int, default=1, metavar='N',
                            help='Number of parallel workers (default: 1)')
    basic_group.add_argument('--skip-existing', action='store_true',
                            help='Skip files that are already processed')
    
    # File operations
    file_group = parser.add_argument_group('File Operations')
    file_group.add_argument('--rename', action='store_true', dest='allow_rename',
                           help='Allow renaming files based on metadata')
    file_group.add_argument('--year-in-filename', action='store_true', dest='include_year_in_filename',
                           help='Include year in renamed filenames')
    file_group.add_argument('--organize', action='store_true', dest='organize_files',
                           help='Organize files into folder structure')
    file_group.add_argument('--output-dir', dest='output_directory', metavar='DIR',
                           help='Output directory for organized files')
    
    # Metadata options
    metadata_group = parser.add_argument_group('Metadata Processing')
    metadata_group.add_argument('--enhance-online', action='store_true',
                               help='Enable online metadata enhancement')
    metadata_group.add_argument('--no-online', action='store_false', dest='enhance_online',
                               help='Disable online metadata enhancement')
    metadata_group.add_argument('--acoustid-api-key', dest='acoustid_api_key', metavar='KEY',
                               help='AcoustID API key for enhanced identification (overrides ACOUSTID_API_KEY env var)')
    
    # Audio analysis
    analysis_group = parser.add_argument_group('Audio Analysis')
    analysis_group.add_argument('--analyze-bpm', action='store_true', default=True,
                               help='Analyze BPM/tempo (default: enabled)')
    analysis_group.add_argument('--no-bpm', action='store_false', dest='analyze_bpm',
                               help='Skip BPM analysis')
    analysis_group.add_argument('--analyze-key', action='store_true', default=True,
                               help='Analyze musical key (default: enabled)')
    analysis_group.add_argument('--no-key', action='store_false', dest='analyze_key',
                               help='Skip key analysis')
    analysis_group.add_argument('--analyze-energy', action='store_true', default=True,
                               help='Analyze energy level (default: enabled)')
    analysis_group.add_argument('--no-energy', action='store_false', dest='analyze_energy',
                               help='Skip energy analysis')
    analysis_group.add_argument('--analyze-quality', action='store_true',
                               help='Analyze audio quality (slower)')
    
    # Advanced features
    advanced_group = parser.add_argument_group('Advanced Features')
    advanced_group.add_argument('--advanced-cues', action='store_true', dest='enable_advanced_cues',
                               help='Enable advanced cue point detection')
    advanced_group.add_argument('--advanced-beatgrid', action='store_true', dest='enable_advanced_beatgrid',
                               help='Enable advanced beat grid analysis')
    advanced_group.add_argument('--calibrated-energy', action='store_true', dest='enable_calibrated_energy',
                               help='Enable calibrated energy analysis')
    advanced_group.add_argument('--professional-reporting', action='store_true', dest='enable_professional_reporting',
                               help='Enable professional analytics and reporting')
    
    # Cache options
    cache_group = parser.add_argument_group('Cache Options')
    cache_group.add_argument('--no-cache', action='store_false', dest='use_cache',
                            help='Disable caching')
    cache_group.add_argument('--force-refresh', action='store_true',
                            help='Force refresh of cached data')
    cache_group.add_argument('--cache-timeout', type=int, default=30, dest='cache_timeout_days',
                            metavar='DAYS', help='Cache timeout in days (default: 30)')
    
    # Rekordbox integration  
    rekordbox_group = parser.add_argument_group('Rekordbox Integration')
    rekordbox_group.add_argument('--rekordbox-xml', dest='rekordbox_xml_path', metavar='FILE',
                                help='Path to Rekordbox XML database')
    rekordbox_group.add_argument('--update-rekordbox', action='store_true',
                                help='Update Rekordbox XML with enhanced metadata')
    rekordbox_group.add_argument('--rekordbox-backup', action='store_true', default=True,
                                help='Create backup before updating Rekordbox XML (default: enabled)')
    rekordbox_group.add_argument('--no-rekordbox-backup', action='store_false', dest='rekordbox_backup',
                                help='Skip backup creation')
    rekordbox_group.add_argument('--rekordbox-report', dest='rekordbox_report_path', metavar='FILE',
                                help='Generate detailed Rekordbox integration report')
    rekordbox_group.add_argument('--preserve-rekordbox-data', action='store_true', default=True,
                                help='Preserve existing Rekordbox data (default: enabled)')
    rekordbox_group.add_argument('--overwrite-rekordbox-data', action='store_false', dest='preserve_rekordbox_data',
                                help='Overwrite existing Rekordbox metadata with cleaned data')
    
    # Quality and filtering
    quality_group = parser.add_argument_group('Quality and Filtering')
    quality_group.add_argument('--min-quality', type=float, default=0.0, dest='min_quality_score',
                              metavar='SCORE', help='Minimum quality score (0-10)')
    quality_group.add_argument('--skip-duplicates', action='store_true', default=True,
                              help='Skip duplicate files (default: enabled)')
    quality_group.add_argument('--no-skip-duplicates', action='store_false', dest='skip_duplicates',
                              help='Process duplicate files')
    
    # Reporting options
    report_group = parser.add_argument_group('Reporting Options')
    report_group.add_argument('--no-report', action='store_false', dest='generate_report',
                             help='Disable report generation')
    report_group.add_argument('--report-format', choices=['html', 'json', 'csv'],
                             help='Report format (default: html)')
    report_group.add_argument('--report-path', metavar='FILE',
                             help='Custom path for generated report')
    
    # Logging options
    logging_group = parser.add_argument_group('Logging Options')
    logging_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                              default='INFO', help='Console logging level (default: INFO)')
    logging_group.add_argument('--log-file-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                              default='DEBUG', help='File logging level (default: DEBUG)')
    logging_group.add_argument('--log-dir', metavar='DIR',
                              help='Directory for log files (default: ~/.dj_music_cleaner_unified/logs)')
    logging_group.add_argument('--no-console-log', action='store_true',
                              help='Disable console logging (file logging only)')
    
    # Utility options
    utility_group = parser.add_argument_group('Utility Options')
    utility_group.add_argument('--stats', action='store_true',
                              help='Show performance statistics and exit')
    utility_group.add_argument('--optimize', action='store_true',
                              help='Optimize performance and exit')
    utility_group.add_argument('--config', metavar='FILE',
                              help='Load configuration from JSON file')
    utility_group.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose output')
    utility_group.add_argument('--version', action='version', version='DJ Music Cleaner 2.0 Unified')
    
    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        sys.exit(1)


def create_processing_options(args, config: Dict[str, Any] = None) -> ProcessingOptions:
    """Create ProcessingOptions from command line arguments and config"""
    config = config or {}
    
    # Helper to get value from args, config, or default
    def get_value(arg_name: str, config_section: str, config_key: str, default_value: Any = None):
        # CLI args override config, config overrides defaults
        if hasattr(args, arg_name):
            cli_value = getattr(args, arg_name, None)
            # For boolean store_true actions, only use CLI value if explicitly set
            if arg_name in ['update_rekordbox'] and cli_value is False:
                # If it's False and this is a store_true action, it wasn't explicitly set
                # So use config value instead
                pass
            elif cli_value is not None:
                return cli_value
        
        return config.get(config_section, {}).get(config_key, default_value)
    
    return ProcessingOptions(
        # Core processing  
        enhance_online=get_value('enhance_online', 'processing', 'enhance_online', True),
        acoustid_api_key=get_value('acoustid_api_key', 'processing', 'acoustid_api_key', None),
        skip_existing=get_value('skip_existing', 'processing', 'skip_existing', True),
        allow_rename=get_value('allow_rename', 'file_operations', 'allow_rename', False),
        include_year_in_filename=get_value('include_year_in_filename', 'file_operations', 'include_year_in_filename', False),
        workers=get_value('workers', 'processing', 'workers', 1),
        dry_run=get_value('dry_run', 'processing', 'dry_run', False),
        
        # Audio analysis
        analyze_bpm=get_value('analyze_bpm', 'audio_analysis', 'analyze_bpm', True),
        analyze_key=get_value('analyze_key', 'audio_analysis', 'analyze_key', True),
        analyze_energy=get_value('analyze_energy', 'audio_analysis', 'analyze_energy', True),
        analyze_quality=get_value('analyze_quality', 'audio_analysis', 'analyze_quality', False),
        
        # Advanced features
        enable_advanced_cues=get_value('enable_advanced_cues', 'advanced_features', 'enable_advanced_cues', False),
        enable_advanced_beatgrid=get_value('enable_advanced_beatgrid', 'advanced_features', 'enable_advanced_beatgrid', False),
        enable_calibrated_energy=get_value('enable_calibrated_energy', 'advanced_features', 'enable_calibrated_energy', False),
        enable_professional_reporting=get_value('enable_professional_reporting', 'advanced_features', 'enable_professional_reporting', True),
        
        # Caching
        use_cache=get_value('use_cache', 'cache', 'use_cache', True),
        cache_timeout_days=get_value('cache_timeout_days', 'cache', 'cache_timeout_days', 30),
        force_refresh=get_value('force_refresh', 'cache', 'force_refresh', False),
        
        # Rekordbox
        rekordbox_xml_path=get_value('rekordbox_xml_path', 'rekordbox', 'xml_path', None),
        update_rekordbox=get_value('update_rekordbox', 'rekordbox', 'update_rekordbox', False),
        rekordbox_backup=get_value('rekordbox_backup', 'rekordbox', 'create_backup', True),
        preserve_rekordbox_data=get_value('preserve_rekordbox_data', 'rekordbox', 'preserve_existing_data', True),
        rekordbox_report_path=get_value('rekordbox_report_path', 'rekordbox', 'report_path', None),
        
        # Reporting
        generate_report=get_value('generate_report', 'reporting', 'generate_report', True),
        report_format=get_value('report_format', 'reporting', 'report_format', 'html'),
        report_path=get_value('report_path', 'reporting', 'report_path', None),
        
        # File organization
        organize_files=get_value('organize_files', 'file_operations', 'organize_files', False),
        output_directory=get_value('output_directory', 'file_operations', 'output_directory', None),
        
        # Quality filtering
        min_quality_score=get_value('min_quality_score', 'quality_filtering', 'min_quality_score', 0.0),
        skip_duplicates=get_value('skip_duplicates', 'quality_filtering', 'skip_duplicates', True)
    )


def print_progress(current: int, total: int, filepath: str):
    """Print processing progress"""
    percentage = (current / total) * 100
    filename = os.path.basename(filepath)
    
    # Truncate filename if too long
    if len(filename) > 50:
        filename = filename[:47] + "..."
    
    print(f"   [{current:4d}/{total:4d}] ({percentage:5.1f}%) {filename}")


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration file - use default if not specified
    config = {}
    config_path = getattr(args, 'config', None)
    
    if config_path:
        # User specified a config file
        config = load_config_file(config_path)
    else:
        # Try to load default config from project root
        default_config_path = os.path.join(os.getcwd(), 'dj_music_cleaner_config.json')
        if os.path.exists(default_config_path):
            print(f"📄 Using default config: {default_config_path}")
            try:
                config = load_config_file(default_config_path)
            except Exception as e:
                print(f"⚠️  Warning: Could not load default config, using CLI defaults: {e}")
        else:
            print("📄 No config file found, using CLI defaults")
    
    # Validate configuration and create directories
    if config:
        print("\n🔍 Validating configuration...")
        config = validate_config(config)
        
    # Ensure analytics config exists with proper settings
    if 'analytics' not in config:
        config['analytics'] = {}
        
    # Propagate professional reporting flag if enabled
    if hasattr(args, 'enable_professional_reporting') and args.enable_professional_reporting:
        config['analytics']['advanced_reporting'] = True
        config['analytics']['detailed_tracking'] = True
    
    # Setup logging configuration
    if 'logging' not in config:
        config['logging'] = {}
    
    # Apply command line logging options
    if hasattr(args, 'log_level'):
        config['logging']['console_level'] = args.log_level
    if hasattr(args, 'log_file_level'):
        config['logging']['file_level'] = args.log_file_level
    if hasattr(args, 'log_dir'):
        config['logging']['log_dir'] = args.log_dir
    if hasattr(args, 'no_console_log'):
        config['logging']['enable_console'] = not args.no_console_log
    
    try:
        # Initialize application
        print("🚀 DJ Music Cleaner - Unified Edition")
        print("   Professional Audio Library Management System")
        print("   Service-Oriented Architecture | Multiprocessing-Safe")
        if hasattr(args, 'verbose') and args.verbose:
            print(f"   Logging: Console={config['logging'].get('console_level', 'INFO')}, File={config['logging'].get('file_level', 'DEBUG')}")
        print()
        
        app = DJMusicCleanerUnified(config)
        
        # Handle utility commands
        if hasattr(args, 'stats') and args.stats:
            stats = app.get_session_stats()
            print("📊 Performance Statistics:")
            print(json.dumps(stats, indent=2))
            return
        
        if hasattr(args, 'optimize') and args.optimize:
            results = app.optimize_performance()
            print("🔧 Optimization Results:")
            print(json.dumps(results, indent=2))
            return
        
        # Validate path
        if not os.path.exists(args.path):
            print(f"❌ Path not found: {args.path}")
            sys.exit(1)
        
        # Create processing options with config
        options = create_processing_options(args, config)
        
        # Validate Rekordbox XML path if provided
        if options.rekordbox_xml_path:
            if os.path.isfile(options.rekordbox_xml_path):
                # Existing XML file - will import and update
                print(f"✅ Rekordbox XML file found: {options.rekordbox_xml_path}")
                print("   Will import existing collection and update with enhanced metadata")
            elif os.path.isdir(options.rekordbox_xml_path):
                # Directory provided - create new XML file in this directory
                xml_filename = f"rekordbox_export_{time.strftime('%Y%m%d_%H%M%S')}.xml"
                options.rekordbox_xml_path = os.path.join(options.rekordbox_xml_path, xml_filename)
                print(f"📁 Rekordbox XML directory provided, will create new file:")
                print(f"   {options.rekordbox_xml_path}")
            elif not os.path.exists(options.rekordbox_xml_path):
                # File path doesn't exist - will create new file
                # Ensure directory exists
                xml_dir = os.path.dirname(options.rekordbox_xml_path)
                if xml_dir and not os.path.exists(xml_dir):
                    os.makedirs(xml_dir, exist_ok=True)
                    print(f"📁 Created directory: {xml_dir}")
                print(f"🆕 Will create new Rekordbox XML file: {options.rekordbox_xml_path}")
            else:
                print(f"⚠️ Warning: Rekordbox XML path exists but is not a file: {options.rekordbox_xml_path}")
                print("   Rekordbox integration will be disabled for this session.")
                options.rekordbox_xml_path = None
                options.update_rekordbox = False
        
        # Print processing configuration
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
        print(f"   Update Rekordbox: {options.update_rekordbox}")
        print(f"   Rekordbox XML path: {options.rekordbox_xml_path}")
        print()
        
        # Process files
        start_time = time.time()
        
        if os.path.isfile(args.path):
            # Single file processing
            result = app.process_single_file(args.path, options)
            
            if result.success:
                print(f"\n✅ Processing completed successfully")
                if result.metadata:
                    print(f"   Title: {result.metadata.title}")
                    print(f"   Artist: {result.metadata.artist}")
                    if result.metadata.bpm:
                        print(f"   BPM: {result.metadata.bpm}")
                    if result.metadata.musical_key:
                        print(f"   Key: {result.metadata.musical_key}")
                
                # Rekordbox integration for single file
                if options.update_rekordbox and options.rekordbox_xml_path:
                    try:
                        print(f"\n🎛️ Integrating with Rekordbox...")
                        
                        # Load/create Rekordbox collection
                        app.load_rekordbox_xml(options.rekordbox_xml_path)
                        
                        # Save with the single file result
                        rekordbox_result = app.save_rekordbox_xml(options.rekordbox_xml_path, [result])
                        
                        print(f"🎛️ Rekordbox Integration Complete:")
                        print(f"   Enhanced tracks: {rekordbox_result.get('enhancements_applied', 0)}")
                        print(f"   XML file: {options.rekordbox_xml_path}")
                        print(f"   Validation: {'✅ Passed' if rekordbox_result.get('validation', {}).get('validation_passed') else '⚠️ Issues detected'}")
                        
                        if rekordbox_result.get('backup_created'):
                            print(f"   Backup created: ✅")
                    except Exception as e:
                        print(f"\n⚠️ Rekordbox integration failed: {str(e)}")
            else:
                print(f"\n❌ Processing failed")
                for error in result.errors:
                    print(f"   Error: {error}")
                sys.exit(1)
        
        else:
            # Folder processing
            progress_callback = print_progress if args.verbose or options.workers == 1 else None
            
            batch_result = app.process_folder(
                args.path, 
                options,
                progress_callback=progress_callback
            )
            
            if batch_result.successful > 0:
                print(f"\n✅ Batch processing completed")
            else:
                print(f"\n❌ Batch processing failed - no files processed successfully")
                sys.exit(1)
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n📈 Session completed in {total_time:.1f}s")
        
        if hasattr(args, 'verbose') and args.verbose:
            stats = app.get_session_stats()
            print("\n📊 Final Statistics:")
            print(f"   Files processed: {stats['session']['files_processed']}")
            print(f"   Success rate: {stats['session']['success_rate']}%")
            print(f"   Cache hits: {stats['session']['cache_hits']}")
            print(f"   Average time per file: {stats['session']['average_processing_time']:.2f}s")
        
        # Generate reports if enabled
        if options.generate_report:
            try:
                report_format = options.report_format
                report_path = options.report_path
                
                # Handle directory paths for reports
                if report_path and os.path.isdir(report_path):
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    report_filename = f"dj_music_cleaner_report_{timestamp}.{report_format}"
                    report_path = os.path.join(report_path, report_filename)
                
                # If no specific path is provided but output_directory exists, put report there
                elif not report_path and options.output_directory:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    report_path = os.path.join(options.output_directory, f"dj_music_cleaner_report_{timestamp}.{report_format}")
                
                # Generate the report
                report_file = app.analytics_service.export_report(report_format, report_path)
                print(f"\n📊 Analytics report generated: {report_file}")
            except Exception as e:
                print(f"\n⚠️ Failed to generate report: {str(e)}")
                
        # Cleanup
        app.cleanup_resources()
        
    except DJMusicCleanerError as e:
        print(f"❌ Application Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()