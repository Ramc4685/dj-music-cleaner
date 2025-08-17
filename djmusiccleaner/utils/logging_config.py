"""
Comprehensive Logging Configuration for DJ Music Cleaner

This module provides centralized logging configuration for the entire application,
ensuring consistent logging across all services and operations.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class DJMusicCleanerLogger:
    """Centralized logger configuration for DJ Music Cleaner"""
    
    def __init__(self, log_dir: Optional[str] = None, console_level: str = "INFO", 
                 file_level: str = "DEBUG", enable_console: bool = True):
        """
        Initialize comprehensive logging system
        
        Args:
            log_dir: Directory for log files (default: ~/.dj_music_cleaner_unified/logs)
            console_level: Console logging level
            file_level: File logging level  
            enable_console: Whether to enable console logging
        """
        self.log_dir = log_dir or os.path.expanduser('~/.dj_music_cleaner_unified/logs')
        self.console_level = getattr(logging, console_level.upper())
        self.file_level = getattr(logging, file_level.upper())
        self.enable_console = enable_console
        
        # Ensure log directory exists
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        self._setup_root_logger()
        
        # Configure component loggers
        self._setup_component_loggers()
        
    def _setup_root_logger(self):
        """Setup root logger with handlers"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level)
            console_formatter = ColoredFormatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Main log file (rotating)
        main_log_file = os.path.join(self.log_dir, 'dj_music_cleaner.log')
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(self.file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Session-specific log file
        session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_log_file = os.path.join(self.log_dir, f'session_{session_timestamp}.log')
        session_handler = logging.FileHandler(session_log_file)
        session_handler.setLevel(logging.DEBUG)
        session_handler.setFormatter(file_formatter)
        root_logger.addHandler(session_handler)
        
        # Performance log file (for detailed timing analysis)
        perf_log_file = os.path.join(self.log_dir, f'performance_{session_timestamp}.log')
        self.perf_handler = logging.FileHandler(perf_log_file)
        self.perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter(
            '%(asctime)s,%(message)s',  # CSV-like format for analysis
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.perf_handler.setFormatter(perf_formatter)
        
        # Error log file (errors and warnings only)
        error_log_file = os.path.join(self.log_dir, f'errors_{session_timestamp}.log')
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
    def _setup_component_loggers(self):
        """Setup loggers for specific components"""
        # Component loggers with specific levels
        components = {
            'djmusiccleaner.main': logging.INFO,
            'djmusiccleaner.processing': logging.INFO,
            'djmusiccleaner.audio_analysis': logging.DEBUG,
            'djmusiccleaner.metadata': logging.DEBUG,
            'djmusiccleaner.cache': logging.INFO,
            'djmusiccleaner.file_ops': logging.INFO,
            'djmusiccleaner.analytics': logging.DEBUG,
            'djmusiccleaner.batch': logging.INFO,
            'djmusiccleaner.performance': logging.DEBUG,
        }
        
        for component, level in components.items():
            logger = logging.getLogger(component)
            logger.setLevel(level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific component"""
        return logging.getLogger(f'djmusiccleaner.{name}')
    
    def get_performance_logger(self) -> logging.Logger:
        """Get performance logger for timing data"""
        perf_logger = logging.getLogger('djmusiccleaner.performance')
        if not any(h == self.perf_handler for h in perf_logger.handlers):
            perf_logger.addHandler(self.perf_handler)
        return perf_logger
    
    def log_processing_start(self, filepath: str, options: Dict[str, Any]):
        """Log start of file processing"""
        logger = self.get_logger('processing')
        logger.info(f"Starting processing: {os.path.basename(filepath)}")
        logger.debug(f"Full path: {filepath}")
        logger.debug(f"Options: {options}")
    
    def log_processing_complete(self, filepath: str, success: bool, processing_time: float, 
                               operations: list, errors: list = None):
        """Log completion of file processing"""
        logger = self.get_logger('processing')
        perf_logger = self.get_performance_logger()
        
        basename = os.path.basename(filepath)
        if success:
            logger.info(f"Completed processing: {basename} in {processing_time:.2f}s")
            perf_logger.info(f"FILE_COMPLETE,{filepath},{processing_time:.3f},{len(operations)},{success}")
        else:
            logger.error(f"Failed processing: {basename} in {processing_time:.2f}s")
            if errors:
                for error in errors:
                    logger.error(f"  Error: {error}")
            perf_logger.info(f"FILE_FAILED,{filepath},{processing_time:.3f},{len(operations)},{success}")
    
    def log_batch_start(self, folder_path: str, file_count: int, workers: int, options: Dict[str, Any]):
        """Log start of batch processing"""
        logger = self.get_logger('batch')
        perf_logger = self.get_performance_logger()
        
        logger.info(f"Starting batch processing: {folder_path}")
        logger.info(f"Files to process: {file_count}, Workers: {workers}")
        logger.debug(f"Batch options: {options}")
        
        perf_logger.info(f"BATCH_START,{folder_path},{file_count},{workers}")
    
    def log_batch_complete(self, folder_path: str, total_files: int, successful: int, 
                          failed: int, total_time: float):
        """Log completion of batch processing"""
        logger = self.get_logger('batch')
        perf_logger = self.get_performance_logger()
        
        success_rate = (successful / total_files * 100) if total_files > 0 else 0
        logger.info(f"Batch processing complete: {folder_path}")
        logger.info(f"Results: {successful}/{total_files} successful ({success_rate:.1f}%)")
        logger.info(f"Total time: {total_time:.1f}s, Average: {total_time/total_files:.2f}s per file")
        
        perf_logger.info(f"BATCH_COMPLETE,{folder_path},{total_files},{successful},{failed},{total_time:.3f}")
    
    def log_service_operation(self, service: str, operation: str, duration: float, 
                             success: bool, details: Dict[str, Any] = None):
        """Log service operations with timing"""
        logger = self.get_logger(service.lower())
        perf_logger = self.get_performance_logger()
        
        if success:
            logger.debug(f"{operation} completed in {duration:.3f}s")
            if details:
                logger.debug(f"  Details: {details}")
        else:
            logger.error(f"{operation} failed after {duration:.3f}s")
            if details and 'error' in details:
                logger.error(f"  Error: {details['error']}")
        
        perf_logger.info(f"SERVICE,{service},{operation},{duration:.3f},{success}")
    
    def log_cache_operation(self, operation: str, key: str, hit: bool, duration: float = 0):
        """Log cache operations"""
        logger = self.get_logger('cache')
        perf_logger = self.get_performance_logger()
        
        if operation == 'get':
            status = 'HIT' if hit else 'MISS'
            logger.debug(f"Cache {status}: {os.path.basename(key)}")
        elif operation == 'set':
            logger.debug(f"Cache SET: {os.path.basename(key)}")
        elif operation == 'cleanup':
            logger.info(f"Cache cleanup: {key}")
        
        perf_logger.info(f"CACHE,{operation},{hit},{duration:.3f}")
    
    def log_file_operation(self, operation: str, source: str, target: str = None, 
                          success: bool = True, error: str = None):
        """Log file operations"""
        logger = self.get_logger('file_ops')
        
        if success:
            if target:
                logger.info(f"{operation}: {os.path.basename(source)} -> {os.path.basename(target)}")
            else:
                logger.info(f"{operation}: {os.path.basename(source)}")
        else:
            logger.error(f"{operation} failed: {os.path.basename(source)}")
            if error:
                logger.error(f"  Error: {error}")
    
    def log_audio_analysis(self, filepath: str, analysis_type: str, result: Dict[str, Any], 
                          duration: float, engine: str = 'aubio'):
        """Log audio analysis operations"""
        logger = self.get_logger('audio_analysis')
        perf_logger = self.get_performance_logger()
        
        basename = os.path.basename(filepath)
        logger.debug(f"Audio analysis ({engine}): {basename} - {analysis_type}")
        logger.debug(f"  Duration: {duration:.3f}s, Results: {result}")
        
        perf_logger.info(f"AUDIO,{analysis_type},{engine},{duration:.3f},{len(result)}")
    
    def log_metadata_extraction(self, filepath: str, source: str, metadata: Dict[str, Any], 
                               duration: float, enhanced: bool = False):
        """Log metadata extraction"""
        logger = self.get_logger('metadata')
        perf_logger = self.get_performance_logger()
        
        basename = os.path.basename(filepath)
        fields_extracted = len([v for v in metadata.values() if v])
        
        logger.debug(f"Metadata extraction ({source}): {basename}")
        logger.debug(f"  Fields extracted: {fields_extracted}, Enhanced: {enhanced}")
        
        perf_logger.info(f"METADATA,{source},{duration:.3f},{fields_extracted},{enhanced}")
    
    def log_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Log errors with context"""
        logger = self.get_logger(component)
        
        logger.error(f"Error in {component}: {type(error).__name__}: {str(error)}")
        if context:
            logger.error(f"  Context: {context}")
            
        # Log stack trace at debug level
        logger.debug("Stack trace:", exc_info=True)
    
    def log_performance_summary(self, stats: Dict[str, Any]):
        """Log performance summary"""
        logger = self.get_logger('performance')
        perf_logger = self.get_performance_logger()
        
        logger.info("Performance Summary:")
        logger.info(f"  Files processed: {stats.get('files_processed', 0)}")
        logger.info(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
        logger.info(f"  Average processing time: {stats.get('avg_processing_time', 0):.3f}s")
        logger.info(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")
        
        perf_logger.info(f"SUMMARY,{stats.get('files_processed', 0)},{stats.get('success_rate', 0):.1f},{stats.get('avg_processing_time', 0):.3f}")


# Global logger instance
_logger_instance = None

def setup_logging(log_dir: Optional[str] = None, console_level: str = "INFO", 
                  file_level: str = "DEBUG", enable_console: bool = True) -> DJMusicCleanerLogger:
    """Setup global logging configuration"""
    global _logger_instance
    _logger_instance = DJMusicCleanerLogger(log_dir, console_level, file_level, enable_console)
    return _logger_instance

def get_logger(name: str = 'main') -> logging.Logger:
    """Get a component logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = setup_logging()
    return _logger_instance.get_logger(name)

def get_app_logger() -> DJMusicCleanerLogger:
    """Get the application logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = setup_logging()
    return _logger_instance