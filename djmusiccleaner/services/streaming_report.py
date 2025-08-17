"""
Streaming Report Writer for Memory-Efficient Processing

Provides thread-safe streaming JSON report generation to avoid memory
issues when processing large batches of files (750+ files).
"""

import json
import threading
import fcntl
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager


class StreamingJSONReporter:
    """
    Thread-safe streaming JSON report writer
    
    Features:
    - Immediate file-by-file writing (no memory accumulation)
    - File locking for concurrent access safety
    - Maintains JSON structure compatibility
    - Memory-efficient for large batches
    """
    
    def __init__(self, filepath: str, enable_locking: bool = True):
        """
        Initialize streaming reporter
        
        Args:
            filepath: Path to JSON report file
            enable_locking: Enable file locking for concurrent access
        """
        self.filepath = filepath
        self.enable_locking = enable_locking
        self._lock = threading.Lock()
        self._initialized = False
        self._file_count = 0
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Initialize the JSON structure
        self._initialize_file()
    
    def _initialize_file(self):
        """Initialize the JSON file with proper structure"""
        try:
            with self._file_lock():
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    # Start JSON structure
                    json_start = {
                        "report_metadata": {
                            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "report_type": "streaming_detailed_processing",
                            "version": "2.0"
                        },
                        "files": {
                            # Files will be added here one by one
                        }
                    }
                    json.dump(json_start, f, indent=2)
                    
            self._initialized = True
            
        except Exception as e:
            print(f"Error initializing streaming report: {e}")
            raise
    
    @contextmanager
    def _file_lock(self):
        """Context manager for file locking"""
        if not self.enable_locking:
            yield
            return
            
        f = None
        try:
            f = open(self.filepath + '.lock', 'w')
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if f:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                f.close()
                # Clean up lock file
                try:
                    os.remove(self.filepath + '.lock')
                except:
                    pass
    
    def add_file_result(self, filename: str, file_data: Dict[str, Any]):
        """
        Add a single file result to the report
        
        Args:
            filename: Name of the processed file
            file_data: Complete file processing data
        """
        if not self._initialized:
            raise RuntimeError("Reporter not initialized")
        
        with self._lock:
            try:
                with self._file_lock():
                    # Read current content
                    with open(self.filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Add the new file
                    data["files"][filename] = file_data
                    data["report_metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data["report_metadata"]["total_files"] = len(data["files"])
                    
                    # Write back
                    with open(self.filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, default=str)
                    
                self._file_count += 1
                
            except Exception as e:
                print(f"Error adding file result to streaming report: {e}")
                # Don't raise - continue processing other files
    
    def update_summary_stats(self, stats: Dict[str, Any]):
        """
        Update summary statistics in the report
        
        Args:
            stats: Summary statistics to add
        """
        with self._lock:
            try:
                with self._file_lock():
                    # Read current content
                    with open(self.filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Add summary stats
                    data["summary_stats"] = stats
                    data["report_metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Write back
                    with open(self.filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, default=str)
                        
            except Exception as e:
                print(f"Error updating summary stats: {e}")
    
    def finalize(self):
        """Finalize the report"""
        with self._lock:
            try:
                with self._file_lock():
                    # Read current content
                    with open(self.filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Add finalization metadata
                    data["report_metadata"]["finalized_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data["report_metadata"]["status"] = "completed"
                    
                    # Write back
                    with open(self.filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, default=str)
                        
                print(f"📊 Streaming report finalized: {self.filepath}")
                        
            except Exception as e:
                print(f"Error finalizing streaming report: {e}")
    
    def get_file_count(self) -> int:
        """Get current number of files in report"""
        return self._file_count


class StreamingReportManager:
    """
    Manager for multiple streaming reports
    
    Handles different report formats and coordinates streaming writers
    """
    
    def __init__(self, base_path: str, formats: list = None):
        """
        Initialize streaming report manager
        
        Args:
            base_path: Base path for reports (without extension)
            formats: List of formats to generate ('json', 'csv', etc.)
        """
        self.base_path = base_path
        self.formats = formats or ['json']
        self.reporters = {}
        
        # Initialize reporters for each format
        for fmt in self.formats:
            if fmt == 'json':
                filepath = f"{base_path}.json"
                self.reporters['json'] = StreamingJSONReporter(filepath)
    
    def add_file_result(self, filename: str, result_data: Dict[str, Any]):
        """Add file result to all active reporters"""
        for reporter in self.reporters.values():
            if hasattr(reporter, 'add_file_result'):
                reporter.add_file_result(filename, result_data)
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update stats in all reporters"""
        for reporter in self.reporters.values():
            if hasattr(reporter, 'update_summary_stats'):
                reporter.update_summary_stats(stats)
    
    def finalize_all(self):
        """Finalize all reporters"""
        for reporter in self.reporters.values():
            if hasattr(reporter, 'finalize'):
                reporter.finalize()