"""
Professional Analytics Service

Provides comprehensive analytics and reporting capabilities for DJ Music Cleaner.
Tracks performance metrics, collection statistics, and generates professional
reports for DJs and music library managers.
"""

import time
import threading
import json
import statistics
from typing import Dict, List, Any, Optional, DefaultDict, Tuple, Set, Union
import time
import json
from collections import defaultdict
import threading
import os
from datetime import datetime
from dataclasses import dataclass, field

from ..core.models import TrackMetadata, ProcessingResult, BatchProcessingResult


@dataclass
class CollectionAnalytics:
    """Analytics data for a music collection"""
    total_tracks: int = 0
    total_duration_hours: float = 0.0
    total_size_gb: float = 0.0
    genre_distribution: Dict[str, int] = field(default_factory=dict)
    bpm_distribution: Dict[str, int] = field(default_factory=dict)
    key_distribution: Dict[str, int] = field(default_factory=dict)
    year_distribution: Dict[str, int] = field(default_factory=dict)
    format_distribution: Dict[str, int] = field(default_factory=dict)
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    average_bpm: float = 0.0
    most_common_key: str = ""
    most_common_genre: str = ""
    decade_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass  
class PerformanceMetrics:
    """Performance metrics for processing operations"""
    operation_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_time_seconds: float = 0.0
    average_time_seconds: float = 0.0
    min_time_seconds: float = float('inf')
    max_time_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def success_rate(self) -> float:
        return (self.successful_operations / self.total_operations * 100) if self.total_operations > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total_cache_ops = self.cache_hits + self.cache_misses
        return (self.cache_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0.0


class AnalyticsService:
    """
    Professional analytics and reporting service
    
    Features:
    - Real-time performance monitoring
    - Collection analysis and statistics
    - Trend analysis over time
    - Quality assessment and recommendations
    - Professional report generation
    - Export capabilities (JSON, CSV, HTML)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the analytics service"""
        self.config = config or {}
        
        # Configuration
        self.enable_detailed_tracking = self.config.get('detailed_tracking', True)
        self.max_history_days = self.config.get('max_history_days', 30)
        self.enable_realtime_stats = self.config.get('enable_realtime_stats', True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Analytics data
        self.session_start_time = time.time()
        self.performance_metrics: Dict[str, PerformanceMetrics] = defaultdict(lambda: PerformanceMetrics(""))
        self.collection_analytics = CollectionAnalytics()
        self.processing_history: List[Dict[str, Any]] = []
        
        # Detailed file tracking
        self.file_details: Dict[str, Dict[str, Any]] = {}
        self.session_summary: Dict[str, Any] = {
            'files_processed': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'online_enhancements': 0,
            'errors_by_type': defaultdict(int),
            'warnings_by_type': defaultdict(int)
        }
        
        # Trend data (time-series)
        self.daily_stats: Dict[str, Dict[str, Any]] = {}
        
    def record_processing_result(self, result: ProcessingResult):
        """
        Record a single file processing result
        
        Args:
            result: ProcessingResult to analyze and record
        """
        with self._lock:
            # Update session summary
            self.session_summary['files_processed'] += 1
            
            if result.success:
                self.session_summary['successful_processes'] += 1
            else:
                self.session_summary['failed_processes'] += 1
            
            self.session_summary['total_processing_time'] += result.processing_time
            
            if result.cache_hit:
                self.session_summary['cache_hits'] += 1
            
            if result.online_enhancement:
                self.session_summary['online_enhancements'] += 1
            
            # Record errors and warnings
            for error in result.errors:
                error_type = self._classify_error(error)
                self.session_summary['errors_by_type'][error_type] += 1
            
            for warning in result.warnings:
                warning_type = self._classify_warning(warning)
                self.session_summary['warnings_by_type'][warning_type] += 1
            
            # Update performance metrics for each operation
            for operation in result.operations_performed:
                metrics = self.performance_metrics[operation]
                metrics.operation_name = operation
                metrics.total_operations += 1
                
                if result.success:
                    metrics.successful_operations += 1
                else:
                    metrics.failed_operations += 1
                
                metrics.total_time_seconds += result.processing_time
                metrics.average_time_seconds = metrics.total_time_seconds / metrics.total_operations
                metrics.min_time_seconds = min(metrics.min_time_seconds, result.processing_time)
                metrics.max_time_seconds = max(metrics.max_time_seconds, result.processing_time)
                
                if result.cache_hit:
                    metrics.cache_hits += 1
                else:
                    metrics.cache_misses += 1
            
            # Add to processing history if detailed tracking enabled
            if self.enable_detailed_tracking:
                history_entry = {
                    'timestamp': result.timestamp,
                    'filepath': result.filepath,
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'operations': result.operations_performed,
                    'cache_hit': result.cache_hit,
                    'online_enhancement': result.online_enhancement
                }
                self.processing_history.append(history_entry)
                
                # Limit history size
                max_history = 1000  # Keep last 1000 entries
                if len(self.processing_history) > max_history:
                    self.processing_history = self.processing_history[-max_history:]
            
            # Update collection analytics if metadata available
            if result.metadata:
                self._update_collection_analytics(result.metadata)
    
    def record_batch_result(self, batch_result: BatchProcessingResult):
        """
        Record batch processing results
        
        Args:
            batch_result: BatchProcessingResult to analyze
        """
        with self._lock:
            # Record individual results
            for result in batch_result.results:
                self.record_processing_result(result)
            
            # Record daily statistics
            date_key = time.strftime('%Y-%m-%d')
            if date_key not in self.daily_stats:
                self.daily_stats[date_key] = {
                    'files_processed': 0,
                    'successful': 0,
                    'failed': 0,
                    'total_time': 0.0,
                    'cache_hits': 0,
                    'online_enhancements': 0
                }
            
            daily = self.daily_stats[date_key]
            daily['files_processed'] += batch_result.total_files
            daily['successful'] += batch_result.successful
            daily['failed'] += batch_result.failed
            daily['total_time'] += batch_result.total_time
            daily['cache_hits'] += batch_result.cache_hits
            daily['online_enhancements'] += batch_result.online_enhancements
    
    def analyze_collection(self, metadata_list: List[TrackMetadata]) -> CollectionAnalytics:
        """
        Perform comprehensive analysis of a music collection
        
        Args:
            metadata_list: List of TrackMetadata objects to analyze
            
        Returns:
            CollectionAnalytics with comprehensive statistics
        """
        analytics = CollectionAnalytics()
        
        # Basic statistics
        analytics.total_tracks = len(metadata_list)
        
        if not metadata_list:
            return analytics
        
        # Collect data for analysis
        bpms = []
        keys = []
        genres = []
        years = []
        formats = []
        qualities = []
        durations = []
        file_sizes = []
        
        for metadata in metadata_list:
            if metadata.duration:
                durations.append(metadata.duration)
            if metadata.filesize:
                file_sizes.append(metadata.filesize)
            if metadata.bpm:
                bpms.append(metadata.bpm)
            if metadata.musical_key:
                keys.append(metadata.musical_key)
            if metadata.genre:
                genres.append(metadata.genre)
            if metadata.year:
                years.append(metadata.year)
            if metadata.format:
                formats.append(metadata.format)
            if metadata.quality_score:
                # Categorize quality scores
                if metadata.quality_score >= 8:
                    qualities.append("Excellent")
                elif metadata.quality_score >= 6:
                    qualities.append("Good")
                elif metadata.quality_score >= 4:
                    qualities.append("Fair")
                else:
                    qualities.append("Poor")
        
        # Calculate distributions
        analytics.genre_distribution = dict(Counter(genres).most_common(20))
        analytics.key_distribution = dict(Counter(keys).most_common(12))
        analytics.year_distribution = dict(Counter(years).most_common(20))
        analytics.format_distribution = dict(Counter(formats))
        analytics.quality_distribution = dict(Counter(qualities))
        
        # BPM distribution (group by ranges)
        bpm_ranges = {}
        for bpm in bpms:
            range_key = f"{int(bpm // 10) * 10}-{int(bpm // 10) * 10 + 9}"
            bpm_ranges[range_key] = bpm_ranges.get(range_key, 0) + 1
        analytics.bpm_distribution = bpm_ranges
        
        # Calculate averages and totals
        if durations:
            analytics.total_duration_hours = sum(durations) / 3600
        if file_sizes:
            analytics.total_size_gb = sum(file_sizes) / (1024**3)
        if bpms:
            analytics.average_bpm = statistics.mean(bpms)
        
        # Most common values
        if genres:
            analytics.most_common_genre = Counter(genres).most_common(1)[0][0]
        if keys:
            analytics.most_common_key = Counter(keys).most_common(1)[0][0]
        
        # Decade breakdown
        decades = [f"{(year // 10) * 10}s" for year in years if year]
        analytics.decade_breakdown = dict(Counter(decades).most_common(10))
        
        return analytics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for unified interface compatibility"""
        return self.get_session_stats()
        
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        with self._lock:
            session_duration = time.time() - self.session_start_time
            
            stats = {
                'session_duration_seconds': session_duration,
                'session_duration_formatted': self._format_duration(session_duration),
                **self.session_summary
            }
            
            # Calculate rates
            if stats['files_processed'] > 0:
                stats['success_rate'] = stats['successful_processes'] / stats['files_processed'] * 100
                stats['average_processing_time'] = stats['total_processing_time'] / stats['files_processed']
                stats['files_per_hour'] = stats['files_processed'] / (session_duration / 3600)
            else:
                stats['success_rate'] = 0
                stats['average_processing_time'] = 0
                stats['files_per_hour'] = 0
            
            return stats
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        with self._lock:
            report = {
                'session_stats': self.get_session_stats(),
                'operation_metrics': {},
                'collection_analytics': self.collection_analytics,
                'trends': self._analyze_trends(),
                'recommendations': self._generate_recommendations()
            }
            
            # Convert performance metrics to dict
            for operation, metrics in self.performance_metrics.items():
                report['operation_metrics'][operation] = {
                    'total_operations': metrics.total_operations,
                    'success_rate': metrics.success_rate,
                    'average_time': metrics.average_time_seconds,
                    'cache_hit_rate': metrics.cache_hit_rate
                }
            
            return report
    
    def export_analytics(self, format: str = 'json', filepath: Optional[str] = None) -> str:
        """
        Export analytics data to file
        
        Args:
            format: Export format ('json', 'csv', 'html')
            filepath: Output file path (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if not filepath:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = f"dj_music_cleaner_analytics_{timestamp}.{format}"
        
        try:
            report = self.get_performance_report()
            
            if format == 'json':
                self._export_json(report, filepath)
            elif format == 'csv':
                self._export_csv(report, filepath)
            elif format == 'html':
                self._export_html(report, filepath)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            return filepath
            
        except Exception as e:
            raise Exception(f"Analytics export failed: {str(e)}")
    
    def track_file_details(self, file_path: str, original_metadata: Dict[str, Any], 
                       cleaned_metadata: Dict[str, Any], changes: List[str],
                       is_high_quality: bool = False, bitrate: int = 0,
                       sample_rate: float = 0.0, enhanced: bool = False,
                       output_path: Optional[str] = None) -> None:
        """Track detailed information about a processed file
        
        Args:
            file_path: Path to the original file
            original_metadata: Original ID3 metadata before processing
            cleaned_metadata: Updated ID3 metadata after processing
            changes: List of changes made to the file
            is_high_quality: Whether the file is considered high quality
            bitrate: Audio bitrate in kbps
            sample_rate: Audio sample rate in kHz
            enhanced: Whether online enhancement was applied
            output_path: Path where the file was saved after processing
        """
        with self._lock:
            filename = os.path.basename(file_path)
            self.file_details[filename] = {
                "input_path": file_path,
                "original_metadata": original_metadata,
                "cleaned_metadata": cleaned_metadata,
                "changes": changes,
                "enhanced": enhanced,
                "is_high_quality": is_high_quality,
                "bitrate": bitrate,
                "sample_rate": sample_rate,
                "last_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "output_path": output_path
            }

    def export_report(self, format: str = 'html', filepath: Optional[str] = None) -> str:
        """Export analytics report in specified format
        
        Args:
            format: Report format ('html', 'json', or 'csv')
            filepath: Custom output filepath (optional)
            
        Returns:
            Path to exported file
        """
        if not filepath:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = f"dj_music_cleaner_analytics_{timestamp}.{format}"
        
        try:
            # For standard reports
            report = self.get_performance_report()
            
            if format == 'json':
                # Check if we should use detailed file report
                if self.file_details and self.config.get('advanced_reporting', True):
                    self._export_detailed_json(filepath)
                else:
                    self._export_json(report, filepath)
            elif format == 'csv':
                self._export_csv(report, filepath)
            elif format == 'html':
                self._export_html(report, filepath)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            return filepath
            
        except Exception as e:
            raise Exception(f"Analytics export failed: {str(e)}")
            
    def _export_detailed_json(self, filepath: str) -> None:
        """Export detailed file-by-file JSON report
        
        Args:
            filepath: Path to save the JSON report
        """
        with self._lock:
            try:
                report = {
                    "files": self.file_details,
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
                
                # Serialize with proper handling of complex objects
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str)
            except Exception as e:
                print(f"Error generating detailed JSON report: {str(e)}")
                raise
    
    def reset_session_stats(self):
        """Reset session statistics"""
        with self._lock:
            self.session_start_time = time.time()
            self.session_summary = {
                'files_processed': 0,
                'successful_processes': 0,
                'failed_processes': 0,
                'total_processing_time': 0.0,
                'cache_hits': 0,
                'online_enhancements': 0,
                'errors_by_type': defaultdict(int),
                'warnings_by_type': defaultdict(int)
            }
            self.performance_metrics.clear()
            self.processing_history.clear()
    
    # Private helper methods
    
    def _update_collection_analytics(self, metadata: TrackMetadata):
        """Update collection analytics with new metadata"""
        self.collection_analytics.total_tracks += 1
        
        if metadata.duration:
            self.collection_analytics.total_duration_hours += metadata.duration / 3600
        
        if metadata.filesize:
            self.collection_analytics.total_size_gb += metadata.filesize / (1024**3)
        
        if metadata.genre:
            self.collection_analytics.genre_distribution[metadata.genre] = \
                self.collection_analytics.genre_distribution.get(metadata.genre, 0) + 1
        
        if metadata.bpm:
            range_key = f"{int(metadata.bpm // 10) * 10}-{int(metadata.bpm // 10) * 10 + 9}"
            self.collection_analytics.bpm_distribution[range_key] = \
                self.collection_analytics.bpm_distribution.get(range_key, 0) + 1
        
        if metadata.musical_key:
            self.collection_analytics.key_distribution[metadata.musical_key] = \
                self.collection_analytics.key_distribution.get(metadata.musical_key, 0) + 1
    
    def _classify_error(self, error: str) -> str:
        """Classify error message into category"""
        error_lower = error.lower()
        
        if 'file not found' in error_lower or 'does not exist' in error_lower:
            return 'file_not_found'
        elif 'permission' in error_lower:
            return 'permission_denied'
        elif 'corrupt' in error_lower or 'invalid' in error_lower:
            return 'file_corruption'
        elif 'metadata' in error_lower:
            return 'metadata_error'
        elif 'audio' in error_lower or 'analysis' in error_lower:
            return 'audio_analysis_error'
        elif 'network' in error_lower or 'online' in error_lower:
            return 'network_error'
        else:
            return 'other'
    
    def _classify_warning(self, warning: str) -> str:
        """Classify warning message into category"""
        warning_lower = warning.lower()
        
        if 'cache' in warning_lower:
            return 'cache_warning'
        elif 'quality' in warning_lower:
            return 'quality_warning'
        elif 'metadata' in warning_lower:
            return 'metadata_warning'
        elif 'audio' in warning_lower:
            return 'audio_warning'
        else:
            return 'other'
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in processing data"""
        trends = {
            'daily_volume': {},
            'success_rate_trend': {},
            'performance_trend': {}
        }
        
        # Analyze daily statistics
        sorted_days = sorted(self.daily_stats.keys())
        
        for day in sorted_days[-7:]:  # Last 7 days
            if day in self.daily_stats:
                stats = self.daily_stats[day]
                trends['daily_volume'][day] = stats['files_processed']
                trends['success_rate_trend'][day] = (
                    stats['successful'] / stats['files_processed'] * 100
                    if stats['files_processed'] > 0 else 0
                )
                trends['performance_trend'][day] = (
                    stats['total_time'] / stats['files_processed']
                    if stats['files_processed'] > 0 else 0
                )
        
        return trends
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance and optimization recommendations"""
        recommendations = []
        
        # Cache performance recommendations
        session_stats = self.get_session_stats()
        cache_hit_rate = (session_stats['cache_hits'] / session_stats['files_processed'] * 100
                         if session_stats['files_processed'] > 0 else 0)
        
        if cache_hit_rate < 20:
            recommendations.append("Low cache hit rate - consider increasing cache timeout or processing newer files")
        
        # Processing speed recommendations
        if session_stats.get('average_processing_time', 0) > 5.0:
            recommendations.append("Slow processing detected - consider increasing worker count or disabling expensive analysis features")
        
        # Error rate recommendations
        if session_stats.get('success_rate', 100) < 90:
            recommendations.append("High error rate detected - check file integrity and permissions")
        
        # Collection recommendations
        if self.collection_analytics.total_tracks > 0:
            genre_count = len(self.collection_analytics.genre_distribution)
            if genre_count > 50:
                recommendations.append("Many unique genres detected - consider consolidating similar genres")
        
        return recommendations
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _export_json(self, report: Dict[str, Any], filepath: str):
        """Export report as JSON"""
        # Convert defaultdicts and other non-serializable objects
        serializable_report = json.loads(json.dumps(report, default=str))
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
    
    def _export_csv(self, report: Dict[str, Any], filepath: str):
        """Export report as CSV (simplified version)"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Session stats
            writer.writerow(['Session Statistics'])
            writer.writerow(['Metric', 'Value'])
            for key, value in report['session_stats'].items():
                writer.writerow([key, value])
            
            writer.writerow([])  # Empty row
            
            # Operation metrics
            writer.writerow(['Operation Metrics'])
            writer.writerow(['Operation', 'Total', 'Success Rate', 'Avg Time', 'Cache Hit Rate'])
            for operation, metrics in report['operation_metrics'].items():
                writer.writerow([
                    operation,
                    metrics['total_operations'],
                    f"{metrics['success_rate']:.1f}%",
                    f"{metrics['average_time']:.3f}s",
                    f"{metrics['cache_hit_rate']:.1f}%"
                ])
    
    def _export_html(self, report: Dict[str, Any], filepath: str):
        """Export report as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DJ Music Cleaner Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .recommendations {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>DJ Music Cleaner Analytics Report</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Session Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for key, value in report['session_stats'].items():
            html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Recommendations</h2>
            <div class="recommendations">
        """
        
        for rec in report['recommendations']:
            html_content += f"<p>• {rec}</p>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)


__all__ = ['AnalyticsService', 'CollectionAnalytics', 'PerformanceMetrics']