"""
Unified Cache Service

Consolidates caching functionality from both implementations:
1. Multi-layer caching (memory + SQLite) from evolved version
2. Comprehensive metadata caching from original
3. Performance analytics and cache optimization
"""

import os
import sqlite3
import json
import time
import hashlib
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import gzip

from ..core.models import TrackMetadata, ProcessingOptions
from ..core.exceptions import CacheError
from ..utils.filesystem import ensure_directory


class UnifiedCacheService:
    """
    Unified caching service with multi-layer caching and analytics
    
    Features:
    - Memory cache for hot data
    - SQLite persistent cache
    - Compressed storage for large objects
    - Cache analytics and performance tracking
    - Automatic cleanup and optimization
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified cache service"""
        self.config = config or {}
        
        # Configuration
        self.cache_dir = self.config.get('cache_dir', os.path.expanduser('~/.dj_music_cleaner'))
        self.db_name = self.config.get('db_name', 'dj_music_cleaner_unified.db')
        self.memory_limit = self.config.get('memory_limit', 100)  # MB
        self.default_timeout = self.config.get('default_timeout_days', 30)
        self.enable_compression = self.config.get('enable_compression', True)
        
        # Cache paths
        ensure_directory(self.cache_dir)
        self.db_path = os.path.join(self.cache_dir, self.db_name)
        
        # Memory cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_access_times: Dict[str, float] = {}
        self._memory_size_estimate = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'writes': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Initialize database
        self._init_database()
        
        # Load hot cache items into memory
        self._load_hot_cache()
    
    def _init_database(self):
        """Initialize SQLite database with unified schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Metadata cache table (enhanced from both versions)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metadata_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filepath TEXT UNIQUE NOT NULL,
                        file_hash TEXT,
                        file_size INTEGER,
                        modified_time REAL,
                        cached_time REAL NOT NULL,
                        expires_time REAL,
                        cache_version INTEGER DEFAULT 1,
                        
                        -- Basic metadata
                        title TEXT,
                        artist TEXT,
                        album TEXT,
                        year INTEGER,
                        genre TEXT,
                        track_number INTEGER,
                        
                        -- Extended metadata
                        album_artist TEXT,
                        composer TEXT,
                        publisher TEXT,
                        isrc TEXT,
                        catalog_number TEXT,
                        label TEXT,
                        
                        -- Audio analysis
                        bpm REAL,
                        bpm_confidence REAL,
                        musical_key TEXT,
                        key_confidence REAL,
                        camelot_key TEXT,
                        energy_level REAL,
                        
                        -- Quality metrics
                        quality_score REAL,
                        dynamic_range REAL,
                        peak_level REAL,
                        rms_level REAL,
                        
                        -- Processing metadata
                        processing_time REAL,
                        online_enhanced BOOLEAN,
                        validation_errors TEXT,
                        
                        -- Binary data (compressed)
                        cue_points BLOB,
                        beat_grid BLOB,
                        extended_analysis BLOB,
                        
                        UNIQUE(filepath)
                    )
                ''')
                
                # Performance analytics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        operation TEXT NOT NULL,
                        filepath TEXT,
                        hit_type TEXT,  -- 'memory', 'disk', 'miss'
                        response_time REAL,
                        data_size INTEGER
                    )
                ''')
                
                # Cache configuration table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_config (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_time REAL
                    )
                ''')
                
                # Create indices for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_filepath ON metadata_cache(filepath)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON metadata_cache(file_hash)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cached_time ON metadata_cache(cached_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires_time ON metadata_cache(expires_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON cache_analytics(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            raise CacheError(f"Failed to initialize cache database: {str(e)}")
    
    def get_metadata(self, filepath: str, max_age_days: Optional[int] = None) -> Optional[TrackMetadata]:
        """
        Retrieve cached metadata for a file
        
        Args:
            filepath: Path to the audio file
            max_age_days: Maximum age of cached data in days
            
        Returns:
            TrackMetadata object if found and valid, None otherwise
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self.stats['total_requests'] += 1
                
                # Normalize path
                filepath = os.path.abspath(filepath)
                cache_key = self._get_cache_key(filepath)
                
                # Check memory cache first
                if cache_key in self._memory_cache:
                    cached_data = self._memory_cache[cache_key]
                    
                    # Check if still valid
                    if self._is_cache_valid(cached_data, filepath, max_age_days):
                        self._memory_access_times[cache_key] = time.time()
                        self.stats['hits'] += 1
                        self.stats['memory_hits'] += 1
                        
                        self._record_analytics('get', filepath, 'memory', time.time() - start_time)
                        return self._deserialize_metadata(cached_data['metadata'])
                    else:
                        # Remove expired entry
                        del self._memory_cache[cache_key]
                        if cache_key in self._memory_access_times:
                            del self._memory_access_times[cache_key]
                
                # Check disk cache
                disk_data = self._get_from_disk(filepath, max_age_days)
                if disk_data:
                    # Load into memory for faster future access
                    self._memory_cache[cache_key] = disk_data
                    self._memory_access_times[cache_key] = time.time()
                    self._manage_memory_size()
                    
                    self.stats['hits'] += 1
                    self.stats['disk_hits'] += 1
                    
                    self._record_analytics('get', filepath, 'disk', time.time() - start_time)
                    return self._deserialize_metadata(disk_data['metadata'])
                
                # Cache miss
                self.stats['misses'] += 1
                self._record_analytics('get', filepath, 'miss', time.time() - start_time)
                return None
                
        except Exception as e:
            raise CacheError(f"Failed to get cached metadata: {str(e)}", filepath=filepath)
    
    def set_metadata(self, filepath: str, metadata: TrackMetadata, 
                    timeout_days: Optional[int] = None) -> bool:
        """
        Store metadata in cache
        
        Args:
            filepath: Path to the audio file
            metadata: TrackMetadata object to cache
            timeout_days: Cache timeout in days (uses default if None)
            
        Returns:
            True if successfully cached
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self.stats['writes'] += 1
                
                # Normalize path
                filepath = os.path.abspath(filepath)
                cache_key = self._get_cache_key(filepath)
                
                # Get file info for validation
                file_info = self._get_file_info(filepath)
                if not file_info:
                    return False
                
                # Prepare cache data
                timeout = timeout_days or self.default_timeout
                expires_time = time.time() + (timeout * 24 * 3600)
                
                cache_data = {
                    'metadata': self._serialize_metadata(metadata),
                    'filepath': filepath,
                    'file_hash': file_info['hash'],
                    'file_size': file_info['size'],
                    'modified_time': file_info['modified_time'],
                    'cached_time': time.time(),
                    'expires_time': expires_time
                }
                
                # Store in memory cache
                self._memory_cache[cache_key] = cache_data
                self._memory_access_times[cache_key] = time.time()
                self._manage_memory_size()
                
                # Store in disk cache
                self._set_to_disk(filepath, metadata, cache_data)
                
                self._record_analytics('set', filepath, 'write', time.time() - start_time, 
                                     len(str(metadata)))
                
                return True
                
        except Exception as e:
            raise CacheError(f"Failed to cache metadata: {str(e)}", filepath=filepath)
    
    def invalidate(self, filepath: str) -> bool:
        """
        Invalidate cached data for a file
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            True if invalidated
        """
        try:
            with self._lock:
                filepath = os.path.abspath(filepath)
                cache_key = self._get_cache_key(filepath)
                
                # Remove from memory
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                if cache_key in self._memory_access_times:
                    del self._memory_access_times[cache_key]
                
                # Remove from disk
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM metadata_cache WHERE filepath = ?', (filepath,))
                    conn.commit()
                
                return True
                
        except Exception as e:
            raise CacheError(f"Failed to invalidate cache: {str(e)}", filepath=filepath)
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries
        
        Returns:
            Number of entries removed
        """
        try:
            with self._lock:
                current_time = time.time()
                removed_count = 0
                
                # Clean memory cache
                expired_keys = []
                for key, data in self._memory_cache.items():
                    if data.get('expires_time', 0) < current_time:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._memory_cache[key]
                    if key in self._memory_access_times:
                        del self._memory_access_times[key]
                    removed_count += 1
                
                # Clean disk cache
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM metadata_cache WHERE expires_time < ?', (current_time,))
                    removed_count += cursor.rowcount
                    conn.commit()
                
                return removed_count
                
        except Exception as e:
            raise CacheError(f"Failed to cleanup expired cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            with self._lock:
                # Memory cache stats
                memory_count = len(self._memory_cache)
                memory_size_mb = self._memory_size_estimate / (1024 * 1024)
                
                # Disk cache stats
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT COUNT(*) FROM metadata_cache')
                    disk_count = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM metadata_cache WHERE expires_time < ?', 
                                 (time.time(),))
                    expired_count = cursor.fetchone()[0]
                    
                    # Database size
                    db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                
                # Hit rates
                total_requests = self.stats['total_requests']
                hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
                memory_hit_rate = (self.stats['memory_hits'] / total_requests * 100) if total_requests > 0 else 0
                
                return {
                    'memory_cache': {
                        'entries': memory_count,
                        'size_mb': round(memory_size_mb, 2),
                        'limit_mb': self.memory_limit
                    },
                    'disk_cache': {
                        'entries': disk_count,
                        'expired_entries': expired_count,
                        'size_mb': round(db_size_mb, 2)
                    },
                    'performance': {
                        'hit_rate': round(hit_rate, 1),
                        'memory_hit_rate': round(memory_hit_rate, 1),
                        'total_requests': total_requests,
                        'hits': self.stats['hits'],
                        'misses': self.stats['misses'],
                        'writes': self.stats['writes'],
                        'evictions': self.stats['evictions']
                    },
                    'configuration': {
                        'cache_dir': self.cache_dir,
                        'default_timeout_days': self.default_timeout,
                        'compression_enabled': self.enable_compression
                    }
                }
                
        except Exception as e:
            raise CacheError(f"Failed to get cache stats: {str(e)}")
    
    def optimize_cache(self) -> Dict[str, int]:
        """
        Optimize cache performance by cleaning up and reorganizing
        
        Returns:
            Dictionary with optimization results
        """
        try:
            results = {
                'expired_removed': 0,
                'orphaned_removed': 0,
                'memory_evictions': 0,
                'database_optimized': False
            }
            
            # Clean up expired entries
            results['expired_removed'] = self.cleanup_expired()
            
            # Remove orphaned entries (files that no longer exist)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT filepath FROM metadata_cache')
                
                orphaned = []
                for row in cursor.fetchall():
                    filepath = row[0]
                    if not os.path.exists(filepath):
                        orphaned.append(filepath)
                
                for filepath in orphaned:
                    cursor.execute('DELETE FROM metadata_cache WHERE filepath = ?', (filepath,))
                    results['orphaned_removed'] += 1
                
                conn.commit()
            
            # Optimize memory cache
            if self._memory_size_estimate > self.memory_limit * 1024 * 1024 * 0.8:
                results['memory_evictions'] = self._evict_lru_memory()
            
            # Optimize database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('VACUUM')
                conn.execute('ANALYZE')
                results['database_optimized'] = True
            
            return results
            
        except Exception as e:
            raise CacheError(f"Failed to optimize cache: {str(e)}")
    
    # Private helper methods
    
    def _get_cache_key(self, filepath: str) -> str:
        """Generate cache key for filepath"""
        return hashlib.md5(filepath.encode()).hexdigest()
    
    def _get_file_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Get file information for cache validation"""
        try:
            if not os.path.exists(filepath):
                return None
            
            stat = os.stat(filepath)
            
            # Calculate file hash (first 8KB + last 8KB for speed)
            file_hash = ""
            try:
                with open(filepath, 'rb') as f:
                    first_chunk = f.read(8192)
                    f.seek(-8192, 2)  # Seek to 8KB from end
                    last_chunk = f.read(8192)
                    
                    hasher = hashlib.md5()
                    hasher.update(first_chunk)
                    hasher.update(last_chunk)
                    file_hash = hasher.hexdigest()
            except:
                pass
            
            return {
                'size': stat.st_size,
                'modified_time': stat.st_mtime,
                'hash': file_hash
            }
            
        except:
            return None
    
    def _is_cache_valid(self, cached_data: Dict[str, Any], filepath: str, 
                       max_age_days: Optional[int] = None) -> bool:
        """Check if cached data is still valid"""
        try:
            current_time = time.time()
            
            # Check expiration
            expires_time = cached_data.get('expires_time', 0)
            if max_age_days:
                max_age_seconds = max_age_days * 24 * 3600
                oldest_allowed = current_time - max_age_seconds
                if cached_data.get('cached_time', 0) < oldest_allowed:
                    return False
            
            if expires_time < current_time:
                return False
            
            # Check file hasn't changed
            file_info = self._get_file_info(filepath)
            if not file_info:
                return False
            
            if (file_info['modified_time'] != cached_data.get('modified_time') or
                file_info['size'] != cached_data.get('file_size') or
                (file_info['hash'] and file_info['hash'] != cached_data.get('file_hash'))):
                return False
            
            return True
            
        except:
            return False
    
    def _get_from_disk(self, filepath: str, max_age_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieve data from disk cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get cache data
                cursor.execute('''
                    SELECT * FROM metadata_cache WHERE filepath = ?
                ''', (filepath,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Convert row to dict
                columns = [desc[0] for desc in cursor.description]
                cache_data = dict(zip(columns, row))
                
                # Reconstruct full cache data structure
                full_data = {
                    'metadata': self._deserialize_db_row(cache_data),
                    'filepath': cache_data['filepath'],
                    'file_hash': cache_data['file_hash'],
                    'file_size': cache_data['file_size'],
                    'modified_time': cache_data['modified_time'],
                    'cached_time': cache_data['cached_time'],
                    'expires_time': cache_data['expires_time']
                }
                
                # Validate cache
                if self._is_cache_valid(full_data, filepath, max_age_days):
                    return full_data
                else:
                    # Remove invalid entry
                    cursor.execute('DELETE FROM metadata_cache WHERE filepath = ?', (filepath,))
                    conn.commit()
                    return None
                
        except Exception:
            return None
    
    def _set_to_disk(self, filepath: str, metadata: TrackMetadata, cache_data: Dict[str, Any]):
        """Store data to disk cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare data for database
                db_data = self._prepare_for_db(metadata, cache_data)
                
                # Insert or replace
                cursor.execute('''
                    INSERT OR REPLACE INTO metadata_cache (
                        filepath, file_hash, file_size, modified_time, cached_time, expires_time,
                        title, artist, album, year, genre, track_number,
                        album_artist, composer, publisher, isrc, catalog_number, label,
                        bpm, bpm_confidence, musical_key, key_confidence, camelot_key, energy_level,
                        quality_score, dynamic_range, peak_level, rms_level,
                        processing_time, online_enhanced, validation_errors,
                        cue_points, beat_grid, extended_analysis
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', db_data)
                
                conn.commit()
                
        except Exception as e:
            raise CacheError(f"Failed to store to disk cache: {str(e)}")
    
    def _serialize_metadata(self, metadata: TrackMetadata) -> Dict[str, Any]:
        """Serialize metadata for storage"""
        return metadata.to_dict()
    
    def _deserialize_metadata(self, data: Dict[str, Any]) -> TrackMetadata:
        """Deserialize metadata from storage"""
        return TrackMetadata.from_dict(data)
    
    def _deserialize_db_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert database row back to metadata dict"""
        # Extract binary data
        cue_points = []
        if row_data.get('cue_points'):
            try:
                if self.enable_compression:
                    cue_points = pickle.loads(gzip.decompress(row_data['cue_points']))
                else:
                    cue_points = json.loads(row_data['cue_points'])
            except:
                pass
        
        beat_grid = None
        if row_data.get('beat_grid'):
            try:
                if self.enable_compression:
                    beat_grid = pickle.loads(gzip.decompress(row_data['beat_grid']))
                else:
                    beat_grid = json.loads(row_data['beat_grid'])
            except:
                pass
        
        # Build metadata dict
        return {
            'filepath': row_data.get('filepath', ''),
            'filename': os.path.basename(row_data.get('filepath', '')),
            'title': row_data.get('title', ''),
            'artist': row_data.get('artist', ''),
            'album': row_data.get('album', ''),
            'year': row_data.get('year'),
            'genre': row_data.get('genre', ''),
            'track_number': row_data.get('track_number'),
            'album_artist': row_data.get('album_artist', ''),
            'composer': row_data.get('composer', ''),
            'publisher': row_data.get('publisher', ''),
            'isrc': row_data.get('isrc', ''),
            'catalog_number': row_data.get('catalog_number', ''),
            'label': row_data.get('label', ''),
            'bpm': row_data.get('bpm'),
            'bpm_confidence': row_data.get('bpm_confidence', 0.0),
            'musical_key': row_data.get('musical_key', ''),
            'key_confidence': row_data.get('key_confidence', 0.0),
            'camelot_key': row_data.get('camelot_key', ''),
            'energy_level': row_data.get('energy_level'),
            'quality_score': row_data.get('quality_score', 0.0),
            'dynamic_range': row_data.get('dynamic_range'),
            'peak_level': row_data.get('peak_level'),
            'rms_level': row_data.get('rms_level'),
            'processing_time': row_data.get('processing_time', 0.0),
            'online_enhanced': bool(row_data.get('online_enhanced', False)),
            'validation_errors': json.loads(row_data.get('validation_errors', '[]')),
            'cue_points': cue_points,
            'beat_grid': beat_grid,
        }
    
    def _prepare_for_db(self, metadata: TrackMetadata, cache_data: Dict[str, Any]) -> tuple:
        """Prepare data for database storage"""
        # Serialize complex objects
        cue_points_data = None
        if metadata.cue_points:
            if self.enable_compression:
                cue_points_data = gzip.compress(pickle.dumps(metadata.cue_points))
            else:
                cue_points_data = json.dumps(metadata.cue_points).encode()
        
        beat_grid_data = None
        if metadata.beat_grid:
            if self.enable_compression:
                beat_grid_data = gzip.compress(pickle.dumps(metadata.beat_grid))
            else:
                beat_grid_data = json.dumps(metadata.beat_grid).encode()
        
        extended_analysis = None  # For future use
        
        return (
            cache_data['filepath'],
            cache_data.get('file_hash'),
            cache_data.get('file_size'),
            cache_data.get('modified_time'),
            cache_data['cached_time'],
            cache_data['expires_time'],
            
            metadata.title,
            metadata.artist,
            metadata.album,
            metadata.year,
            metadata.genre,
            metadata.track_number,
            
            metadata.album_artist,
            metadata.composer,
            metadata.publisher,
            metadata.isrc,
            metadata.catalog_number,
            metadata.label,
            
            metadata.bpm,
            metadata.bpm_confidence,
            metadata.musical_key,
            metadata.key_confidence,
            metadata.camelot_key,
            metadata.energy_level,
            
            metadata.quality_score,
            metadata.dynamic_range,
            metadata.peak_level,
            metadata.rms_level,
            
            metadata.processing_time,
            metadata.online_enhanced,
            json.dumps(metadata.validation_errors),
            
            cue_points_data,
            beat_grid_data,
            extended_analysis
        )
    
    def _manage_memory_size(self):
        """Manage memory cache size and evict if needed"""
        # Estimate memory usage (rough calculation)
        estimated_size = len(str(self._memory_cache))
        self._memory_size_estimate = estimated_size
        
        max_size = self.memory_limit * 1024 * 1024  # Convert MB to bytes
        
        if estimated_size > max_size:
            self._evict_lru_memory()
    
    def _evict_lru_memory(self) -> int:
        """Evict least recently used items from memory cache"""
        if not self._memory_cache:
            return 0
        
        # Sort by access time (oldest first)
        sorted_items = sorted(self._memory_access_times.items(), key=lambda x: x[1])
        
        evicted = 0
        target_size = self.memory_limit * 1024 * 1024 * 0.7  # Target 70% of limit
        
        for cache_key, _ in sorted_items:
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
                del self._memory_access_times[cache_key]
                evicted += 1
                self.stats['evictions'] += 1
                
                # Re-estimate size
                self._memory_size_estimate = len(str(self._memory_cache))
                if self._memory_size_estimate <= target_size:
                    break
        
        return evicted
    
    def _record_analytics(self, operation: str, filepath: str, hit_type: str, 
                         response_time: float, data_size: int = 0):
        """Record cache analytics for performance monitoring"""
        try:
            # Only record analytics occasionally to avoid overhead
            if self.stats['total_requests'] % 100 == 0:  # Every 100 requests
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO cache_analytics 
                        (timestamp, operation, filepath, hit_type, response_time, data_size)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (time.time(), operation, filepath, hit_type, response_time, data_size))
                    conn.commit()
        except:
            pass  # Don't fail on analytics errors
    
    def _load_hot_cache(self):
        """Load frequently accessed items into memory cache on startup"""
        try:
            # Load recent and frequently accessed items
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recently cached items
                cursor.execute('''
                    SELECT filepath FROM metadata_cache 
                    WHERE cached_time > ? 
                    ORDER BY cached_time DESC 
                    LIMIT 50
                ''', (time.time() - 24 * 3600,))  # Last 24 hours
                
                for row in cursor.fetchall():
                    filepath = row[0]
                    # Load into memory (will use _get_from_disk)
                    self.get_metadata(filepath)
                    
        except:
            pass  # Don't fail on hot cache loading


__all__ = ['UnifiedCacheService']