"""
Export Services

Provides comprehensive export capabilities for various DJ software formats and platforms.
Handles playlist exports, metadata exports, cue point exports, and collection syncing.
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import urllib.parse

from ...core.models import TrackMetadata
from ...core.exceptions import ExportError
from .cue_detection import CuePoint
from .beatgrid import BeatGrid
from .energy_calibration import EnergyProfile


@dataclass
class ExportFormat:
    """Export format specification"""
    name: str
    extension: str
    supports_metadata: bool
    supports_cues: bool
    supports_beatgrid: bool
    supports_playlists: bool
    description: str


@dataclass
class ExportJob:
    """Export job tracking"""
    job_id: str
    format_name: str
    output_path: str
    track_count: int
    progress: float
    status: str  # pending, running, completed, failed
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    errors: List[str] = field(default_factory=list)


class ExportService:
    """
    Comprehensive export service for DJ software integration
    
    Supported formats:
    - Rekordbox XML
    - Serato crates and database
    - Engine DJ database
    - Traktor NML
    - Virtual DJ XML
    - iTunes XML
    - M3U/M3U8 playlists
    - CSV spreadsheets
    - JSON data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the export service"""
        self.config = config or {}
        
        # Configuration
        self.preserve_folder_structure = self.config.get('preserve_folder_structure', True)
        self.convert_file_paths = self.config.get('convert_file_paths', True)
        self.include_analysis_data = self.config.get('include_analysis_data', True)
        self.backup_existing_files = self.config.get('backup_existing_files', True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Export jobs tracking
        self.active_jobs: Dict[str, ExportJob] = {}
        
        # Supported formats
        self.formats = {
            'rekordbox': ExportFormat(
                name='Rekordbox XML',
                extension='xml',
                supports_metadata=True,
                supports_cues=True,
                supports_beatgrid=True,
                supports_playlists=True,
                description='Pioneer Rekordbox collection format'
            ),
            'serato': ExportFormat(
                name='Serato Crate',
                extension='crate',
                supports_metadata=True,
                supports_cues=True,
                supports_beatgrid=False,
                supports_playlists=True,
                description='Serato DJ crate format'
            ),
            'engine': ExportFormat(
                name='Engine DJ',
                extension='db',
                supports_metadata=True,
                supports_cues=True,
                supports_beatgrid=True,
                supports_playlists=True,
                description='Denon Engine DJ database format'
            ),
            'traktor': ExportFormat(
                name='Traktor NML',
                extension='nml',
                supports_metadata=True,
                supports_cues=True,
                supports_beatgrid=True,
                supports_playlists=True,
                description='Native Instruments Traktor collection'
            ),
            'virtualdj': ExportFormat(
                name='Virtual DJ XML',
                extension='xml',
                supports_metadata=True,
                supports_cues=True,
                supports_beatgrid=False,
                supports_playlists=True,
                description='Virtual DJ database format'
            ),
            'm3u8': ExportFormat(
                name='M3U8 Playlist',
                extension='m3u8',
                supports_metadata=False,
                supports_cues=False,
                supports_beatgrid=False,
                supports_playlists=True,
                description='Extended M3U playlist format'
            ),
            'csv': ExportFormat(
                name='CSV Spreadsheet',
                extension='csv',
                supports_metadata=True,
                supports_cues=False,
                supports_beatgrid=False,
                supports_playlists=False,
                description='Comma-separated values for data analysis'
            ),
            'json': ExportFormat(
                name='JSON Data',
                extension='json',
                supports_metadata=True,
                supports_cues=True,
                supports_beatgrid=True,
                supports_playlists=True,
                description='Structured JSON format for developers'
            )
        }
        
        # Performance tracking
        self.stats = {
            'exports_completed': 0,
            'total_tracks_exported': 0,
            'export_formats_used': {},
            'total_export_time': 0.0
        }
    
    def get_supported_formats(self) -> List[ExportFormat]:
        """Get list of supported export formats"""
        return list(self.formats.values())
    
    def export_collection(self, tracks: List[Dict[str, Any]], 
                         format_name: str, 
                         output_path: str,
                         options: Optional[Dict[str, Any]] = None) -> str:
        """
        Export a collection of tracks to specified format
        
        Args:
            tracks: List of track dictionaries with metadata
            format_name: Target export format
            output_path: Output file path
            options: Export options
            
        Returns:
            Job ID for tracking export progress
        """
        if format_name not in self.formats:
            raise ExportError(f"Unsupported export format: {format_name}")
        
        options = options or {}
        job_id = f"export_{int(time.time())}_{format_name}"
        
        # Create export job
        job = ExportJob(
            job_id=job_id,
            format_name=format_name,
            output_path=output_path,
            track_count=len(tracks),
            progress=0.0,
            status='pending',
            created_at=time.time()
        )
        
        with self._lock:
            self.active_jobs[job_id] = job
        
        # Start export in background thread
        def run_export():
            self._execute_export(job, tracks, options)
        
        export_thread = threading.Thread(target=run_export, daemon=True)
        export_thread.start()
        
        return job_id
    
    def _execute_export(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Execute the export job"""
        try:
            job.status = 'running'
            job.started_at = time.time()
            
            format_info = self.formats[job.format_name]
            
            # Backup existing file if requested
            if self.backup_existing_files and os.path.exists(job.output_path):
                backup_path = f"{job.output_path}.backup_{int(time.time())}"
                import shutil
                shutil.copy2(job.output_path, backup_path)
            
            # Route to appropriate export method
            if job.format_name == 'rekordbox':
                self._export_rekordbox(job, tracks, options)
            elif job.format_name == 'serato':
                self._export_serato(job, tracks, options)
            elif job.format_name == 'engine':
                self._export_engine_dj(job, tracks, options)
            elif job.format_name == 'traktor':
                self._export_traktor(job, tracks, options)
            elif job.format_name == 'virtualdj':
                self._export_virtualdj(job, tracks, options)
            elif job.format_name == 'm3u8':
                self._export_m3u8(job, tracks, options)
            elif job.format_name == 'csv':
                self._export_csv(job, tracks, options)
            elif job.format_name == 'json':
                self._export_json(job, tracks, options)
            else:
                raise ExportError(f"Export method not implemented for {job.format_name}")
            
            job.status = 'completed'
            job.completed_at = time.time()
            job.progress = 100.0
            
            # Update statistics
            with self._lock:
                self.stats['exports_completed'] += 1
                self.stats['total_tracks_exported'] += job.track_count
                self.stats['export_formats_used'][job.format_name] = (
                    self.stats['export_formats_used'].get(job.format_name, 0) + 1
                )
                if job.completed_at and job.started_at:
                    self.stats['total_export_time'] += (job.completed_at - job.started_at)
                    
        except Exception as e:
            job.status = 'failed'
            job.errors.append(f"Export failed: {str(e)}")
            job.completed_at = time.time()
    
    def _export_rekordbox(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Export to Rekordbox XML format"""
        try:
            # Create XML structure
            root = ET.Element('DJ_PLAYLISTS', {'Version': '1.0.0'})
            
            # Add product info
            product = ET.SubElement(root, 'PRODUCT', {
                'Name': 'DJ Music Cleaner',
                'Version': '1.0.0',
                'Company': 'DJ Music Cleaner'
            })
            
            # Create collection
            collection = ET.SubElement(root, 'COLLECTION', {'Entries': str(len(tracks))})
            
            for i, track_data in enumerate(tracks):
                # Update progress
                job.progress = (i / len(tracks)) * 90  # Reserve 10% for file writing
                
                # Create track element
                track_attrs = {
                    'TrackID': str(i + 1),
                    'Name': track_data.get('title', ''),
                    'Artist': track_data.get('artist', ''),
                    'Album': track_data.get('album', ''),
                    'Genre': track_data.get('genre', ''),
                    'Kind': 'MP3 File',
                    'Size': str(track_data.get('filesize', 0)),
                    'TotalTime': str(int(track_data.get('duration', 0))),
                    'Year': str(track_data.get('year', 0)) if track_data.get('year') else '',
                    'AverageBpm': str(track_data.get('bpm', 0)) if track_data.get('bpm') else '',
                    'DateAdded': track_data.get('date_added', ''),
                    'BitRate': str(track_data.get('bitrate', 320)),
                    'SampleRate': str(track_data.get('sample_rate', 44100)),
                    'Location': f"file://localhost{urllib.parse.quote(track_data.get('filepath', ''))}",
                    'Tonality': track_data.get('musical_key', ''),
                    'Comments': track_data.get('comment', '')
                }
                
                track_elem = ET.SubElement(collection, 'TRACK', track_attrs)
                
                # Add cue points if available
                cue_points = track_data.get('cue_points', [])
                if isinstance(cue_points, list):
                    for j, cue in enumerate(cue_points):
                        if isinstance(cue, dict):
                            cue_attrs = {
                                'Name': cue.get('description', f'Cue {j+1}'),
                                'Type': '0',  # Hot cue
                                'Start': str(cue.get('position_seconds', 0)),
                                'Num': str(j)
                            }
                            ET.SubElement(track_elem, 'POSITION_MARK', cue_attrs)
            
            # Add playlists if specified
            playlists_data = options.get('playlists', [])
            if playlists_data:
                playlists_elem = ET.SubElement(root, 'PLAYLISTS')
                for playlist in playlists_data:
                    self._add_rekordbox_playlist(playlists_elem, playlist, tracks)
            
            # Write XML file
            job.progress = 95
            self._write_formatted_xml(root, job.output_path)
            
        except Exception as e:
            raise ExportError(f"Rekordbox export failed: {str(e)}")
    
    def _export_serato(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Export to Serato crate format"""
        try:
            # Serato uses a binary crate format, so we'll create a text-based version
            # that can be imported or converted
            
            crate_name = options.get('crate_name', 'DJ Music Cleaner Export')
            
            lines = []
            lines.append(f'[Crate]')
            lines.append(f'Name={crate_name}')
            lines.append(f'Tracks={len(tracks)}')
            lines.append('')
            
            for i, track_data in enumerate(tracks):
                job.progress = (i / len(tracks)) * 100
                
                lines.append(f'[Track{i+1}]')
                lines.append(f'Filename={track_data.get("filepath", "")}')
                lines.append(f'Title={track_data.get("title", "")}')
                lines.append(f'Artist={track_data.get("artist", "")}')
                lines.append(f'Album={track_data.get("album", "")}')
                lines.append(f'Genre={track_data.get("genre", "")}')
                
                if track_data.get('bpm'):
                    lines.append(f'BPM={track_data["bpm"]}')
                if track_data.get('musical_key'):
                    lines.append(f'Key={track_data["musical_key"]}')
                
                lines.append('')
            
            # Write crate file
            with open(job.output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            raise ExportError(f"Serato export failed: {str(e)}")
    
    def _export_engine_dj(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Export to Engine DJ format (simplified XML representation)"""
        try:
            root = ET.Element('EngineLibrary')
            
            for i, track_data in enumerate(tracks):
                job.progress = (i / len(tracks)) * 100
                
                track_elem = ET.SubElement(root, 'Track')
                track_elem.set('id', str(i + 1))
                
                # Basic metadata
                ET.SubElement(track_elem, 'FilePath').text = track_data.get('filepath', '')
                ET.SubElement(track_elem, 'Title').text = track_data.get('title', '')
                ET.SubElement(track_elem, 'Artist').text = track_data.get('artist', '')
                ET.SubElement(track_elem, 'Album').text = track_data.get('album', '')
                ET.SubElement(track_elem, 'Genre').text = track_data.get('genre', '')
                
                if track_data.get('bpm'):
                    ET.SubElement(track_elem, 'BPM').text = str(track_data['bpm'])
                if track_data.get('musical_key'):
                    ET.SubElement(track_elem, 'Key').text = track_data['musical_key']
                
                # Beat grid data if available
                beat_grid = track_data.get('beat_grid')
                if beat_grid and isinstance(beat_grid, dict):
                    grid_elem = ET.SubElement(track_elem, 'BeatGrid')
                    beats = beat_grid.get('beats', [])
                    for beat in beats[:100]:  # Limit to first 100 beats
                        if isinstance(beat, dict):
                            beat_elem = ET.SubElement(grid_elem, 'Beat')
                            beat_elem.set('position', str(beat.get('position_seconds', 0)))
                            beat_elem.set('bpm', str(beat.get('tempo_bpm', 120)))
            
            self._write_formatted_xml(root, job.output_path)
            
        except Exception as e:
            raise ExportError(f"Engine DJ export failed: {str(e)}")
    
    def _export_traktor(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Export to Traktor NML format"""
        try:
            root = ET.Element('NML', {'VERSION': '19'})
            
            # Create collection
            collection = ET.SubElement(root, 'COLLECTION', {'ENTRIES': str(len(tracks))})
            
            for i, track_data in enumerate(tracks):
                job.progress = (i / len(tracks)) * 100
                
                entry = ET.SubElement(collection, 'ENTRY', {
                    'MODIFIED_DATE': str(int(time.time())),
                    'MODIFIED_TIME': str(int(time.time())),
                    'AUDIO_ID': str(i + 1)
                })
                
                # Location
                location = ET.SubElement(entry, 'LOCATION', {
                    'DIR': os.path.dirname(track_data.get('filepath', '')),
                    'FILE': os.path.basename(track_data.get('filepath', '')),
                    'VOLUME': '',
                    'VOLUMEID': ''
                })
                
                # Info
                info_attrs = {
                    'TITLE': track_data.get('title', ''),
                    'ARTIST': track_data.get('artist', ''),
                    'ALBUM': track_data.get('album', ''),
                    'GENRE': track_data.get('genre', ''),
                    'BITRATE': str(track_data.get('bitrate', 320)),
                    'PLAYTIME': str(int(track_data.get('duration', 0))),
                    'IMPORT_DATE': str(int(time.time())),
                    'RELEASE_DATE': str(track_data.get('year', '')) if track_data.get('year') else '',
                    'FLAGS': '12'
                }
                
                if track_data.get('bmp'):
                    info_attrs['BPM'] = str(track_data['bpm'])
                if track_data.get('musical_key'):
                    info_attrs['KEY'] = track_data['musical_key']
                
                ET.SubElement(entry, 'INFO', info_attrs)
            
            self._write_formatted_xml(root, job.output_path)
            
        except Exception as e:
            raise ExportError(f"Traktor export failed: {str(e)}")
    
    def _export_virtualdj(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Export to Virtual DJ XML format"""
        try:
            root = ET.Element('VirtualDJ_Database', {'Version': '2021'})
            
            for i, track_data in enumerate(tracks):
                job.progress = (i / len(tracks)) * 100
                
                song = ET.SubElement(root, 'Song', {
                    'FilePath': track_data.get('filepath', ''),
                    'FileSize': str(track_data.get('filesize', 0))
                })
                
                # Tags
                tags = ET.SubElement(song, 'Tags')
                tags.set('Title', track_data.get('title', ''))
                tags.set('Artist', track_data.get('artist', ''))
                tags.set('Album', track_data.get('album', ''))
                tags.set('Genre', track_data.get('genre', ''))
                
                if track_data.get('year'):
                    tags.set('Year', str(track_data['year']))
                if track_data.get('bpm'):
                    tags.set('Bpm', str(track_data['bpm']))
                if track_data.get('musical_key'):
                    tags.set('Key', track_data['musical_key'])
                
                # Infos
                infos = ET.SubElement(song, 'Infos')
                infos.set('SongLength', str(track_data.get('duration', 0)))
                infos.set('Bitrate', str(track_data.get('bitrate', 320)))
            
            self._write_formatted_xml(root, job.output_path)
            
        except Exception as e:
            raise ExportError(f"Virtual DJ export failed: {str(e)}")
    
    def _export_m3u8(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Export to M3U8 playlist format"""
        try:
            lines = ['#EXTM3U']
            
            for i, track_data in enumerate(tracks):
                job.progress = (i / len(tracks)) * 100
                
                # Extended info line
                duration = int(track_data.get('duration', 0))
                artist = track_data.get('artist', 'Unknown Artist')
                title = track_data.get('title', 'Unknown Title')
                
                lines.append(f'#EXTINF:{duration},{artist} - {title}')
                lines.append(track_data.get('filepath', ''))
                lines.append('')  # Empty line for readability
            
            # Write playlist file
            with open(job.output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            raise ExportError(f"M3U8 export failed: {str(e)}")
    
    def _export_csv(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Export to CSV format"""
        try:
            # Define CSV columns
            columns = options.get('columns', [
                'filepath', 'title', 'artist', 'album', 'genre', 'year', 
                'bpm', 'musical_key', 'duration', 'bitrate', 'filesize'
            ])
            
            with open(job.output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(columns)
                
                # Write tracks
                for i, track_data in enumerate(tracks):
                    job.progress = (i / len(tracks)) * 100
                    
                    row = []
                    for column in columns:
                        value = track_data.get(column, '')
                        # Handle special formatting
                        if column == 'duration' and isinstance(value, (int, float)):
                            # Convert to MM:SS format
                            minutes = int(value // 60)
                            seconds = int(value % 60)
                            value = f"{minutes:02d}:{seconds:02d}"
                        
                        row.append(str(value))
                    
                    writer.writerow(row)
                    
        except Exception as e:
            raise ExportError(f"CSV export failed: {str(e)}")
    
    def _export_json(self, job: ExportJob, tracks: List[Dict[str, Any]], options: Dict[str, Any]):
        """Export to JSON format"""
        try:
            # Prepare export data
            export_data = {
                'export_info': {
                    'format': 'DJ Music Cleaner JSON Export',
                    'version': '1.0',
                    'created_at': time.time(),
                    'track_count': len(tracks)
                },
                'tracks': []
            }
            
            for i, track_data in enumerate(tracks):
                job.progress = (i / len(tracks)) * 100
                
                # Clean track data for JSON serialization
                clean_track = {}
                for key, value in track_data.items():
                    if value is not None:
                        clean_track[key] = value
                
                export_data['tracks'].append(clean_track)
            
            # Write JSON file
            with open(job.output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            raise ExportError(f"JSON export failed: {str(e)}")
    
    def _add_rekordbox_playlist(self, playlists_elem: ET.Element, 
                               playlist_data: Dict[str, Any], 
                               all_tracks: List[Dict[str, Any]]):
        """Add playlist to Rekordbox XML"""
        try:
            playlist_name = playlist_data.get('name', 'Untitled Playlist')
            track_ids = playlist_data.get('track_ids', [])
            
            node = ET.SubElement(playlists_elem, 'NODE', {
                'Type': '0',
                'Name': playlist_name,
                'Count': str(len(track_ids))
            })
            
            # Add tracks to playlist
            for track_id in track_ids:
                if isinstance(track_id, int) and 0 <= track_id < len(all_tracks):
                    ET.SubElement(node, 'TRACK', {'Key': str(track_id + 1)})
                    
        except Exception:
            # Skip playlist if there's an error
            pass
    
    def _write_formatted_xml(self, root: ET.Element, filepath: str):
        """Write XML with proper formatting"""
        try:
            # Convert to string and format
            rough_string = ET.tostring(root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            
            # Write formatted XML
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(reparsed.toprettyxml(indent='  '))
                
        except Exception as e:
            raise ExportError(f"XML formatting failed: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an export job"""
        with self._lock:
            job = self.active_jobs.get(job_id)
            if job:
                return {
                    'job_id': job.job_id,
                    'format_name': job.format_name,
                    'output_path': job.output_path,
                    'track_count': job.track_count,
                    'progress': job.progress,
                    'status': job.status,
                    'created_at': job.created_at,
                    'started_at': job.started_at,
                    'completed_at': job.completed_at,
                    'errors': job.errors
                }
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel an export job"""
        with self._lock:
            job = self.active_jobs.get(job_id)
            if job and job.status in ['pending', 'running']:
                job.status = 'cancelled'
                job.completed_at = time.time()
                return True
            return False
    
    def cleanup_completed_jobs(self, max_age_hours: float = 24.0):
        """Clean up completed jobs older than specified age"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self._lock:
            job_ids_to_remove = []
            for job_id, job in self.active_jobs.items():
                if (job.status in ['completed', 'failed', 'cancelled'] and
                    job.completed_at and job.completed_at < cutoff_time):
                    job_ids_to_remove.append(job_id)
            
            for job_id in job_ids_to_remove:
                del self.active_jobs[job_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get export service statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats['active_jobs'] = len(self.active_jobs)
            stats['supported_formats'] = len(self.formats)
            
            if stats['total_export_time'] > 0 and stats['exports_completed'] > 0:
                stats['average_export_time'] = stats['total_export_time'] / stats['exports_completed']
            else:
                stats['average_export_time'] = 0.0
            
            return stats


__all__ = ['ExportService', 'ExportFormat', 'ExportJob']