"""
Reports module for DJ Music Cleaner
Handles detailed reporting for processed files and quality metrics
"""
import os
import json
import time
from datetime import datetime
import shutil

class DJReportManager:
    """Manages detailed reports for DJ music processing"""
    
    def __init__(self, base_dir=None):
        """Initialize the report manager with base directory"""
        self.base_dir = base_dir or os.getcwd()
        self.reports_dir = os.path.join(self.base_dir, "reports")
        self.ensure_reports_dir()
        
        # Keep track of files processed in this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.processed_files = []
        self.low_quality_files = []
        self.duplicate_files = []
        self.changes_log = []
        
    def ensure_reports_dir(self):
        """Ensure reports directory exists"""
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
            print(f"📁 Created reports directory: {self.reports_dir}")
    
    def get_processed_files_db_path(self):
        """Get path to the processed files database"""
        return os.path.join(self.reports_dir, "processed_files.json")
    
    def load_processed_files_db(self):
        """Load previously processed files database"""
        db_path = self.get_processed_files_db_path()
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Processed files database corrupted, creating new one")
                return {"files": {}, "last_update": ""}
        else:
            return {"files": {}, "last_update": ""}
    
    def save_processed_files_db(self, db_data):
        """Save processed files database"""
        db_path = self.get_processed_files_db_path()
        db_data["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, indent=2)
    
    def is_file_already_processed(self, file_path):
        """Check if a file has already been processed"""
        db = self.load_processed_files_db()
        file_key = os.path.basename(file_path)
        
        if file_key in db["files"]:
            # Check file modification time to see if it has changed
            current_mtime = os.path.getmtime(file_path)
            stored_mtime = db["files"][file_key].get("mtime", 0)
            
            # If file hasn't been modified since last processing
            if current_mtime <= stored_mtime:
                return db["files"][file_key]
        
        return None
    
    def mark_file_as_processed(self, file_path, file_info):
        """Mark a file as processed in the database"""
        db = self.load_processed_files_db()
        file_key = os.path.basename(file_path)
        
        # Store file info with modification time
        file_info["mtime"] = os.path.getmtime(file_path)
        file_info["last_processed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        db["files"][file_key] = file_info
        self.save_processed_files_db(db)
        
        # Add to session processed files
        self.processed_files.append(file_info)
    
    def add_low_quality_file(self, file_path, quality_info):
        """Add a file to the low quality list"""
        self.low_quality_files.append({
            "file_path": file_path,
            "quality_info": quality_info,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def add_duplicate_file(self, original_file, duplicate_file, similarity_score):
        """Add a file to the duplicates list"""
        self.duplicate_files.append({
            "original": original_file,
            "duplicate": duplicate_file,
            "similarity_score": similarity_score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def log_file_changes(self, file_path, changes_dict):
        """Log changes made to a file"""
        self.changes_log.append({
            "file_path": file_path,
            "changes": changes_dict,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def generate_low_quality_report(self):
        """Generate a report of low quality files that need replacement"""
        if not self.low_quality_files:
            return None
            
        report_path = os.path.join(self.reports_dir, f"low_quality_files_{self.session_timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DJ MUSIC CLEANER - LOW QUALITY FILES REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"The following {len(self.low_quality_files)} files are below the quality threshold (320kbps) and need to be replaced:\n\n")
            
            for i, file_info in enumerate(self.low_quality_files, 1):
                f.write(f"{i}. {os.path.basename(file_info['file_path'])}\n")
                f.write(f"   Path: {file_info['file_path']}\n")
                f.write(f"   Quality: {file_info['quality_info']['bitrate_kbps']}kbps, {file_info['quality_info']['sample_rate_khz']}kHz\n")
                f.write(f"   Length: {file_info['quality_info']['length']}\n")
                f.write("\n")
                
        print(f"📝 Low quality files report saved: {report_path}")
        return report_path
    
    def generate_changes_report(self):
        """Generate a detailed report of all changes made to files"""
        if not self.changes_log:
            return None
            
        report_path = os.path.join(self.reports_dir, f"changes_report_{self.session_timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DJ MUSIC CLEANER - DETAILED CHANGES REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, log_entry in enumerate(self.changes_log, 1):
                f.write(f"{i}. {os.path.basename(log_entry['file_path'])}\n")
                f.write(f"   Path: {log_entry['file_path']}\n")
                f.write(f"   Timestamp: {log_entry['timestamp']}\n")
                f.write(f"   Changes:\n")
                
                # Write detailed changes
                changes = log_entry['changes']
                
                # Metadata changes
                if 'metadata_changed' in changes and changes['metadata_changed']:
                    f.write(f"   - Metadata changes:\n")
                    for field, values in changes.get('changes', {}).items():
                        if 'original' in values and 'new' in values:
                            f.write(f"     * {field.capitalize()}: '{values['original']}' → '{values['new']}'\n")
                
                # Cleaning actions
                if 'cleaning_actions' in changes and changes['cleaning_actions']:
                    f.write(f"   - Cleaning actions:\n")
                    for action in changes['cleaning_actions']:
                        f.write(f"     * {action}\n")
                
                # Online enhancement
                if 'enhanced' in changes and changes['enhanced']:
                    f.write(f"   - Enhanced with online data: Yes\n")
                    if 'source' in changes:
                        f.write(f"     * Source: {changes['source']}\n")
                
                # Quality analysis
                if 'quality_info' in changes:
                    quality = changes['quality_info']
                    f.write(f"   - Quality analysis: {quality.get('quality_rating', 'Unknown')}\n")
                    f.write(f"     * Bitrate: {quality.get('bitrate_kbps', 0)}kbps\n")
                    f.write(f"     * Sample rate: {quality.get('sample_rate_khz', 0)}kHz\n")
                
                # DJ-specific analysis
                if 'dj_analysis' in changes:
                    dj_info = changes['dj_analysis']
                    if 'key' in dj_info:
                        f.write(f"   - Musical key: {dj_info['key']} (Camelot: {dj_info.get('camelot_key', 'N/A')})\n")
                    if 'energy' in dj_info:
                        f.write(f"   - Energy rating: {dj_info['energy']}/10\n")
                    if 'cue_points' in dj_info:
                        f.write(f"   - Cue points detected: {len(dj_info['cue_points'])}\n")
                
                # Normalization
                if 'normalized' in changes and changes['normalized']:
                    f.write(f"   - Loudness normalized: {changes.get('target_lufs', 'Unknown')} LUFS\n")
                
                f.write("\n")
                
        print(f"📝 Detailed changes report saved: {report_path}")
        return report_path
    
    def generate_duplicates_report(self, duplicates=None):
        """Generate a report of duplicate files"""
        # Use passed duplicates or fall back to stored duplicate_files
        duplicate_data = duplicates if duplicates is not None else self.duplicate_files
        if not duplicate_data:
            return None
            
        report_path = os.path.join(self.reports_dir, f"duplicate_files_{self.session_timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DJ MUSIC CLEANER - DUPLICATE FILES REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"The following {len(duplicate_data)} potential duplicate groups were found:\n\n")
            
            # Write duplicate groups
            for i, dup_group in enumerate(duplicate_data, 1):
                f.write(f"{i}. Duplicate Group #{i} ({dup_group['type']})\n")
                f.write(f"   Match Type: {dup_group.get('match_on', 'unknown')}\n")
                
                if 'similarity' in dup_group:
                    f.write(f"   Similarity: {dup_group['similarity']:.2f}\n")
                
                f.write(f"   Files in group ({len(dup_group['tracks'])}):\n")
                
                for j, track in enumerate(dup_group['tracks'], 1):
                    track_path = str(track['path'])
                    track_size = track.get('size', 0) // 1024  # Convert to KB
                    f.write(f"   {j}. {os.path.basename(track_path)}\n")
                    f.write(f"      Path: {track_path}\n")
                    f.write(f"      Size: {track_size} KB\n")
                    if track.get('artist'):
                        f.write(f"      Artist: {track['artist']}\n")
                    if track.get('title'):
                        f.write(f"      Title: {track['title']}\n")
                
                f.write("\n")
                
        print(f"📝 Duplicates report saved: {report_path}")
        return report_path
    
    def generate_session_summary(self, output_folder=None):
        """Generate a summary of the session"""
        summary_path = os.path.join(self.reports_dir, f"session_summary_{self.session_timestamp}.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("DJ MUSIC CLEANER - SESSION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Session: {self.session_timestamp}\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Files processed: {len(self.processed_files)}\n")
            f.write(f"High quality files moved: {len([f for f in self.processed_files if f.get('is_high_quality', False)])}\n")
            f.write(f"Low quality files (not moved): {len(self.low_quality_files)}\n")
            f.write(f"Potential duplicates found: {len(self.duplicate_files)}\n\n")
            
            if output_folder:
                f.write(f"Clean files location: {output_folder}\n\n")
            
            f.write("Reports generated:\n")
            if self.low_quality_files:
                f.write(f"- Low quality files report\n")
            if self.changes_log:
                f.write(f"- Detailed changes report\n")
            if self.duplicate_files:
                f.write(f"- Duplicates report\n")
        
        print(f"📝 Session summary saved: {summary_path}")
        return summary_path
