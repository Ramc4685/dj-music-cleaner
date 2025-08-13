#!/usr/bin/env python3
"""
Settings persistence module for DJ Music Cleaner GUI
Handles saving and loading user preferences
"""

import os
import json
from pathlib import Path

# Settings file location - in user home directory
SETTINGS_DIR = os.path.expanduser("~/.djprep_gui")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")

# Default settings
DEFAULT_SETTINGS = {
    # Basic settings
    "input_folder": "",
    "output_folder": "",
    "acoustid_api_key": "",
    "last_used_date": "",  # Will be updated when saving
    
    # Option categories
    "online_enhancement": False,
    "dj_analysis": True,
    "high_quality_only": False,
    
    # Online Enhancement options
    "cache": "",
    "skip_id3": False,
    
    # DJ Analysis options - defaults when dj_analysis is TRUE
    "detect_key": True,
    "detect_cues": True,
    "calculate_energy": True,
    "detect_bpm": True,
    "normalize_tags": True,
    "normalize_loudness": False,
    "target_lufs": -14.0,
    
    # Rekordbox options - defaults are all FALSE
    "rekordbox_xml": "",
    "export_rekordbox": "",
    "rekordbox_preserve": False,
    
    # High Quality options - defaults when high_quality_only is TRUE
    "analyze_audio": True,
    "report": True,
    "detailed_report": True,
    "json_report": True,
    "csv_report": False,
    "html_report_path": "",
    "year_in_filename": False,
    "dry_run": False,
    "workers": 0,
    "find_duplicates": False,  # Changed from True to match GUI default
    "show_priorities": True
}

def ensure_settings_dir():
    """Create settings directory if it doesn't exist"""
    os.makedirs(SETTINGS_DIR, exist_ok=True)

def load_settings():
    """Load settings from file, or return defaults if file doesn't exist"""
    ensure_settings_dir()
    
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                
                # Update with any new default keys that might be missing
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in settings:
                        settings[key] = value
                        
                return settings
    except (json.JSONDecodeError, IOError, OSError) as e:
        print(f"Error loading settings: {e}")
    
    # Return default settings if file doesn't exist or has errors
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """Save settings to file"""
    ensure_settings_dir()
    
    # Update last used date
    from datetime import datetime
    settings["last_used_date"] = datetime.now().isoformat()
    
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except (IOError, OSError) as e:
        print(f"Error saving settings: {e}")
        return False
