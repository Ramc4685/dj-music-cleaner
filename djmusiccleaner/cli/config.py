"""
CLI Configuration Management

Provides configuration loading, validation, and management for the DJ Music Cleaner CLI.
Supports multiple configuration sources and formats with environment variable overrides.
"""

import os
import json
import sys
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import platform

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class CLIConfig:
    """
    CLI configuration manager
    
    Features:
    - Multiple configuration sources (file, environment, defaults)
    - Platform-specific configuration paths
    - Configuration validation
    - Environment variable overrides
    - Configuration file generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_path = config_path or self._get_default_config_path()
        self._config_cache: Optional[Dict[str, Any]] = None
        self._defaults = self._get_default_config()
        
        # Load .env file if available
        if DOTENV_AVAILABLE:
            # Look for .env in current directory and parent directories
            env_path = self._find_env_file()
            if env_path:
                load_dotenv(env_path)
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path based on platform"""
        if platform.system() == "Windows":
            config_dir = os.path.expandvars(r"%APPDATA%\DJMusicCleaner")
        elif platform.system() == "Darwin":  # macOS
            config_dir = os.path.expanduser("~/Library/Application Support/DJMusicCleaner")
        else:  # Linux and others
            config_dir = os.path.expanduser("~/.config/djmusiccleaner")
        
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "config.json")
    
    def _find_env_file(self) -> Optional[str]:
        """Find .env file in current directory or parent directories"""
        current_dir = Path.cwd()
        
        # Check current directory and up to 3 parent directories
        for _ in range(4):
            env_file = current_dir / '.env'
            if env_file.exists():
                return str(env_file)
            current_dir = current_dir.parent
            if current_dir == current_dir.parent:  # Reached root
                break
        
        return None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Application settings
            "app": {
                "version": "2.0",
                "debug": False,
                "log_level": "INFO",
                "max_log_size_mb": 10,
                "log_retention_days": 7
            },
            
            # Processing settings
            "processing": {
                "default_workers": 1,
                "max_workers": 8,
                "chunk_size": 100,
                "timeout_seconds": 300,
                "memory_limit_mb": 1024,
                "enable_multiprocessing": True
            },
            
            # Audio analysis
            "audio_analysis": {
                "enable_bpm_analysis": True,
                "enable_key_analysis": True,
                "enable_energy_analysis": True,
                "enable_quality_analysis": False,
                "sample_rate": 44100,
                "hop_length": 512,
                "frame_length": 2048,
                "confidence_threshold": 0.7
            },
            
            # Caching
            "cache": {
                "enable_cache": True,
                "cache_timeout_days": 30,
                "max_cache_size_mb": 500,
                "cache_compression": True,
                "cache_location": "auto"  # auto, custom path
            },
            
            # Metadata enhancement
            "metadata": {
                "enable_online_enhancement": True,
                "online_timeout_seconds": 10,
                "max_retries": 3,
                "preferred_sources": ["musicbrainz", "acoustid", "lastfm"],
                "rate_limit_delay": 1.0,
                "acoustid_api_key": None
            },
            
            # File operations
            "file_operations": {
                "enable_backup": True,
                "backup_location": "auto",
                "allow_rename": False,
                "include_year_in_filename": False,
                "organize_files": False,
                "filename_format": "{artist} - {title}",
                "folder_format": "{artist}/{album}",
                "safe_characters_only": True
            },
            
            # Quality and filtering
            "quality": {
                "min_quality_score": 0.0,
                "skip_duplicates": True,
                "min_file_size_kb": 100,
                "supported_formats": [".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aiff"],
                "skip_corrupted": True
            },
            
            # Reporting
            "reporting": {
                "enable_reports": True,
                "default_format": "html",
                "include_analytics": True,
                "auto_open_report": False,
                "report_location": "auto"
            },
            
            # Advanced features
            "advanced": {
                "enable_advanced_cues": False,
                "enable_advanced_beatgrid": False,
                "enable_calibrated_energy": False,
                "enable_professional_reporting": True,
                "max_cue_points": 8,
                "beatgrid_precision": "high"
            },
            
            # Rekordbox integration
            "rekordbox": {
                "auto_detect_xml": True,
                "backup_xml": True,
                "update_xml": False,
                "preserve_playlists": True,
                "preserve_cues": True
            },
            
            # Export settings
            "export": {
                "default_format": "json",
                "preserve_folder_structure": True,
                "convert_file_paths": True,
                "include_analysis_data": True,
                "backup_existing_files": True
            },
            
            # Performance optimization
            "performance": {
                "auto_optimize": True,
                "preferred_backend": "auto",  # auto, aubio, librosa
                "enable_gpu_acceleration": False,
                "memory_monitoring": True,
                "performance_profiling": False
            },
            
            # User interface
            "ui": {
                "color_output": True,
                "progress_bars": True,
                "verbose_by_default": False,
                "confirm_destructive_operations": True,
                "show_tips": True
            }
        }
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from all sources
        
        Args:
            force_reload: Force reload from file (ignore cache)
            
        Returns:
            Merged configuration dictionary
        """
        if self._config_cache is not None and not force_reload:
            return self._config_cache
        
        # Start with defaults
        config = self._defaults.copy()
        
        # Load from configuration file
        file_config = self._load_from_file()
        if file_config:
            config = self._deep_merge(config, file_config)
        
        # Apply environment variable overrides
        env_config = self._load_from_environment()
        if env_config:
            config = self._deep_merge(config, env_config)
        
        # Validate configuration
        config = self._validate_config(config)
        
        # Cache the result
        self._config_cache = config
        return config
    
    def _load_from_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config file {self.config_path}: {e}")
        
        return None
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables"""
        config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            'DJMC_WORKERS': ('processing', 'default_workers', int),
            'DJMC_CACHE_ENABLED': ('cache', 'enable_cache', self._str_to_bool),
            'DJMC_CACHE_TIMEOUT_DAYS': ('cache', 'cache_timeout_days', int),
            'DJMC_ONLINE_ENHANCEMENT': ('metadata', 'enable_online_enhancement', self._str_to_bool),
            'DJMC_VERBOSE': ('ui', 'verbose_by_default', self._str_to_bool),
            'DJMC_DEBUG': ('app', 'debug', self._str_to_bool),
            'DJMC_LOG_LEVEL': ('app', 'log_level', str),
            'DJMC_QUALITY_THRESHOLD': ('quality', 'min_quality_score', float),
            'DJMC_REKORDBOX_XML': ('rekordbox', 'auto_detect_xml', self._str_to_bool),
            'DJMC_BACKUP_ENABLED': ('file_operations', 'enable_backup', self._str_to_bool),
            'DJMC_ALLOW_RENAME': ('file_operations', 'allow_rename', self._str_to_bool),
            'ACOUSTID_API_KEY': ('metadata', 'acoustid_api_key', str),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    if section not in config:
                        config[section] = {}
                    config[section][key] = converted_value
                except Exception as e:
                    print(f"Warning: Invalid environment variable {env_var}={value}: {e}")
        
        return config
    
    def _str_to_bool(self, value: str) -> bool:
        """Convert string to boolean"""
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration values"""
        # Validate processing settings
        processing = config.get('processing', {})
        processing['default_workers'] = max(1, min(processing.get('default_workers', 1), 32))
        processing['max_workers'] = max(1, min(processing.get('max_workers', 8), 32))
        processing['memory_limit_mb'] = max(128, min(processing.get('memory_limit_mb', 1024), 8192))
        
        # Validate cache settings
        cache = config.get('cache', {})
        cache['cache_timeout_days'] = max(1, min(cache.get('cache_timeout_days', 30), 365))
        cache['max_cache_size_mb'] = max(10, min(cache.get('max_cache_size_mb', 500), 10000))
        
        # Validate quality settings
        quality = config.get('quality', {})
        quality['min_quality_score'] = max(0.0, min(quality.get('min_quality_score', 0.0), 10.0))
        quality['min_file_size_kb'] = max(1, quality.get('min_file_size_kb', 100))
        
        # Validate audio analysis settings
        audio = config.get('audio_analysis', {})
        audio['confidence_threshold'] = max(0.0, min(audio.get('confidence_threshold', 0.7), 1.0))
        audio['sample_rate'] = max(8000, min(audio.get('sample_rate', 44100), 192000))
        
        return config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save (uses current if None)
            
        Returns:
            True if saved successfully
        """
        try:
            if config is None:
                config = self.load_config()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Write configuration file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Clear cache to force reload
            self._config_cache = None
            
            return True
            
        except Exception as e:
            print(f"Error: Failed to save config file {self.config_path}: {e}")
            return False
    
    def get_option(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration option using dot notation
        
        Args:
            path: Dot-separated path (e.g., 'processing.default_workers')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_config()
        
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_option(self, path: str, value: Any) -> bool:
        """
        Set a configuration option using dot notation
        
        Args:
            path: Dot-separated path (e.g., 'processing.default_workers')
            value: Value to set
            
        Returns:
            True if set successfully
        """
        try:
            config = self.load_config()
            keys = path.split('.')
            
            # Navigate to the parent dict
            target = config
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # Set the value
            target[keys[-1]] = value
            
            return self.save_config(config)
            
        except Exception as e:
            print(f"Error: Failed to set option {path}: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        return self.save_config(self._defaults.copy())
    
    def validate_file_config(self, filepath: str) -> tuple[bool, List[str]]:
        """
        Validate a configuration file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            if not os.path.exists(filepath):
                errors.append(f"Configuration file not found: {filepath}")
                return False, errors
            
            # Try to load the JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if not isinstance(config, dict):
                errors.append("Configuration must be a JSON object")
                return False, errors
            
            # Validate structure against defaults
            self._validate_config_structure(config, self._defaults, [], errors)
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON syntax: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors
    
    def _validate_config_structure(self, config: Dict[str, Any], reference: Dict[str, Any], 
                                 path: List[str], errors: List[str]):
        """Recursively validate configuration structure"""
        for key, value in config.items():
            current_path = path + [key]
            path_str = '.'.join(current_path)
            
            if key not in reference:
                errors.append(f"Unknown configuration option: {path_str}")
                continue
            
            reference_value = reference[key]
            
            if isinstance(reference_value, dict):
                if not isinstance(value, dict):
                    errors.append(f"Configuration option {path_str} must be an object")
                else:
                    self._validate_config_structure(value, reference_value, current_path, errors)
            else:
                # Type validation would go here if needed
                pass
    
    def export_schema(self, filepath: str) -> bool:
        """Export configuration schema for documentation"""
        try:
            schema = self._generate_schema(self._defaults)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error: Failed to export schema: {e}")
            return False
    
    def _generate_schema(self, config: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Generate JSON schema from default configuration"""
        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
        
        for key, value in config.items():
            if isinstance(value, dict):
                schema["properties"][key] = self._generate_schema(value, f"{path}.{key}" if path else key)
            else:
                schema["properties"][key] = {
                    "type": type(value).__name__.lower(),
                    "default": value
                }
                
                # Add constraints for known types
                if isinstance(value, int) and key.endswith('_workers'):
                    schema["properties"][key]["minimum"] = 1
                    schema["properties"][key]["maximum"] = 32
                elif isinstance(value, float) and 'score' in key:
                    schema["properties"][key]["minimum"] = 0.0
                    schema["properties"][key]["maximum"] = 10.0
        
        return schema
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about current configuration"""
        config = self.load_config()
        
        return {
            "config_path": self.config_path,
            "config_exists": os.path.exists(self.config_path),
            "sections": list(config.keys()),
            "total_options": sum(len(v) if isinstance(v, dict) else 1 for v in config.values()),
            "last_modified": os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else None
        }


def load_config_from_args(args) -> Dict[str, Any]:
    """
    Load configuration from command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    config_manager = CLIConfig(getattr(args, 'config', None))
    return config_manager.load_config()


__all__ = ['CLIConfig', 'load_config_from_args']