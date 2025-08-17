"""
Configuration Validator and Directory Handler

Validates configuration files and creates necessary directories.
Handles path resolution, directory creation, and file existence checks.
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..utils.logging_config import get_logger


class ConfigValidator:
    """Validates configuration and ensures required directories exist"""
    
    def __init__(self):
        self.logger = get_logger('config_validator')
        self.created_directories = []
        self.validation_warnings = []
        
    def validate_and_prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and create necessary directories
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Updated configuration with validated paths
        """
        self.logger.debug("Starting configuration validation")
        
        # Create a working copy
        validated_config = config.copy()
        
        # Validate and create directories
        self._validate_output_directories(validated_config)
        self._validate_rekordbox_paths(validated_config)
        self._validate_reporting_paths(validated_config)
        self._validate_logging_paths(validated_config)
        
        # Report what was created
        if self.created_directories:
            print("📁 Created directories:")
            for directory in self.created_directories:
                print(f"   ✅ {directory}")
        
        # Report warnings
        if self.validation_warnings:
            print("⚠️  Configuration warnings:")
            for warning in self.validation_warnings:
                print(f"   ⚠️  {warning}")
        
        self.logger.debug(f"Configuration validation completed. Created {len(self.created_directories)} directories.")
        
        return validated_config
    
    def _validate_output_directories(self, config: Dict[str, Any]):
        """Validate and create output directories"""
        file_ops = config.get('file_operations', {})
        
        # Output directory for organized files
        output_dir = file_ops.get('output_directory')
        if output_dir:
            self._ensure_directory_exists(output_dir, 'output directory')
    
    def _validate_rekordbox_paths(self, config: Dict[str, Any]):
        """Validate and create Rekordbox-related paths"""
        rekordbox = config.get('rekordbox', {})
        
        # XML path - could be directory or file
        xml_path = rekordbox.get('xml_path')
        if xml_path:
            xml_path_obj = Path(xml_path)
            
            if xml_path.endswith('.xml') or '.' in xml_path_obj.name:
                # It's a file path - ensure parent directory exists
                parent_dir = xml_path_obj.parent
                self._ensure_directory_exists(str(parent_dir), 'Rekordbox XML parent directory')
                
                # Check if XML file exists, if not, warn user
                if not xml_path_obj.exists():
                    self.validation_warnings.append(f"Rekordbox XML file not found: {xml_path}")
                    self.validation_warnings.append("XML integration will be disabled until file is available")
            else:
                # It's a directory path - create and look for XML files
                self._ensure_directory_exists(xml_path, 'Rekordbox XML directory')
                
                # Look for XML files in the directory
                xml_files = list(Path(xml_path).glob('*.xml'))
                if xml_files:
                    # Found XML file(s) - use the first one
                    selected_xml = str(xml_files[0])
                    config['rekordbox']['xml_path'] = selected_xml
                    print(f"📄 Found Rekordbox XML: {selected_xml}")
                else:
                    # No XML files found - directory will be used for creating new XML
                    self.validation_warnings.append(f"No XML files found in directory: {xml_path}")
                    print(f"📁 Directory will be used for creating new Rekordbox XML file")
                    # Don't disable integration - let the CLI handle creating new files
        
        # Report path
        report_path = rekordbox.get('report_path')
        if report_path and not report_path.endswith('.xml'):
            # It's a directory for reports
            self._ensure_directory_exists(report_path, 'Rekordbox report directory')
    
    def _validate_reporting_paths(self, config: Dict[str, Any]):
        """Validate and create reporting paths"""
        reporting = config.get('reporting', {})
        
        # Report path
        report_path = reporting.get('report_path')
        if report_path:
            report_path_obj = Path(report_path)
            
            # Check if it's a file path or directory path
            if report_path.endswith(('.html', '.json', '.csv')) or '.' in report_path_obj.suffix:
                # It's a file path - ensure parent directory exists
                parent_dir = report_path_obj.parent
                self._ensure_directory_exists(str(parent_dir), 'report parent directory')
            else:
                # It's a directory path - create directory
                # CLI will handle automatic filename generation
                self._ensure_directory_exists(report_path, 'report directory')
    
    def _validate_logging_paths(self, config: Dict[str, Any]):
        """Validate and create logging directories"""
        logging_config = config.get('logging', {})
        
        # Log directory
        log_dir = logging_config.get('log_dir')
        if log_dir:
            self._ensure_directory_exists(log_dir, 'log directory')
    
    def _ensure_directory_exists(self, directory_path: str, description: str = "directory"):
        """
        Ensure directory exists, create if it doesn't
        
        Args:
            directory_path: Path to the directory
            description: Human-readable description for logging
        """
        try:
            directory_path = os.path.expanduser(directory_path)
            path_obj = Path(directory_path)
            
            if not path_obj.exists():
                path_obj.mkdir(parents=True, exist_ok=True)
                self.created_directories.append(directory_path)
                self.logger.debug(f"Created {description}: {directory_path}")
            elif not path_obj.is_dir():
                self.validation_warnings.append(f"Path exists but is not a directory: {directory_path}")
                self.logger.warning(f"Path exists but is not a directory: {directory_path}")
            else:
                self.logger.debug(f"Directory already exists: {directory_path}")
                
        except PermissionError:
            error_msg = f"Permission denied creating {description}: {directory_path}"
            self.validation_warnings.append(error_msg)
            self.logger.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to create {description} {directory_path}: {str(e)}"
            self.validation_warnings.append(error_msg)
            self.logger.error(error_msg)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        return {
            'directories_created': len(self.created_directories),
            'created_paths': self.created_directories.copy(),
            'warnings_count': len(self.validation_warnings),
            'warnings': self.validation_warnings.copy()
        }


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration
    """
    validator = ConfigValidator()
    return validator.validate_and_prepare_config(config)


__all__ = ['ConfigValidator', 'validate_config']