"""
Custom exceptions for DJ Music Cleaner

This module defines all custom exceptions used throughout the application
to provide clear error handling and debugging information.
"""

class DJMusicCleanerError(Exception):
    """Base exception for all DJ Music Cleaner errors"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__(message)
        self.message = message
        self.details = details
        self.filepath = filepath
    
    def __str__(self):
        parts = [self.message]
        if self.filepath:
            parts.append(f"File: {self.filepath}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class ProcessingError(DJMusicCleanerError):
    """Raised when file processing fails"""
    pass


class ServiceError(DJMusicCleanerError):
    """Raised when a service operation fails"""
    
    def __init__(self, service_name: str, message: str, details: str = None, filepath: str = None):
        super().__init__(message, details, filepath)
        self.service_name = service_name
    
    def __str__(self):
        return f"[{self.service_name}] {super().__str__()}"


class AudioAnalysisError(ServiceError):
    """Raised when audio analysis fails"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__("AudioAnalysis", message, details, filepath)


class MetadataError(ServiceError):
    """Raised when metadata processing fails"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__("Metadata", message, details, filepath)


class CacheError(ServiceError):
    """Raised when cache operations fail"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__("Cache", message, details, filepath)


class FileOperationError(ServiceError):
    """Raised when file operations fail"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__("FileOperations", message, details, filepath)


class ValidationError(ServiceError):
    """Raised when file validation fails"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__("Validation", message, details, filepath)


class RekordboxError(ServiceError):
    """Raised when Rekordbox integration fails"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__("Rekordbox", message, details, filepath)


class AnalyticsError(ServiceError):
    """Raised when analytics processing fails"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__("Analytics", message, details, filepath)


class ExportError(ServiceError):
    """Raised when export operations fail"""
    
    def __init__(self, message: str, details: str = None, filepath: str = None):
        super().__init__("Export", message, details, filepath)