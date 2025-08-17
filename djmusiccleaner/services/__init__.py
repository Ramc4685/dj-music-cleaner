"""
Unified Service Layer for DJ Music Cleaner

This package contains the new unified services that consolidate
functionality from all previous implementations with a clean,
service-oriented architecture.
"""

# Import only the new unified services to avoid dependency issues
from .unified_audio_analysis import UnifiedAudioAnalysisService
from .unified_cache import UnifiedCacheService
from .unified_metadata import UnifiedMetadataService

# Use compatibility layers where needed
from .analytics_compatibility import AnalyticsService

# Legacy services are preserved in their original files but not imported here
# to avoid sklearn and other heavy dependencies during initialization.
# They can still be imported explicitly if needed for specific features.

__all__ = [
    'UnifiedAudioAnalysisService',
    'UnifiedCacheService', 
    'UnifiedMetadataService'
]