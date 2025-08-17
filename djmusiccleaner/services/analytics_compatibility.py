"""
Analytics Service Compatibility Layer

This module patches compatibility issues between legacy service calls
and the new unified service architecture.
"""

from typing import Dict, Any
from .analytics import AnalyticsService as BaseAnalyticsService


class AnalyticsService(BaseAnalyticsService):
    """
    Enhanced AnalyticsService with unified interface compatibility
    
    Adds compatibility methods for the unified architecture while
    maintaining all functionality from the base analytics service.
    """
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for unified interface
        
        This method is called by the unified CLI and service layer.
        It simply delegates to the existing get_session_stats method
        to maintain backward compatibility.
        
        Returns:
            Dict containing current session statistics
        """
        return self.get_session_stats()
