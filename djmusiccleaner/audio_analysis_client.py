#!/usr/bin/env python3
"""
DJ Music Cleaner - Audio Analysis Client

Client module for communicating with the audio analysis service.
This provides a clean interface for the main application to use
audio analysis features without the risk of crashing due to
segmentation faults in native audio analysis libraries.
"""

import os
import sys
import json
import time
import atexit
import traceback
import multiprocessing
from typing import Dict, Any, Tuple, List, Optional, Union

# Import the audio analysis service module
from djmusiccleaner.audio_analysis_service import start_audio_analysis_service

# Define timeout for service communication
SERVICE_TIMEOUT = 60  # seconds


class AudioAnalysisClient:
    """
    Client for interacting with the audio analysis service.
    
    This class provides a simple interface for the main application
    to use audio analysis features while isolating potentially
    unstable native libraries in a separate process.
    """
    
    def __init__(self):
        """Initialize the audio analysis client"""
        self.process = None
        self.conn = None
        self.available = False
        self.service_errors = 0
        self.timeout_errors = 0
        self._start_service()
        
        # Register cleanup handler
        atexit.register(self.shutdown)
        
    def _start_service(self):
        """Start the audio analysis service process"""
        try:
            self.process, self.conn = start_audio_analysis_service()
            self.available = True
            
            # Test the connection with a ping
            result = self._send_command('ping', {})
            if not result.get('success'):
                print(f"Warning: Audio analysis service initialization issue: {result.get('error')}")
                self.available = False
        except Exception as e:
            print(f"Error starting audio analysis service: {str(e)}")
            traceback.print_exc()
            self.available = False
    
    def _send_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a command to the audio analysis service.
        
        Args:
            command: The command to send
            params: Parameters for the command
            
        Returns:
            Dictionary with the service response
        """
        if not self.available:
            return {
                'success': False,
                'error': 'Audio analysis service not available',
                'command': command
            }
            
        try:
            # Send the request
            request = {
                'command': command,
                'params': params
            }
            self.conn.send(request)
            
            # Wait for response with timeout
            if self.conn.poll(SERVICE_TIMEOUT):
                response = self.conn.recv()
                return response
            else:
                self.timeout_errors += 1
                return {
                    'success': False,
                    'error': f'Timeout waiting for response from audio analysis service',
                    'command': command
                }
                
        except Exception as e:
            self.service_errors += 1
            return {
                'success': False,
                'error': f'Error communicating with audio analysis service: {str(e)}',
                'command': command
            }
    
    def detect_bpm(self, filepath: str) -> Tuple[Optional[float], float]:
        """
        Detect BPM (tempo) of an audio file.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            Tuple of (bpm, confidence) or (None, 0) on error
        """
        result = self._send_command('detect_bpm', {'filepath': filepath})
        
        if result.get('success'):
            return result.get('bpm'), result.get('confidence', 0.0)
        else:
            print(f"BPM detection error: {result.get('error')}")
            return None, 0.0
    
    def detect_key(self, filepath: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Detect musical key of an audio file.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            Tuple of (key, camelot_key, confidence) or (None, None, 0) on error
        """
        result = self._send_command('detect_key', {'filepath': filepath})
        
        if result.get('success'):
            return result.get('key'), result.get('camelot_key'), result.get('confidence', 0.0)
        else:
            print(f"Key detection error: {result.get('error')}")
            return None, None, 0.0
    
    def calculate_energy(self, filepath: str) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """
        Calculate energy rating of a track for DJ applications.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            Tuple of (energy_score, characteristics) or (None, None) on error
        """
        result = self._send_command('calculate_energy', {'filepath': filepath})
        
        if result.get('success'):
            return result.get('energy_score'), result.get('characteristics', {})
        else:
            print(f"Energy calculation error: {result.get('error')}")
            return None, None
            
    def detect_cue_points(self, filepath: str, output_file: str = None) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        """
        Detect ideal cue points for DJ mixing using aubio.
        
        Args:
            filepath: Path to the audio file
            output_file: Optional path for saving cue points
            
        Returns:
            Tuple of (cue_points, sections) or (None, None) on error
            cue_points is a list of dictionaries with 'position', 'type', and 'description'
            sections is a list of dictionaries with section markers and types
        """
        params = {'filepath': filepath}
        if output_file:
            params['output_file'] = output_file
            
        result = self._send_command('detect_cue_points', params)
        
        if result.get('success'):
            return result.get('cue_points', []), result.get('sections', [])
        else:
            print(f"Cue point detection error: {result.get('error')}")
            return None, None
    
    def check_status(self) -> Dict[str, Any]:
        """
        Check the status of the audio analysis service.
        
        Returns:
            Dictionary with status information
        """
        if not self.available:
            return {
                'available': False,
                'error': 'Service not initialized'
            }
            
        result = self._send_command('ping', {})
        
        if result.get('success'):
            return {
                'available': True,
                'libraries': result.get('libraries', {}),
                'service_errors': self.service_errors,
                'timeout_errors': self.timeout_errors
            }
        else:
            return {
                'available': False,
                'error': result.get('error'),
                'service_errors': self.service_errors,
                'timeout_errors': self.timeout_errors
            }
    
    def shutdown(self):
        """Shut down the audio analysis service"""
        if self.conn and self.available:
            try:
                self.conn.send({'command': 'exit'})
                time.sleep(0.5)  # Give it time to exit cleanly
            except:
                pass
            
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
            
        self.available = False


# Singleton instance
_client_instance = None


def get_audio_analysis_client() -> AudioAnalysisClient:
    """
    Get the singleton instance of the AudioAnalysisClient.
    
    Returns:
        AudioAnalysisClient instance
    """
    global _client_instance
    
    if _client_instance is None:
        _client_instance = AudioAnalysisClient()
        
    return _client_instance


if __name__ == "__main__":
    # Simple test if run directly
    client = get_audio_analysis_client()
    status = client.check_status()
    print("Audio analysis service status:")
    print(json.dumps(status, indent=2))
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        
        if os.path.exists(filepath):
            print(f"\nTesting BPM detection on {filepath}")
            bpm, confidence = client.detect_bpm(filepath)
            print(f"BPM: {bpm}, Confidence: {confidence:.2f}")
            
            print(f"\nTesting key detection on {filepath}")
            key, camelot, confidence = client.detect_key(filepath)
            print(f"Key: {key} (Camelot: {camelot}), Confidence: {confidence:.2f}")
            
            print(f"\nTesting energy calculation on {filepath}")
            energy, characteristics = client.calculate_energy(filepath)
            print(f"Energy score: {energy}/10")
            if characteristics:
                print("Characteristics:")
                for k, v in characteristics.items():
                    print(f"  {k}: {v}")
                    
            print(f"\nTesting cue point detection on {filepath}")
            cue_points, sections = client.detect_cue_points(filepath)
            if cue_points:
                print(f"Found {len(cue_points)} cue points:")
                for i, cue in enumerate(cue_points[:5]):
                    print(f"  {i+1}. {cue['description']} at {cue['position']:.2f}s")
                if len(cue_points) > 5:
                    print(f"  ...and {len(cue_points) - 5} more")
            else:
                print("No cue points detected")
        else:
            print(f"File not found: {filepath}")
    else:
        print("Usage: python audio_analysis_client.py <audio_file>")
