#!/usr/bin/env python3
"""
DJ Analysis Client - Interface to the isolated audio analysis service
Creates and manages a subprocess for DJ features to isolate segmentation faults
"""

import os
import sys
import json
import time
import signal
import subprocess
from threading import Timer


class DJAnalysisClient:
    """
    Client for communicating with the DJ analysis service in a separate process
    """
    
    def __init__(self, timeout=30, fallback_to_direct=True):
        """
        Initialize the DJ analysis client
        
        Args:
            timeout: Maximum time in seconds to wait for analysis (default: 30)
            fallback_to_direct: Whether to fall back to direct execution on failure (default: True)
        """
        self.process = None
        self.timeout = timeout
        self.fallback_to_direct = fallback_to_direct
        
    def _start_service(self):
        """
        Start the DJ analysis service in a separate process
        
        Returns:
            bool: True if service started successfully, False otherwise
        """
        if self.process is not None and self.process.poll() is None:
            # Service already running
            return True
            
        try:
            # Get the path to the service module
            service_path = os.path.join(os.path.dirname(__file__), 'dj_analysis_service.py')
            
            # Start the service as a subprocess
            self.process = subprocess.Popen(
                [sys.executable, service_path, "--service"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Check if process started successfully
            if self.process.poll() is None:
                print("✅ DJ analysis service started in isolated process")
                return True
            else:
                print("❌ Failed to start DJ analysis service")
                return False
                
        except Exception as e:
            print(f"❌ Error starting DJ analysis service: {e}")
            return False
    
    def _send_request(self, filepath, analysis_type="all"):
        """
        Send a request to the DJ analysis service
        
        Args:
            filepath: Path to audio file
            analysis_type: Type of analysis ('key', 'bpm', 'energy', 'all')
            
        Returns:
            dict: Analysis results or error
        """
        if not self._start_service():
            return {"error": "Failed to start DJ analysis service"}
            
        try:
            # Prepare request
            request = {
                "filepath": filepath,
                "analysis_type": analysis_type
            }
            
            # Send request to subprocess
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            
            # Set up timeout
            response = [None]
            timed_out = [False]
            
            def read_response():
                try:
                    line = self.process.stdout.readline()
                    if line:
                        response[0] = json.loads(line)
                except Exception as e:
                    response[0] = {"error": f"Failed to read response: {e}"}
            
            def timeout_handler():
                timed_out[0] = True
                print(f"⚠️ DJ analysis timed out after {self.timeout} seconds")
                try:
                    # Send SIGTERM to process group to terminate any child processes
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except:
                    pass
                    
                # Force restart the service
                self._stop_service()
                self._start_service()
                
            # Start reading response in a timer
            timer = Timer(self.timeout, timeout_handler)
            timer.start()
            
            # Wait for response or timeout
            read_response()
            
            # Cancel timer if response received
            timer.cancel()
            
            # Check if timed out
            if timed_out[0]:
                return {"error": f"Analysis timed out after {self.timeout} seconds"}
            
            # Return response
            return response[0].get("results", {"error": "No results received"})
            
        except Exception as e:
            print(f"❌ Error communicating with DJ analysis service: {e}")
            
            # Restart service on error
            self._stop_service()
            self._start_service()
            
            return {"error": f"Communication error: {e}"}
    
    def _stop_service(self):
        """
        Stop the DJ analysis service
        """
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
    
    def _direct_analysis(self, filepath, analysis_type):
        """
        Fall back to direct analysis (no process isolation)
        
        Args:
            filepath: Path to audio file
            analysis_type: Type of analysis ('key', 'bpm', 'energy', 'all')
            
        Returns:
            dict: Analysis results or error
        """
        try:
            # Import the service module directly
            from .dj_analysis_service import analyze_file
            
            # Call the analysis function directly
            return analyze_file(filepath, analysis_type)
            
        except Exception as e:
            return {"error": f"Direct analysis failed: {e}"}
    
    def analyze(self, filepath, analysis_type="all"):
        """
        Public API: Analyze audio file with process isolation
        
        Args:
            filepath: Path to audio file
            analysis_type: Type of analysis ('key', 'bpm', 'energy', 'all')
            
        Returns:
            dict: Analysis results
        """
        print(f"🎧 Analyzing {os.path.basename(filepath)} (type: {analysis_type})...")
        
        # First try with process isolation
        results = self._send_request(filepath, analysis_type)
        
        # Check for error
        if "error" in results and self.fallback_to_direct:
            print(f"⚠️ Isolated analysis failed, falling back to direct execution")
            results = self._direct_analysis(filepath, analysis_type)
        
        return results
    
    def get_key(self, filepath):
        """
        Get musical key of audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            tuple: (key, confidence, camelot) or (None, 0, None) if detection failed
        """
        results = self.analyze(filepath, "key")
        
        if "error" in results:
            print(f"❌ Key detection failed: {results['error']}")
            return None, 0, None
            
        return (
            results.get("key"),
            results.get("key_confidence", 0),
            results.get("camelot")
        )
    
    def get_bpm(self, filepath):
        """
        Get BPM of audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            float: BPM value or None if detection failed
        """
        results = self.analyze(filepath, "bpm")
        
        if "error" in results:
            print(f"❌ BPM detection failed: {results['error']}")
            return None
            
        return results.get("bpm")
    
    def get_energy(self, filepath):
        """
        Get energy/intensity of audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            float: Energy value (0-10) or None if detection failed
        """
        results = self.analyze(filepath, "energy")
        
        if "error" in results:
            print(f"❌ Energy detection failed: {results['error']}")
            return None
            
        return results.get("energy")
    
    def get_all(self, filepath):
        """
        Get all DJ analysis features for audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            dict: Analysis results
        """
        return self.analyze(filepath, "all")
    
    def __del__(self):
        """
        Clean up resources
        """
        self._stop_service()


# Singleton instance for reuse
_client_instance = None

def get_client():
    """
    Get singleton instance of DJAnalysisClient
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = DJAnalysisClient()
    return _client_instance


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        # Create client
        client = DJAnalysisClient()
        
        # Analyze file
        filepath = sys.argv[1]
        results = client.get_all(filepath)
        
        # Print results
        print(json.dumps(results, indent=2))
        
        # Clean up
        client._stop_service()
    else:
        print("Usage: python dj_analysis_client.py <audio_file>")
