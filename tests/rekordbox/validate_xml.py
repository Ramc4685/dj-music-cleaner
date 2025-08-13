#!/usr/bin/env python3
"""
Rekordbox XML Validation Script

This script automates the validation of the round-trip preservation of Rekordbox XML data.
It compares an original Rekordbox XML export with a processed version to ensure that all
critical DJ metadata (beat grids, cue points, etc.) is properly preserved.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import tempfile
import logging
import json
import hashlib
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("rekordbox-validator")

class RekordboxValidator:
    """Validates Rekordbox XML exports to ensure DJ metadata is preserved."""
    
    def __init__(self, original_xml, processed_xml):
        """Initialize with paths to original and processed XML files."""
        self.original_xml = original_xml
        self.processed_xml = processed_xml
        self.original_tracks = {}
        self.processed_tracks = {}
        self.results = {
            "total_tracks_original": 0,
            "total_tracks_processed": 0,
            "matched_tracks": 0,
            "unmatched_tracks": 0,
            "metadata_issues": [],
            "track_issues": defaultdict(list),
            "success_rate": 0.0
        }
        
    def parse_xml(self, xml_path):
        """Parse Rekordbox XML and extract track data."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            tracks = {}
            collection = root.find(".//COLLECTION")
            
            if collection is None:
                logger.error(f"No COLLECTION found in {xml_path}")
                return {}
                
            track_elements = collection.findall("TRACK")
            
            for track in track_elements:
                track_id = track.get("TrackID")
                if not track_id:
                    continue
                    
                # Parse basic track attributes
                track_data = {
                    "track_id": track_id,
                    "name": track.get("Name", ""),
                    "artist": track.get("Artist", ""),
                    "album": track.get("Album", ""),
                    "location": track.get("Location", ""),
                    "tempo_data": [],
                    "position_marks": [],
                    "attributes": {k: v for k, v in track.attrib.items()}
                }
                
                # Parse tempo data
                for tempo in track.findall("TEMPO"):
                    tempo_data = {
                        "inizio": tempo.get("Inizio", "0"),
                        "bpm": tempo.get("Bpm", "0"),
                        "metro": tempo.get("Metro", "4/4"),
                        "battito": tempo.get("Battito", "1")
                    }
                    track_data["tempo_data"].append(tempo_data)
                
                # Parse position marks (cue points)
                for mark in track.findall("POSITION_MARK"):
                    mark_data = {
                        "type": mark.get("Type", "0"),
                        "start": mark.get("Start", "0"),
                        "num": mark.get("Num", "0"),
                        "name": mark.get("Name", ""),
                        "color_id": mark.get("ColorID", "")
                    }
                    track_data["position_marks"].append(mark_data)
                
                # Generate a location-based hash for matching tracks between XMLs
                location = track_data["location"]
                if location:
                    # Extract filename from location
                    filename = os.path.basename(location.split("file://")[1] if "file://" in location else location)
                    track_data["filename"] = filename
                    tracks[track_id] = track_data
            
            return tracks
        except Exception as e:
            logger.error(f"Error parsing XML {xml_path}: {str(e)}")
            return {}
            
    def match_tracks(self):
        """Match tracks between original and processed XML by location/filename."""
        matched_tracks = {}
        
        # Create lookup by filename for processed tracks
        processed_by_filename = {track["filename"]: track_id 
                               for track_id, track in self.processed_tracks.items() 
                               if "filename" in track}
        
        # Match original tracks to processed tracks
        for original_id, original_track in self.original_tracks.items():
            if "filename" not in original_track:
                continue
                
            filename = original_track["filename"]
            if filename in processed_by_filename:
                processed_id = processed_by_filename[filename]
                matched_tracks[original_id] = processed_id
        
        return matched_tracks
    
    def compare_track_attribute(self, original_track, processed_track, attr_name, strict=True):
        """Compare a specific attribute between original and processed tracks."""
        original_value = original_track.get(attr_name)
        processed_value = processed_track.get(attr_name)
        
        if strict:
            return original_value == processed_value
        elif original_value and processed_value:
            # Simple fuzzy matching for non-strict comparisons
            return original_value.strip().lower() == processed_value.strip().lower()
        else:
            return False
    
    def compare_tempo_data(self, original_track, processed_track):
        """Compare tempo data between original and processed tracks."""
        original_tempos = original_track.get("tempo_data", [])
        processed_tempos = processed_track.get("tempo_data", [])
        
        if len(original_tempos) != len(processed_tempos):
            return False, f"Tempo count mismatch: {len(original_tempos)} vs {len(processed_tempos)}"
        
        # Sort by inizio for consistent comparison
        original_tempos.sort(key=lambda x: float(x["inizio"]))
        processed_tempos.sort(key=lambda x: float(x["inizio"]))
        
        for i, (orig_tempo, proc_tempo) in enumerate(zip(original_tempos, processed_tempos)):
            # Compare BPM with small tolerance
            orig_bpm = float(orig_tempo["bpm"])
            proc_bpm = float(proc_tempo["bpm"])
            
            if abs(orig_bpm - proc_bpm) > 0.1:  # Allow 0.1 BPM difference
                return False, f"BPM mismatch at index {i}: {orig_bpm} vs {proc_bpm}"
                
            # Compare position with small tolerance
            orig_inizio = float(orig_tempo["inizio"])
            proc_inizio = float(proc_tempo["inizio"])
            
            if abs(orig_inizio - proc_inizio) > 0.1:  # Allow 0.1s difference
                return False, f"Tempo position mismatch at index {i}: {orig_inizio} vs {proc_inizio}"
        
        return True, "Tempo data matched"
    
    def compare_position_marks(self, original_track, processed_track):
        """Compare position marks (cue points) between original and processed tracks."""
        original_marks = original_track.get("position_marks", [])
        processed_marks = processed_track.get("position_marks", [])
        
        if len(original_marks) != len(processed_marks):
            return False, f"Position mark count mismatch: {len(original_marks)} vs {len(processed_marks)}"
        
        # Sort by num for consistent comparison
        original_marks.sort(key=lambda x: int(x["num"]))
        processed_marks.sort(key=lambda x: int(x["num"]))
        
        for i, (orig_mark, proc_mark) in enumerate(zip(original_marks, processed_marks)):
            # Compare type
            if orig_mark["type"] != proc_mark["type"]:
                return False, f"Mark type mismatch at index {i}: {orig_mark['type']} vs {proc_mark['type']}"
            
            # Compare position with small tolerance
            orig_start = float(orig_mark["start"])
            proc_start = float(proc_mark["start"])
            
            if abs(orig_start - proc_start) > 0.1:  # Allow 0.1s difference
                return False, f"Mark position mismatch at index {i}: {orig_start} vs {proc_start}"
                
            # Compare name if present
            if orig_mark.get("name") and proc_mark.get("name") and orig_mark["name"] != proc_mark["name"]:
                return False, f"Mark name mismatch at index {i}: {orig_mark['name']} vs {proc_mark['name']}"
                
            # Compare color
            if orig_mark.get("color_id") != proc_mark.get("color_id"):
                return False, f"Mark color mismatch at index {i}: {orig_mark.get('color_id')} vs {proc_mark.get('color_id')}"
        
        return True, "Position marks matched"
        
    def validate(self):
        """Validate that all DJ metadata is preserved between XML files."""
        logger.info(f"Parsing original XML: {self.original_xml}")
        self.original_tracks = self.parse_xml(self.original_xml)
        
        logger.info(f"Parsing processed XML: {self.processed_xml}")
        self.processed_tracks = self.parse_xml(self.processed_xml)
        
        self.results["total_tracks_original"] = len(self.original_tracks)
        self.results["total_tracks_processed"] = len(self.processed_tracks)
        
        if not self.original_tracks:
            logger.error("No tracks found in original XML")
            return False
            
        if not self.processed_tracks:
            logger.error("No tracks found in processed XML")
            return False
            
        # Match tracks between XMLs
        logger.info("Matching tracks between XMLs...")
        matched_tracks = self.match_tracks()
        
        self.results["matched_tracks"] = len(matched_tracks)
        self.results["unmatched_tracks"] = len(self.original_tracks) - len(matched_tracks)
        
        logger.info(f"Found {len(matched_tracks)} matched tracks between XMLs")
        
        # Compare DJ metadata for matched tracks
        for original_id, processed_id in matched_tracks.items():
            original_track = self.original_tracks[original_id]
            processed_track = self.processed_tracks[processed_id]
            
            track_name = f"{original_track.get('artist', 'Unknown')} - {original_track.get('name', 'Unknown')}"
            logger.info(f"Validating track: {track_name}")
            
            # Compare basic attributes
            for attr in ["name", "artist", "album"]:
                if not self.compare_track_attribute(original_track, processed_track, attr, strict=False):
                    issue = f"{attr.capitalize()} mismatch: '{original_track.get(attr, '')}' vs '{processed_track.get(attr, '')}'"
                    self.results["track_issues"][track_name].append(issue)
            
            # Compare tempo data
            tempo_match, tempo_message = self.compare_tempo_data(original_track, processed_track)
            if not tempo_match:
                self.results["track_issues"][track_name].append(tempo_message)
            
            # Compare position marks
            marks_match, marks_message = self.compare_position_marks(original_track, processed_track)
            if not marks_match:
                self.results["track_issues"][track_name].append(marks_message)
            
            # Compare rating if present
            if "Rating" in original_track["attributes"] and "Rating" in processed_track["attributes"]:
                if original_track["attributes"]["Rating"] != processed_track["attributes"]["Rating"]:
                    issue = f"Rating mismatch: {original_track['attributes']['Rating']} vs {processed_track['attributes']['Rating']}"
                    self.results["track_issues"][track_name].append(issue)
            
            # Compare color if present
            if "Color" in original_track["attributes"] and "Color" in processed_track["attributes"]:
                if original_track["attributes"]["Color"] != processed_track["attributes"]["Color"]:
                    issue = f"Color mismatch: {original_track['attributes']['Color']} vs {processed_track['attributes']['Color']}"
                    self.results["track_issues"][track_name].append(issue)
        
        # Calculate success rate
        tracks_with_issues = len(self.results["track_issues"])
        if self.results["matched_tracks"] > 0:
            success_rate = (self.results["matched_tracks"] - tracks_with_issues) / self.results["matched_tracks"] * 100
            self.results["success_rate"] = round(success_rate, 2)
        
        return self.results["success_rate"] >= 95  # Require 95% success rate
    
    def generate_report(self, output_path=None):
        """Generate a report of the validation results."""
        total_issues = sum(len(issues) for issues in self.results["track_issues"].values())
        
        report = [
            "=" * 80,
            "REKORDBOX XML VALIDATION REPORT",
            "=" * 80,
            f"Original XML: {self.original_xml}",
            f"Processed XML: {self.processed_xml}",
            f"Total tracks in original XML: {self.results['total_tracks_original']}",
            f"Total tracks in processed XML: {self.results['total_tracks_processed']}",
            f"Matched tracks: {self.results['matched_tracks']}",
            f"Unmatched tracks: {self.results['unmatched_tracks']}",
            f"Tracks with issues: {len(self.results['track_issues'])}",
            f"Total issues found: {total_issues}",
            f"Success rate: {self.results['success_rate']}%",
            "-" * 80
        ]
        
        if self.results["track_issues"]:
            report.append("ISSUES BY TRACK:")
            report.append("-" * 80)
            
            for track_name, issues in self.results["track_issues"].items():
                report.append(f"Track: {track_name}")
                for issue in issues:
                    report.append(f"  - {issue}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")
        
        print(report_text)
        
        # Also save as JSON for programmatic use
        if output_path:
            json_path = os.path.splitext(output_path)[0] + ".json"
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self.results, f, indent=2)
                logger.info(f"JSON report saved to {json_path}")
            except Exception as e:
                logger.error(f"Error saving JSON report: {str(e)}")
                
        return self.results["success_rate"] >= 95

def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(description="Validate Rekordbox XML preservation")
    parser.add_argument("--original", required=True, help="Path to original Rekordbox XML export")
    parser.add_argument("--processed", required=True, help="Path to processed Rekordbox XML export")
    parser.add_argument("--report", help="Path to save validation report (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.original):
        logger.error(f"Original XML file not found: {args.original}")
        return False
        
    if not os.path.exists(args.processed):
        logger.error(f"Processed XML file not found: {args.processed}")
        return False
    
    validator = RekordboxValidator(args.original, args.processed)
    result = validator.validate()
    
    report_path = args.report or os.path.join(os.getcwd(), "rekordbox_validation_report.txt")
    validator.generate_report(report_path)
    
    return result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
