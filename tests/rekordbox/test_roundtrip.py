"""
Test the round-trip fidelity of Rekordbox XML import/export.

These tests verify that our Rekordbox implementation correctly preserves
all DJ-specific metadata during import and export operations.
"""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from djmusiccleaner.rekordbox.models import RekordboxTrack
from djmusiccleaner.rekordbox.xml_parser import RekordboxXMLParser
from djmusiccleaner.rekordbox.service import RekordboxService


class TestRekordboxRoundtrip(unittest.TestCase):
    """Test suite for Rekordbox round-trip preservation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal test XML
        self.xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="6.6.2" Company="AlphaTheta"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="1" Name="Test Track" Artist="Test Artist" Album="Test Album"
           TotalTime="240.5" DiscNumber="1" TrackNumber="1" BitRate="320" 
           SampleRate="44100" Tonality="12B" Rating="4" PlayCount="5" Color="pink"
           Location="file:///music/test.mp3">
      <TEMPO Inizio="0.025" Bpm="128.0" Metro="4/4" Battito="1"/>
      <TEMPO Inizio="120.025" Bpm="130.0" Metro="4/4" Battito="1"/>
      <POSITION_MARK Type="0" Start="0.0" Num="0" Name="Intro"/>
      <POSITION_MARK Type="1" Start="16.0" Num="1" Name="Verse" ColorID="1"/>
      <POSITION_MARK Type="1" Start="32.0" Num="2" Name="Drop" ColorID="2"/>
    </TRACK>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Name="ROOT" Type="0" Count="1">
      <NODE Name="Test Playlist" Type="1" Count="1">
        <TRACK Key="1"/>
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>'''

        # Write to temp file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.xml_path = os.path.join(self.temp_dir.name, "rekordbox.xml")
        with open(self.xml_path, 'w', encoding='utf-8') as f:
            f.write(self.xml_content)
            
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        
    def test_track_roundtrip(self):
        """Test that a track can be parsed and serialized without data loss."""
        # Parse XML
        parser = RekordboxXMLParser()
        parser.parse(self.xml_path)
        
        # Get the test track
        track = parser.tracks.get("1")
        self.assertIsNotNone(track)
        
        # Check basic fields
        self.assertEqual(track.title, "Test Track")
        self.assertEqual(track.artist, "Test Artist")
        self.assertEqual(track.key, "12B")
        self.assertEqual(track.rating, 4)
        self.assertEqual(track.color, "pink")
        self.assertEqual(track.play_count, 5)
        
        # Check tempo data
        self.assertEqual(len(track.tempo_data), 2)
        self.assertAlmostEqual(track.tempo_data[0].inizio, 0.025)
        self.assertAlmostEqual(track.tempo_data[0].bpm, 128.0)
        self.assertAlmostEqual(track.tempo_data[1].inizio, 120.025)
        self.assertAlmostEqual(track.tempo_data[1].bpm, 130.0)
        
        # Check position marks
        self.assertEqual(len(track.position_marks), 3)
        self.assertEqual(track.position_marks[0].name, "Intro")
        self.assertEqual(track.position_marks[0].type, 0)  # Memory point
        self.assertEqual(track.position_marks[1].type, 1)  # Hot cue
        self.assertEqual(track.position_marks[1].num, 1)
        self.assertEqual(track.position_marks[2].name, "Drop")
        
        # Now convert back to XML
        output_path = os.path.join(self.temp_dir.name, "output.xml")
        parser.export_xml(output_path)
        
        # Re-parse output to verify
        new_parser = RekordboxXMLParser()
        new_parser.parse(output_path)
        
        # Get the output track and verify all fields
        new_track = new_parser.tracks.get("1")
        self.assertIsNotNone(new_track)
        self.assertEqual(new_track.title, track.title)
        self.assertEqual(new_track.artist, track.artist)
        self.assertEqual(new_track.key, track.key)
        self.assertEqual(new_track.rating, track.rating)
        self.assertEqual(new_track.color, track.color)
        
        # Check tempo data was preserved
        self.assertEqual(len(new_track.tempo_data), len(track.tempo_data))
        self.assertAlmostEqual(new_track.tempo_data[0].bpm, track.tempo_data[0].bpm)
        self.assertAlmostEqual(new_track.tempo_data[1].bpm, track.tempo_data[1].bpm)
        
        # Check position marks were preserved
        self.assertEqual(len(new_track.position_marks), len(track.position_marks))
        self.assertEqual(new_track.position_marks[0].name, track.position_marks[0].name)
        self.assertEqual(new_track.position_marks[2].name, track.position_marks[2].name)
        
    def test_service_hash_matching(self):
        """Test that the service correctly matches tracks by hash."""
        # Create a test file
        test_file_path = os.path.join(self.temp_dir.name, "test.mp3")
        with open(test_file_path, 'wb') as f:
            f.write(b'TEST AUDIO DATA')  # Simple placeholder data
        
        # Initialize service and parse XML
        service = RekordboxService()
        service.import_collection(self.xml_path)
        
        # Track shouldn't match yet (location doesn't point to our test file)
        track = service.get_track_by_path(test_file_path)
        self.assertIsNone(track)
        
        # Now manually add a track with our test file and update track map
        track = list(service.parser.tracks.values())[0]
        track.file_path = test_file_path
        
        # Manually update track map
        file_hash = service._calculate_file_hash(test_file_path)
        service.track_map[file_hash] = track
        
        # Now it should match
        found_track = service.get_track_by_path(test_file_path)
        self.assertIsNotNone(found_track)
        self.assertEqual(found_track.track_id, "1")
        
    def test_location_update(self):
        """Test that track locations are updated correctly."""
        parser = RekordboxXMLParser()
        parser.parse(self.xml_path)
        
        # Get original track
        track = parser.tracks.get("1")
        original_location = track.location
        
        # Update location
        new_path = "/new/path/to/file.mp3"
        parser.update_track_location("1", new_path)
        
        # Export and re-import
        output_path = os.path.join(self.temp_dir.name, "updated.xml")
        parser.export_xml(output_path)
        
        # Re-parse
        new_parser = RekordboxXMLParser()
        new_parser.parse(output_path)
        
        # Check location was updated
        updated_track = new_parser.tracks.get("1")
        self.assertNotEqual(updated_track.location, original_location)
        self.assertTrue(updated_track.location.endswith("file.mp3"))
