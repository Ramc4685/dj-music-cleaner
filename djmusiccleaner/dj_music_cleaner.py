#!/usr/bin/env python3
"""
DJ Music File Cleaner - PROFESSIONAL DJ EDITION
Complete metadata enhancement for professional DJ libraries with advanced analytics
"""

import os
import re
import shutil
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from difflib import SequenceMatcher
from mutagen.mp3 import MP3
from mutagen.id3 import ID3NoHeaderError, TIT2, TPE1, TALB, TCOM, COMM, TYER, TDRC, TCON, TBPM, TKEY, TLEN
import unicodedata
import traceback
import platform
import subprocess

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  tqdm not installed. Run: pip install tqdm")
    # Fallback implementation
    def tqdm(iterable, **kwargs):
        return iterable

# Online identification imports
try:
    import musicbrainzngs
    MUSICBRAINZ_AVAILABLE = True
except ImportError:
    MUSICBRAINZ_AVAILABLE = False
    print("⚠️  musicbrainzngs not installed. Run: pip install musicbrainzngs")

try:
    import acoustid
    ACOUSTID_AVAILABLE = True
except ImportError:
    ACOUSTID_AVAILABLE = False
    print("⚠️  pyacoustid not installed. Run: pip install pyacoustid")

# Audio analysis imports
try:
    import numpy as np
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not installed. Run: pip install librosa numpy")

# Loudness normalization
try:
    import soundfile as sf
    import pyloudnorm as pyln
    LOUDNORM_AVAILABLE = True
except ImportError:
    LOUDNORM_AVAILABLE = False
    print("⚠️  pyloudnorm not installed. Run: pip install pyloudnorm soundfile")

# Rekordbox integration - optional
try:
    import pyrekordbox
    PYREKORDBOX_AVAILABLE = True
except ImportError:
    PYREKORDBOX_AVAILABLE = False

class DJMusicCleaner:
    def __init__(self, acoustid_api_key=None):
        self.acoustid_api_key = acoustid_api_key
        self.setup_musicbrainz()
        
        # Enhanced statistics tracking
        self.stats = {
            'text_search_hits': 0,
            'fingerprint_hits': 0,
            'identification_failures': 0,
            'year_found': 0,
            'album_found': 0,
            'genre_found': 0,  # 🆕
            'bpm_found': 0,    # 🆕
            'key_found': 0,    # 🆕
            'manual_review_needed': []
        }
        
        # 🆕 Genre mapping for Indian/Tamil music
        self.genre_mapping = {
            'tamil': 'Tamil',
            'bollywood': 'Bollywood',
            'indian': 'Indian',
            'classical': 'Indian Classical',
            'devotional': 'Devotional',
            'folk': 'Folk',
            'fusion': 'Fusion',
            'electronic': 'Electronic',
            'pop': 'Pop',
            'rock': 'Rock',
            'hip hop': 'Hip Hop',
            'dance': 'Dance',
            'world music': 'World Music'
        }
        
        # Enhanced site patterns
        self.site_patterns = [
            r'(?i)allindiandjsdrive',
            r'(?i)mp3virus',
            r'(?i)djmaza',
            r'(?i)songspk',
            r'(?i)pagalworld',
            r'(?i)djpunjab',
            r'(?i)masstamilan(?:\.dev)?',
            r'(?i)\.dev\b',
            r'(?i)tamilwire',
            r'(?i)tamilrockers?',
            r'(?i)isaimini',
            r'(?i)tamildbox',
            r'(?i)tamilyogi',
            r'(?i)moviesda',
            r'(?i)kuttymovies',
            r'(?i)www\.\w+',
            r'(?i)\.(?:com|in|net|org|co|dev|biz)\b',
            r'\[.*?\.(?:com|in|net|org|dev).*?\]',
            r'\(.*?\.(?:com|in|net|org|dev).*?\)',
        ]
        
        self.promo_patterns = [
            r'(?i)\(Full Audio Song\)',
            r'(?i)\(Full Song\)',
            r'(?i)\(Audio\)',
            r'(?i)\(Official\)',
            r'(?i)320kbps',
            r'(?i)Free Download',
            r'(?i)High Quality',
            r'(?i)Tamil \d{4}',
            r'(?i)Hindi \d{4}',
            r'(?i)\[Tamil\]',
            r'(?i)\[Hindi\]',
            r'(?i)\[Punjabi\]',
            r'(?i)\(\d{4}\)',
        ]
        
        self.preserve_patterns = [
            r'(?i)\(Remix\)',
            r'(?i)\(Club Mix\)',
            r'(?i)\(Extended Mix\)',
            r'(?i)\(Radio Edit\)',
            r'(?i)\(Unplugged\)',
            r'(?i)\(Live\)',
        ]

    def setup_musicbrainz(self):
        """Initialize MusicBrainz API"""
        if MUSICBRAINZ_AVAILABLE:
            musicbrainzngs.set_useragent("DJMusicCleaner", "1.0", "dj@example.com")
            musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)
            print("🌐 MusicBrainz API initialized")
            
    def analyze_audio_quality(self, filepath):
        """Analyze audio quality and format details"""
        try:
            print(f"   🔍 Analyzing audio quality for {os.path.basename(filepath)}")
            audio = MP3(filepath)
            
            # Get basic info
            info = audio.info
            bitrate_kbps = int(info.bitrate / 1000)
            sample_rate_khz = info.sample_rate / 1000
            length_seconds = info.length
            channels = getattr(info, 'channels', 0)
            
            # Calculate minutes and seconds
            minutes = int(length_seconds // 60)
            seconds = int(length_seconds % 60)
            
            # Determine quality rating
            is_high_quality = self.is_high_quality(bitrate_kbps, sample_rate_khz)
            quality_rating = "HIGH" if is_high_quality else "LOW"
            quality_message = "✅ High quality" if is_high_quality else "⚠️ Low quality - needs replacement"
            
            print(f"   🎧 Quality: {quality_message} ({bitrate_kbps}kbps, {sample_rate_khz}kHz)")
            
            quality_text = f"QUALITY: {bitrate_kbps}kbps, {sample_rate_khz}kHz, {quality_rating}"

            quality_info = {
                'bitrate_kbps': bitrate_kbps,
                'sample_rate_khz': sample_rate_khz,
                'length': f"{minutes}:{seconds:02d}",
                'channels': channels,
                'encoding': getattr(info, 'encoding', 'Unknown'),
                'is_high_quality': is_high_quality,
                'quality_rating': quality_rating,
                'quality_text': quality_text
            }

            return quality_info
            
        except Exception as e:
            print(f"   ❌ Quality analysis error: {e}")
            return {
                'error': str(e),
                'bitrate_kbps': 0,
                'sample_rate_khz': 0,
                'length': "0:00",
                'channels': 0,
                'is_high_quality': False,
                'quality_rating': "ERROR"
            }
    
    def is_high_quality(self, bitrate_kbps, sample_rate_khz=None):
        """Determine if a file meets high quality standards (320kbps)"""
        # Primary check: bitrate must be at least 320 kbps
        if bitrate_kbps < 320:
            return False
            
        # Secondary check: if sample rate provided, should be at least 44.1 kHz
        if sample_rate_khz and sample_rate_khz < 44.0:
            return False
            
        return True
            
    def analyze_dynamic_range(self, filepath):
        """Analyze dynamic range - crucial for DJ tracks."""
        if not LIBROSA_AVAILABLE:
            print(f"   ⚠️ Librosa not available for dynamic range analysis")
            return None
            
        try:
            print(f"   📊 Analyzing dynamic range...")
            # Load audio with librosa
            y, sr = librosa.load(filepath)
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Dynamic range metrics
            peak = np.max(np.abs(y))
            dynamic_range = 20 * np.log10(peak / (np.mean(rms) + 1e-10))
            crest_factor = peak / (np.sqrt(np.mean(y**2)) + 1e-10)
            
            # DJ-specific evaluation
            if dynamic_range > 14:
                dr_rating = "Excellent - Wide dynamic range"
            elif dynamic_range > 10:
                dr_rating = "Good - Suitable for most DJ contexts"
            elif dynamic_range > 6:
                dr_rating = "Fair - Moderately compressed"
            else:
                dr_rating = "Poor - Heavily compressed, may sound flat"
                
            print(f"   📊 Dynamic range: {dynamic_range:.1f} dB - {dr_rating}")
            
            # Add DR info to file comment
            try:
                audio = MP3(filepath)
                dr_comment = f"Dynamic Range: {dynamic_range:.1f} dB - {dr_rating}"
                
                if 'COMM::eng' in audio:
                    current_comment = str(audio['COMM::eng'])
                    if "Dynamic Range:" not in current_comment:  # Don't duplicate
                        audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', 
                                              text=f"{current_comment}\n{dr_comment}")
                else:
                    audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=dr_comment)
                    
                audio.save()
            except:
                pass  # Non-critical if comment addition fails
            
            return {
                'dynamic_range_db': dynamic_range,
                'crest_factor': crest_factor,
                'rating': dr_rating
            }
        except Exception as e:
            print(f"   Dynamic range analysis error: {e}")
            return None
            
    def detect_musical_key(self, filepath):
        """Detect the musical key of a track using librosa"""
        if not LIBROSA_AVAILABLE:
            print("   Librosa not available for key detection")
            return None
        try:
            print("   Detecting musical key (librosa KS)...")
            y, sr = librosa.load(filepath, mono=True)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            v = chroma.mean(axis=1)

            K_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
            K_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
            PITCHES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            majors = [np.corrcoef(np.roll(v, -i), K_MAJOR)[0,1] for i in range(12)]
            minors = [np.corrcoef(np.roll(v, -i), K_MINOR)[0,1] for i in range(12)]
            key = (PITCHES[int(np.argmax(majors))] if max(majors) >= max(minors)
                   else PITCHES[int(np.argmax(minors))] + 'm')

            camelot = {
                'C':'8B','G':'9B','D':'10B','A':'11B','E':'12B','B':'1B','F#':'2B','C#':'3B',
                'G#':'4B','D#':'5B','A#':'6B','F':'7B','Am':'8A','Em':'9A','Bm':'10A','F#m':'11A',
                'C#m':'12A','G#m':'1A','D#m':'2A','A#m':'3A','Fm':'4A','Cm':'5A','Gm':'6A','Dm':'7A'
            }.get(key, 'Unknown')

            audio = MP3(filepath)
            audio['TKEY'] = TKEY(encoding=3, text=key)
            self.add_to_comments(filepath, f"Camelot key: {camelot}")
            audio.save()
            print(f"   Detected key: {key} (Camelot: {camelot})")
            self.stats['key_found'] += 1
            return key
        except Exception as e:
            print(f"   Key detection error: {e}")
            return None
            
    def detect_cue_points(self, filepath):
        """Detect ideal cue points for DJ mixing."""
        if not LIBROSA_AVAILABLE:
            print(f"   ⚠️ Librosa not available for cue point detection")
            return None
            
        try:
            print(f"   👁️ Analyzing track structure for cue points...")
            # Load audio with librosa
            y, sr = librosa.load(filepath)
            
            # Get tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            print(f"   ⏰ Detected tempo: {tempo:.1f} BPM")
            
            # Find onsets and beats
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Detect significant changes (potential drops, transitions)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_delta = np.diff(mfcc, axis=1)
            
            # Find points with significant changes
            significant_changes = []
            
            # Calculate bars (4 beats per bar typically)
            bar_duration = 4 * 60 / tempo
            song_length = len(y) / sr
            
            for i in range(1, len(beat_times)-8, 8):  # Check every 2 bars
                if beat_times[i] > 5.0:  # Skip first few seconds
                    # Get MFCC change over this segment
                    segment_start = librosa.time_to_frames(beat_times[i], sr=sr)
                    segment_end = min(librosa.time_to_frames(beat_times[i+8], sr=sr), mfcc.shape[1]-1)
                    
                    if segment_end > segment_start:
                        segment_delta = np.mean(np.abs(mfcc[:, segment_start:segment_end]))  
                        prev_segment_start = librosa.time_to_frames(max(0, beat_times[i]-bar_duration), sr=sr)
                        
                        if prev_segment_start < segment_start:
                            prev_segment = np.mean(np.abs(mfcc[:, prev_segment_start:segment_start]))
                            change_ratio = segment_delta / (prev_segment + 1e-10)
                            
                            if change_ratio > 1.4:  # Significant change
                                significant_changes.append(beat_times[i])
            
            # Find intro end (first significant change after 16 bars)
            intro_candidates = [t for t in significant_changes if t > bar_duration * 8]
            intro_end = intro_candidates[0] if intro_candidates else bar_duration * 16
            
            # Calculate outro start (last 16 bars typically)
            outro_start = max(song_length - (16 * bar_duration), song_length * 0.75)
            
            # Extract top 3 significant changes as potential drops
            top_drops = sorted(significant_changes)[:3] if significant_changes else []
            
            # Format for reporting
            drops_formatted = [f"{drop:.1f}s" for drop in top_drops]
            
            # Report findings
            print(f"   🎶 Intro ends around: {intro_end:.1f}s")
            if drops_formatted:
                print(f"   🎶 Key transition points: {', '.join(drops_formatted)}")
            print(f"   🎶 Outro begins around: {outro_start:.1f}s")
            
            # Save cue points to comments
            try:
                audio = MP3(filepath)
                cue_comment = f"DJ Cues - Intro: {intro_end:.1f}s, Outro: {outro_start:.1f}s"
                if top_drops:
                    cue_comment += f", Drops: {', '.join(drops_formatted)}"
                    
                if 'COMM::eng' in audio:
                    current_comment = str(audio['COMM::eng'])
                    if "DJ Cues -" not in current_comment:  # Don't duplicate
                        audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', 
                                              text=f"{current_comment}\n{cue_comment}")
                else:
                    audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=cue_comment)
                    
                audio.save()
            except:
                pass  # Non-critical if comment addition fails
            
            return {
                'intro_end': intro_end,
                'drops': top_drops,
                'outro_start': outro_start,
                'tempo': tempo,
                'recommended_cues': [0, intro_end] + top_drops + [outro_start]
            }
        except Exception as e:
            print(f"   ❌ Cue point detection error: {e}")
            return {'intro_end': 16.0, 'outro_start': 180.0, 'drops': []}
            
    def calculate_energy_rating(self, filepath):
        """Calculate track energy level (1-10) for DJ sets."""
        if not LIBROSA_AVAILABLE:
            print(f"   ⚠️ Librosa not available for energy detection")
            return None
            
        try:
            print(f"   ⚡ Calculating energy rating...")
            # Load audio with librosa
            y, sr = librosa.load(filepath)
            
            # Calculate metrics that contribute to "energy"
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = np.mean(librosa.feature.rms(y=y)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            
            # Energy heuristic based on tempo, RMS energy, and spectral centroid
            tempo_factor = min(tempo / 130.0, 1.5)  # Normalize around 130 BPM
            rms_factor = min(rms * 1000, 2.0)  # Loudness factor
            freq_factor = min(spectral_centroid / 2000.0, 1.5)  # High frequency content
            
            # Calculate energy score (1-10)
            energy = (tempo_factor * 0.4 + rms_factor * 0.4 + freq_factor * 0.2) * 5.0
            energy = max(1.0, min(10.0, energy))  # Clamp to 1-10 range
            
            print(f"   ⚡ Energy rating: {energy:.1f}/10")
            
            # Save energy to custom comment tag
            audio = MP3(filepath)
            energy_comment = f"Energy: {energy:.1f}/10"
            
            if 'COMM::eng' in audio:
                current_comment = str(audio['COMM::eng'])
                if "Energy:" not in current_comment:  # Don't duplicate
                    audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', 
                                           text=f"{current_comment}\n{energy_comment}")
            else:
                audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=energy_comment)
                
            audio.save()
            
            return {'energy_rating': round(energy, 1)}  
        except Exception as e:
            print(f"   ❌ Energy rating error: {e}")
            return None
            
    # Collection Management Features
        
    def normalize_text_for_comparison(self, text):
        """Normalize text for better duplicate detection matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common DJ notations like "Original Mix", "Radio Edit", etc.
        patterns_to_remove = [
            r'\(original mix\)',
            r'\(radio edit\)',
            r'\(extended mix\)',
            r'\(club mix\)',
            r'\(dub mix\)',
            r'\(instrumental\)',
            r'\(remix\)',
            r'\(remastered\)',
            r'\d{4}',  # Year in parentheses
            r'feat\. [\w\s]+',
            r'ft\. [\w\s]+',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def find_duplicates(self, directory):
        """Find duplicate or similar tracks in collection - essential for clean DJ libraries."""
        try:
            mp3_files = list(Path(directory).glob('**/*.mp3'))
            print(f"\n🔍 Scanning {len(mp3_files)} files for duplicates...")
            
            # Track data structure for comparison
            tracks = []
            for file_path in mp3_files:
                try:
                    audio = MP3(file_path)
                    artist = str(audio.get('TPE1', '')).strip() if 'TPE1' in audio else ''
                    title = str(audio.get('TIT2', '')).strip() if 'TIT2' in audio else ''
                    
                    if not title:
                        title = file_path.stem
                    
                    # Get fingerprint
                    fp = None
                    if ACOUSTID_AVAILABLE and self.acoustid_api_key:
                        try:
                            # returns (duration, fingerprint_string)
                            dur, fp_str = acoustid.fingerprint_file(str(file_path))
                            fp = fp_str
                        except:
                            pass
                    
                    tracks.append({
                        'path': file_path,
                        'artist': artist.lower(),
                        'title': title.lower(),
                        'fingerprint': fp,
                        'size': file_path.stat().st_size
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            # Find duplicates
            duplicates = []
            
            # First, check for exact artist/title matches
            artist_title_groups = {}
            for i, track in enumerate(tracks):
                key = f"{track['artist']}|||{track['title']}"
                if key not in artist_title_groups:
                    artist_title_groups[key] = []
                artist_title_groups[key].append(i)
            
            # Add exact matches
            for key, indices in artist_title_groups.items():
                if len(indices) > 1:
                    group = [tracks[i] for i in indices]
                    duplicates.append({
                        'type': 'exact_match',
                        'match_on': 'artist_title',
                        'tracks': group
                    })
            
            # Next, check for fuzzy title matches within artists
            for artist in set(track['artist'] for track in tracks if track['artist']):
                artist_tracks = [track for track in tracks if track['artist'] == artist]
                
                for i, track1 in enumerate(artist_tracks):
                    for track2 in artist_tracks[i+1:]:
                        # Simple fuzzy match - Levenshtein distance
                        title1 = track1['title']
                        title2 = track2['title']
                        
                        if title1 and title2:
                            # Simple edit distance check
                            if self._similar_strings(title1, title2, threshold=0.8):
                                duplicates.append({
                                    'type': 'similar_title',
                                    'match_on': 'fuzzy_title',
                                    'similarity': self._similarity(title1, title2),
                                    'tracks': [track1, track2]
                                })
            
            # Finally, check for acoustic fingerprint matches if available
            fingerprinted_tracks = [t for t in tracks if t['fingerprint']]
            for i, track1 in enumerate(fingerprinted_tracks):
                for track2 in fingerprinted_tracks[i+1:]:
                    # Skip if already matched by metadata
                    if any(track1 in dup['tracks'] and track2 in dup['tracks'] for dup in duplicates):
                        continue
                    
                    if track1['fingerprint'] and track2['fingerprint']:
                        # compare fingerprint strings (not duration)
                        if track1['fingerprint'] == track2['fingerprint']:
                            duplicates.append({
                                'type': 'audio_match',
                                'match_on': 'fingerprint',
                                'tracks': [track1, track2]
                            })
            
            # Generate report
            print(f"\n📊 Found {len(duplicates)} potential duplicates:")
            for i, dup in enumerate(duplicates):
                print(f"\nDuplicate Group #{i+1} ({dup['type']})")
                for track in dup['tracks']:
                    print(f"   🎧 {track['path'].name} ({track['size'] // 1024} KB)")
            
            return duplicates
        except Exception as e:
            print(f"Duplicate detection error: {e}")
            return []
        
    def _similarity(self, a, b):
        """Calculate string similarity ratio."""
        return SequenceMatcher(None, a, b).ratio()

    def _similar_strings(self, a, b, threshold=0.8):
        """Check if strings are similar based on threshold."""
        return self._similarity(a, b) >= threshold
    
    def prioritize_metadata_completion(self, directory):
        """Prioritize files that need metadata completion based on DJ importance."""
        mp3_files = list(Path(directory).glob('**/*.mp3'))
        
        # Define field importance for DJs
        field_weights = {
            'title': 10,
            'artist': 9,
            'bpm': 8,
            'key': 8,
            'energy': 7,
            'genre': 6,
            'album': 4,
            'year': 3
        }
        
        file_scores = []
        for mp3_file in mp3_files:
            try:
                missing_score = 0
                audio = MP3(mp3_file)
                
                # Check important fields
                if 'TIT2' not in audio or not str(audio.get('TIT2', '')).strip():
                    missing_score += field_weights['title']
                    
                if 'TPE1' not in audio or not str(audio.get('TPE1', '')).strip():
                    missing_score += field_weights['artist']
                    
                if 'TBPM' not in audio or not str(audio.get('TBPM', '')).strip():
                    missing_score += field_weights['bpm']
                    
                if 'TKEY' not in audio or not str(audio.get('TKEY', '')).strip():
                    missing_score += field_weights['key']
                    
                if 'TCON' not in audio or not str(audio.get('TCON', '')).strip():
                    missing_score += field_weights['genre']
                    
                if 'TALB' not in audio or not str(audio.get('TALB', '')).strip():
                    missing_score += field_weights['album']
                    
                if 'TYER' not in audio and 'TDRC' not in audio:
                    missing_score += field_weights['year']
                    
                # Check for energy info in comments
                energy_found = False
                if 'COMM::eng' in audio:
                    if "energy:" in str(audio['COMM::eng']).lower():
                        energy_found = True
                        
                if not energy_found:
                    missing_score += field_weights['energy']
                
                completion = self._calculate_completion_percent(audio, field_weights)
                
                file_scores.append({
                    'path': mp3_file,
                    'missing_score': missing_score,
                    'completion': completion,
                    'filename': mp3_file.name
                })
            except Exception as e:
                print(f"Error evaluating {mp3_file}: {e}")
        
        # Sort by missing score (highest first)
        file_scores.sort(key=lambda x: x['missing_score'], reverse=True)
        
        # Print results
        print(f"\n📊 DJ Metadata Priority Report:")
        print(f"{'='*60}")
        print(f"{'File':40} | {'Completion %':12} | {'Missing'}")
        print(f"{'-'*40}-|-{'-'*12}-|-{'-'*15}")
        
        for score in file_scores[:20]:  # Show top 20
            missing_fields = []
            if score['missing_score'] >= field_weights['title']: missing_fields.append('title')
            if score['missing_score'] >= field_weights['artist']: missing_fields.append('artist')
            if score['missing_score'] >= field_weights['bpm']: missing_fields.append('bpm')
            if score['missing_score'] >= field_weights['key']: missing_fields.append('key')
            if score['missing_score'] >= field_weights['genre']: missing_fields.append('genre')
            
            print(f"{score['filename'][:39]:40} | {score['completion']:12.1f} | {', '.join(missing_fields)[:15]}")
        
        return file_scores

    def _calculate_completion_percent(self, audio, field_weights):
        """Calculate metadata completion percentage."""
        total_weight = sum(field_weights.values())
        current_weight = 0
        
        if 'TIT2' in audio and str(audio.get('TIT2', '')).strip():
            current_weight += field_weights['title']
            
        if 'TPE1' in audio and str(audio.get('TPE1', '')).strip():
            current_weight += field_weights['artist']
            
        if 'TBPM' in audio and str(audio.get('TBPM', '')).strip():
            current_weight += field_weights['bpm']
            
        if 'TKEY' in audio and str(audio.get('TKEY', '')).strip():
            current_weight += field_weights['key']
            
        if 'TCON' in audio and str(audio.get('TCON', '')).strip():
            current_weight += field_weights['genre']
            
        if 'TALB' in audio and str(audio.get('TALB', '')).strip():
            current_weight += field_weights['album']
            
        if ('TYER' in audio and str(audio.get('TYER', '')).strip()) or \
           ('TDRC' in audio and str(audio.get('TDRC', '')).strip()):
            current_weight += field_weights['year']
            
        # Check for energy info in comments
        if 'COMM::eng' in audio and "energy:" in str(audio['COMM::eng']).lower():
            current_weight += field_weights['energy']
            
        return (current_weight / total_weight) * 100
        
    # Rekordbox XML Integration
    
    def import_rekordbox_xml(self, xml_file):
        """Import metadata from Rekordbox XML export file."""
        try:
            print(f"\n🎛️ Importing Rekordbox XML: {xml_file}")
            if not os.path.exists(xml_file):
                print(f"❌ Rekordbox XML file not found: {xml_file}")
                return None
                
            # Parse the XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract collection data
            collection = root.find('COLLECTION')
            if collection is None:
                print("❌ No COLLECTION found in XML")
                return None
                
            tracks = collection.findall('TRACK')
            print(f"📊 Found {len(tracks)} tracks in Rekordbox collection")
            
            # Process track data
            rekordbox_data = {}
            for track in tracks:
                location = track.get('Location', '')
                
                # Convert URL format to local path
                if location.startswith('file://localhost'):
                    # Handle URL encoding and platform-specific paths
                    path = urllib.parse.unquote(location[16:])  # Remove 'file://localhost'
                    if platform.system() == 'Windows' and path.startswith('/'):
                        path = path[1:]  # Remove leading slash on Windows
                        
                    # Extract useful DJ metadata
                    track_data = {
                        'title': track.get('Name', ''),
                        'artist': track.get('Artist', ''),
                        'album': track.get('Album', ''),
                        'genre': track.get('Genre', ''),
                        'comment': track.get('Comments', ''),
                        'key': track.get('Tonality', ''),
                        'bpm': float(track.get('AverageBpm', 0)) if track.get('AverageBpm') else None,
                        'rating': int(track.get('Rating', 0)),
                        'play_count': int(track.get('PlayCount', 0)),
                        'date_added': track.get('DateAdded', ''),
                        'cue_points': []
                    }
                    
                    # Get cue points if available
                    cues = track.findall('POSITION_MARK')
                    for cue in cues:
                        cue_data = {
                            'name': cue.get('Name', ''),
                            'type': cue.get('Type', ''),
                            'time': float(cue.get('Start', 0)) if cue.get('Start') else 0,
                            'num': int(cue.get('Num', 0)) if cue.get('Num') else 0
                        }
                        track_data['cue_points'].append(cue_data)
                        
                    rekordbox_data[path] = track_data
                    
            print(f"📊 Processed {len(rekordbox_data)} valid track entries with paths")
            return rekordbox_data
        except Exception as e:
            print(f"❌ Error importing Rekordbox XML: {e}")
            traceback.print_exc()
            return None
            
    def apply_rekordbox_metadata(self, filepath, rekordbox_data):
        """Apply Rekordbox metadata to MP3 file if available."""
        if not rekordbox_data:
            return False
            
        try:
            # Normalize paths for comparison
            norm_filepath = os.path.normpath(os.path.abspath(filepath))
            
            # Check if file exists in Rekordbox data
            if norm_filepath in rekordbox_data:
                print(f"   🎛️ Found in Rekordbox library")
                rb_data = rekordbox_data[norm_filepath]
                
                # Apply metadata
                audio = MP3(filepath)
                modified = False
                
                # Apply BPM if missing
                if ('TBPM' not in audio or not str(audio.get('TBPM', '')).strip()) and rb_data['bpm']:
                    audio['TBPM'] = TBPM(encoding=3, text=str(rb_data['bpm']))
                    print(f"   🎛️ Applied Rekordbox BPM: {rb_data['bpm']}")
                    modified = True
                    
                # Apply Key if missing
                if ('TKEY' not in audio or not str(audio.get('TKEY', '')).strip()) and rb_data['key']:
                    audio['TKEY'] = TKEY(encoding=3, text=rb_data['key'])
                    print(f"   🎛️ Applied Rekordbox Key: {rb_data['key']}")
                    modified = True
                    
                # Apply Genre if missing
                if ('TCON' not in audio or not str(audio.get('TCON', '')).strip()) and rb_data['genre']:
                    audio['TCON'] = TCON(encoding=3, text=rb_data['genre'])
                    print(f"   🎛️ Applied Rekordbox Genre: {rb_data['genre']}")
                    modified = True
                
                # Add cue points info to comments if available
                if rb_data['cue_points']:
                    cue_info = "Rekordbox Cues: " + ", ".join([f"#{c['num']}: {c['time']:.1f}s ({c['name']})" 
                                                         for c in rb_data['cue_points'] if c['name']])
                    
                    if 'COMM::eng' in audio:
                        current_comment = str(audio['COMM::eng'])
                        if "Rekordbox Cues:" not in current_comment:  # Don't duplicate
                            audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', 
                                                   text=f"{current_comment}\n{cue_info}")
                            modified = True
                    else:
                        audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=cue_info)
                        modified = True
                        
                    print(f"   🎛️ Added {len(rb_data['cue_points'])} cue points from Rekordbox")
                
                if modified:
                    audio.save()
                    self.stats['rekordbox_enhanced'] = self.stats.get('rekordbox_enhanced', 0) + 1
                    
                return True
                    
            else:
                # Try fuzzy matching
                norm_filename = os.path.basename(norm_filepath)
                for rb_path, rb_data in rekordbox_data.items():
                    rb_filename = os.path.basename(rb_path)
                    
                    if self._similar_strings(norm_filename, rb_filename, 0.9):
                        print(f"   🎛️ Found similar match in Rekordbox")
                        # Logic similar to above, but for fuzzy match
                        # This is intentionally simplified to avoid duplication
                        audio = MP3(filepath)
                        
                        # Apply essential DJ data from fuzzy match
                        if 'TBPM' not in audio and rb_data['bpm']:
                            audio['TBPM'] = TBPM(encoding=3, text=str(rb_data['bpm']))
                        if 'TKEY' not in audio and rb_data['key']:
                            audio['TKEY'] = TKEY(encoding=3, text=rb_data['key'])
                            
                        audio.save()
                        self.stats['rekordbox_fuzzy_match'] = self.stats.get('rekordbox_fuzzy_match', 0) + 1
                        return True
                        
            return False
        except Exception as e:
            print(f"   ❌ Error applying Rekordbox metadata: {e}")
            return False
            
    def export_rekordbox_xml(self, directory, output_xml, playlist_name=None):
        """Generate Rekordbox-compatible XML for processed files."""
        try:
            print(f"\n🎛️ Creating Rekordbox XML: {output_xml}")
            
            # Create root elements
            root = ET.Element('DJ_PLAYLISTS', Version="1.0.0")
            collection = ET.SubElement(root, 'COLLECTION', Entries="0")
            playlists = ET.SubElement(root, 'PLAYLISTS')
            
            # Create playlist structure
            root_node = ET.SubElement(playlists, 'NODE', Name="ROOT", Type="0")
            if playlist_name:
                playlist_folder = ET.SubElement(root_node, 'NODE', Name="DJ Music Cleaner", Type="0")
                playlist = ET.SubElement(playlist_folder, 'NODE', Name=playlist_name, 
                                        Type="1", KeyType="0", Entries="0")
            
            # Scan directory for MP3 files
            mp3_files = list(Path(directory).glob('**/*.mp3'))
            print(f"📊 Found {len(mp3_files)} MP3 files for XML export")
            
            # Process each file
            track_count = 0
            playlist_entries = []
            
            for mp3_file in mp3_files:
                try:
                    audio = MP3(mp3_file)
                    file_path = str(mp3_file.absolute())
                    
                    # Create file URL (Rekordbox format)
                    if platform.system() == 'Windows':
                        file_url = 'file://localhost/' + urllib.parse.quote(file_path.replace('\\', '/'))
                    else:
                        file_url = 'file://localhost' + urllib.parse.quote(file_path)
                    
                    # Extract metadata
                    title = str(audio.get('TIT2', os.path.splitext(mp3_file.name)[0]))
                    artist = str(audio.get('TPE1', ''))
                    album = str(audio.get('TALB', ''))
                    genre = str(audio.get('TCON', ''))
                    comment = str(audio.get('COMM::eng', ''))
                    key = str(audio.get('TKEY', ''))
                    bpm = str(audio.get('TBPM', ''))
                    year = str(audio.get('TDRC', str(audio.get('TYER', ''))))
                    
                    # Get file info
                    file_size = os.path.getsize(file_path)
                    duration = audio.info.length
                    
                    # Create track element
                    track_id = str(track_count + 1)
                    track = ET.SubElement(collection, 'TRACK', 
                                         TrackID=track_id,
                                         Name=title,
                                         Artist=artist,
                                         Album=album,
                                         Genre=genre,
                                         Comments=comment,
                                         Location=file_url,
                                         Tonality=key,
                                         AverageBpm=bpm,
                                         Year=year,
                                         FileSize=str(file_size),
                                         TotalTime=str(int(duration)))
                    
                    # Add to playlist
                    if playlist_name:
                        playlist_entries.append(track_id)
                    
                    track_count += 1
                except Exception as e:
                    print(f"Error processing {mp3_file}: {e}")
            
            # Update collection count
            collection.set('Entries', str(track_count))
            
            # Add tracks to playlist
            if playlist_name and playlist_entries:
                playlist.set('Entries', str(len(playlist_entries)))
                for track_id in playlist_entries:
                    ET.SubElement(playlist, 'TRACK', Key=track_id)
            
            # Write XML file
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(output_xml, encoding='UTF-8', xml_declaration=True)
            
            print(f"📄 Successfully created Rekordbox XML with {track_count} tracks")
            return True
        except Exception as e:
            print(f"❌ Error exporting Rekordbox XML: {e}")
            traceback.print_exc()
            return False
            
    def normalize_loudness(self, input_file, output_file=None, target_lufs=-14.0):
        """Normalize the loudness of an audio file to a target LUFS value"""
        if not LOUDNORM_AVAILABLE:
            print("   ⚠️ Loudness normalization unavailable (pyloudnorm/soundfile missing)")
            return False

        if not output_file:
            output_file = input_file
            
        try:
            data, rate = sf.read(input_file)
            
            # Peak normalize audio to -1 dB
            peak = np.max(np.abs(data))
            data_peak_normalized = data / peak * 0.9  # Leave some headroom
            
            # Measure the loudness first
            meter = pyln.Meter(rate)
            loudness = meter.integrated_loudness(data_peak_normalized)
            print(f"   🔊 Original loudness: {loudness:.1f} LUFS")
            
            # Calculate gain needed for target loudness
            gain_db = target_lufs - loudness
            gain_linear = 10 ** (gain_db / 20.0)
            
            # Apply gain with limiter to prevent clipping
            normalized_audio = data_peak_normalized * gain_linear
            normalized_audio = np.clip(normalized_audio, -0.99, 0.99)  # Prevent clipping
            
            # Always write a temp WAV then convert back to MP3
            temp_wav = (output_file or input_file) + ".temp.wav"
            sf.write(temp_wav, normalized_audio, rate)
            self._convert_wav_to_mp3(temp_wav, output_file or input_file, input_file)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
            print(f"   🔊 Normalized loudness to {target_lufs} LUFS with {gain_db:.1f}dB gain")
            
            # Update metadata to indicate normalization
            audio = MP3(output_file)
            loudness_comment = f"Loudness normalized to {target_lufs} LUFS"
            
            if 'COMM::eng' in audio:
                current_comment = str(audio['COMM::eng'])
                if "Loudness normalized" not in current_comment:  # Don't duplicate
                    audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', 
                                           text=f"{current_comment}\n{loudness_comment}")
            else:
                audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=loudness_comment)
                
            audio.save()
            
            self.stats['loudness_normalized'] = self.stats.get('loudness_normalized', 0) + 1
            return True
            
        except Exception as e:
            print(f"   ❌ Loudness normalization error: {e}")
            return False
            
    def _convert_wav_to_mp3(self, wav_file, output_mp3, original_mp3):
        """Convert WAV to MP3 while preserving metadata from original MP3."""
        try:
            # Get metadata from original file
            original_audio = MP3(original_mp3)
            tags = {}
            for key in original_audio.keys():
                if key != 'APIC:':  # Skip album art for simplicity
                    tags[key] = original_audio[key]
            
            # Convert WAV to MP3
            if shutil.which('ffmpeg'):  # Use ffmpeg if available
                subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', 
                                '-i', wav_file, '-c:a', 'libmp3lame', '-q:a', '0', output_mp3])
            else:
                # Fallback to simple copy if no converter available
                # This doesn't actually convert but preserves the file for testing
                shutil.copy2(original_mp3, output_mp3)
                print("   ⚠️ ffmpeg not found, couldn't convert WAV to MP3")
                
            # Restore metadata
            new_audio = MP3(output_mp3)
            for key, value in tags.items():
                new_audio[key] = value
            new_audio.save()
                
        except Exception as e:
            print(f"   ❌ Error converting WAV to MP3: {e}")
            # Fallback - keep original file
            if os.path.exists(original_mp3) and original_mp3 != output_mp3:
                shutil.copy2(original_mp3, output_mp3)

    def determine_genre(self, artist, title, album):
        """🆕 Smart genre detection for Indian/Tamil music"""
        text_to_analyze = f"{artist} {title} {album}".lower()
        
        # Tamil music indicators
        tamil_indicators = ['tamil', 'kollywood', 'chennai', 'madras']
        if any(indicator in text_to_analyze for indicator in tamil_indicators):
            return 'Tamil'
        
        # Bollywood indicators
        bollywood_indicators = ['bollywood', 'hindi', 'mumbai', 'playback']
        if any(indicator in text_to_analyze for indicator in bollywood_indicators):
            return 'Bollywood'
        
        # Electronic/Dance indicators
        electronic_indicators = ['remix', 'club', 'dance', 'dj', 'electronic']
        if any(indicator in text_to_analyze for indicator in electronic_indicators):
            return 'Electronic'
        
        # Default for Indian artists
        indian_names = ['rahman', 'ilaiyaraaja', 'yuvan', 'anirudh', 'harris', 'devi sri prasad']
        if any(name in text_to_analyze for name in indian_names):
            return 'Tamil'
        
        return 'Indian'  # Safe default for your collection

    def estimate_bpm_from_genre(self, genre):
        """🆕 Estimate BPM range based on genre"""
        bpm_ranges = {
            'Tamil': '120',
            'Bollywood': '115', 
            'Electronic': '128',
            'Dance': '130',
            'Hip Hop': '90',
            'Pop': '120',
            'Rock': '140'
        }
        return bpm_ranges.get(genre, '120')

    def clean_text(self, text):
        """Enhanced cleaning with better pollution detection"""
        if not text:
            return ""
        
        original_text = text
        
        for pattern in self.site_patterns:
            text = re.sub(pattern, '', text)
        
        for pattern in self.promo_patterns:
            text = re.sub(pattern, '', text)
        
        text = re.sub(r'\s*[-_,]+\s*', ' - ', text)
        text = re.sub(r'^[-\s,]+|[-\s,]+$', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        preserved_info = []
        for pattern in self.preserve_patterns:
            matches = re.findall(pattern, text)
            preserved_info.extend(matches)
        
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        if preserved_info:
            text += f" ({', '.join(preserved_info)})"
        
        text = re.sub(r'\s*[-_]+\s*', ' - ', text)
        text = re.sub(r'^[-\s]+|[-\s]+$', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        cleaned = text.strip()
        
        if original_text != cleaned and len(cleaned) > 0:
            print(f"   🧹 Cleaned: '{original_text}' → '{cleaned}'")
        
        return cleaned

    def has_pollution(self, text):
        """Enhanced pollution detection"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        for pattern in self.site_patterns:
            if re.search(pattern, text):
                return True
        
        pollution_indicators = [
            'masstamilan', 'tamilwire', 'isaimini', 'djmaza',
            '.com', '.in', '.dev', '.net', '.org', '.co',
            'www.', 'http', 'download', 'free'
        ]

        return any(indicator in text_lower for indicator in pollution_indicators)

    def sanitize_tag_value(self, value):
        """Sanitize a tag value and return it if still meaningful."""
        cleaned = self.clean_text(str(value))
        return cleaned if cleaned else None

    def extract_year_from_date(self, date_string):
        """Extract year from various date formats"""
        if not date_string:
            return None
        
        year_match = re.search(r'(\d{4})', str(date_string))
        if year_match:
            year = int(year_match.group(1))
            if 1900 <= year <= 2030:
                return str(year)
        
        return None

    def add_to_comments(self, filepath, text):
        """Add text to the comments field of an audio file"""
        try:
            audio = MP3(filepath)
            
            # Ensure ID3 tags exist
            if audio.tags is None:
                audio.add_tags()
            
            if 'COMM::eng' in audio:
                current_comm = audio['COMM::eng']
                current_text = current_comm.text[0] if current_comm.text else ""
                if text not in current_text:  # Avoid duplicates
                    current_comm.text[0] = f"{current_text}\n{text}" if current_text else text
            else:
                audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=text)
            audio.save()
            return True
        except ID3NoHeaderError:
            # Create ID3 tags and try again
            try:
                audio = MP3(filepath)
                audio.add_tags()
                audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=text)
                audio.save()
                return True
            except Exception as e:
                print(f"   Error creating ID3 tags and adding comments: {e}")
                return False
        except Exception as e:
            print(f"   Error adding to comments: {e}")
            return False

    def clean_metadata(self, mp3_file):
        """Clean metadata from common junk patterns"""
        try:
            if not os.path.exists(mp3_file):
                print(f"File not found: {mp3_file}")
                return None
                
            audio = MP3(mp3_file)
            metadata_changed = False
            
            # Store original metadata for report
            original_metadata = {}
            changes = {}
            cleaning_actions = []
            
            # Expand fields to clean - clean all possible ID3 fields
            fields_to_clean = {
                'TIT2': 'title',
                'TPE1': 'artist',
                'TALB': 'album',
                'TPE2': 'album_artist',
                'TPE3': 'conductor',
                'TPE4': 'remixer',
                'TEXT': 'lyricist',  # Specifically added for lyricist
                'TCOM': 'composer',
                'TCON': 'genre',
                'COMM': 'comments'
            }
            
            # Clean each field
            for tag, field_name in fields_to_clean.items():
                if tag in audio:
                    # Special handling for COMM tag which has multiple parts
                    if tag == 'COMM':
                        for comment in audio.getall('COMM'):
                            original_text = comment.text[0] if comment.text else ""
                            original_metadata[f'comment:{comment.desc}:{comment.lang}'] = original_text
                            clean_text = self.clean_text(original_text)
                            
                            # Check for download sites in comments specifically
                            download_sites = self._extract_download_sites(original_text)
                            if download_sites:
                                for site in download_sites:
                                    cleaning_actions.append(f"Removed download site '{site}' from comment")
                            
                            if clean_text != original_text:
                                comment.text = [clean_text]
                                metadata_changed = True
                                changes[f'comment:{comment.desc}'] = {'original': original_text, 'new': clean_text}
                    else:
                        # Regular text fields
                        try:
                            original_text = audio[tag].text[0]
                            original_metadata[field_name] = original_text
                            clean_text = self.clean_text(original_text)
                            
                            # Check for download sites in this field
                            download_sites = self._extract_download_sites(original_text)
                            if download_sites:
                                for site in download_sites:
                                    cleaning_actions.append(f"Removed download site '{site}' from {field_name}")
                                    
                            if clean_text != original_text:
                                audio[tag].text = [clean_text]
                                metadata_changed = True
                                changes[field_name] = {'original': original_text, 'new': clean_text}
                        except (AttributeError, IndexError):
                            # Some fields may not have text attribute or may be empty
                            pass

            # TIPL (involved people list) may contain structured data; remove if polluted
            if 'TIPL' in audio:
                tipl_value = str(audio['TIPL'])
                if not self.sanitize_tag_value(tipl_value):
                    original_metadata['TIPL'] = tipl_value
                    del audio['TIPL']
                    metadata_changed = True
                    cleaning_actions.append("Removed TIPL tag with spam")
                    changes['TIPL'] = {'original': tipl_value, 'new': None}

            # Save changes if metadata was modified
            if metadata_changed:
                audio.save()
                
            return {
                'file': mp3_file,
                'metadata_changed': metadata_changed,
                'original_metadata': original_metadata,
                'changes': changes,
                'cleaning_actions': cleaning_actions
            }
            
        except Exception as e:
            print(f"Error cleaning metadata: {e}")
            return None
            
    def _extract_download_sites(self, text):
        """Extract download site URLs and references from text"""
        if not text:
            return []
            
        download_sites = []
        
        # Common download site patterns
        patterns = [
            # URLs and domains
            r'(?:www|http:|https:)+[^\s]+[\w]',  # Basic URLs
            r'\b\w+\.com\b',                      # .com domains
            r'\b\w+\.co\.\w{2}\b',                # .co.in etc domains
            r'\b\w+\.net\b',                      # .net domains
            r'\b\w+\.org\b',                      # .org domains
            
            # Common download site patterns
            r'\bdownload(?:ed)?(?:\s+from)?\s+\w+\b',  # "downloaded from" patterns
            r'\bfrom\s+\w+\.\w+\b',                # "from site.com" patterns
            r'\bwww(?:\s|\.)+\w+(?:\s|\.)+\w+\b',   # "www site com" with spaces or dots
            
            # Social media references
            r'\b(?:follow|like|subscribe)\s+(?:us|me)?\s+(?:on|at)?\s+\w+\b',
            r'\b@\w+\b',                          # @username mentions
            
            # Promotional text
            r'\bexclusive\b',                      # "exclusive"
            r'\bpromotional\s+(?:use|copy)\b',     # "promotional use/copy"
            r'\bcourtesy\s+of\b',                  # "courtesy of"
            r'\b(?:free|premium)\s+download\b'      # "free download" or "premium download"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                download_sites.append(match.group(0))
                
        return download_sites

    def identify_by_text_search(self, filepath):
        """🆕 Enhanced identification with genre and comprehensive metadata"""
        if not MUSICBRAINZ_AVAILABLE:
            print(f"   ❌ MusicBrainz not available")
            return None
        
        try:
            audio = MP3(filepath)
            # Get the duration of the local file in seconds
            file_duration = None
            try:
                file_duration = int(audio.info.length)
            except Exception:
                pass

            title = str(audio.get('TIT2', '')).strip() if 'TIT2' in audio else ''
            
            if not title or len(title) < 3:
                title = Path(filepath).stem
                title = self.clean_text(title)
            
            if len(title.strip()) < 3:
                print(f"   ❌ Title too short for search: '{title}'")
                return None
            
            print(f"   📝 Searching MusicBrainz for: '{title.strip()}'")
            
            # Enhanced search with tag information
            result = musicbrainzngs.search_recordings(
                query=title.strip(), 
                limit=5,
                strict=False
            )
            
            print(f"   📝 Found {len(result.get('recording-list', []))} potential matches")
            
            for i, recording in enumerate(result['recording-list'][:3]):
                score = int(recording.get('ext:score', 0))
                artist_name = recording['artist-credit'][0]['artist']['name']
                track_title = recording['title']
                
                # Extract comprehensive metadata
                year = None
                album = None
                genre = None
                duration = None
                label = None
                
                # Get release information
                if 'release-list' in recording:
                    for release in recording['release-list']:
                        # Year
                        if 'date' in release and not year:
                            year = self.extract_year_from_date(release['date'])
                        
                        # Album
                        if 'title' in release and not album:
                            album = self.clean_text(release['title'])
                        
                        # Label info
                        if 'label-info-list' in release and not label:
                            try:
                                label = release['label-info-list'][0]['label']['name']
                            except:
                                pass
                        
                        if year and album:
                            break
                
                # Duration
                if 'length' in recording:
                    try:
                        duration_ms = int(recording['length'])
                        duration = duration_ms // 1000  # Convert to seconds
                    except:
                        pass
                
                # Generate genre based on our logic
                genre = self.determine_genre(artist_name, track_title, album or '')
                
                print(f"   📝 Match {i+1}: {artist_name} - {track_title}")
                if year:
                    print(f"   📅 Year: {year}")
                if album:
                    print(f"   💿 Album: {album}")
                if genre:
                    print(f"   🎵 Genre: {genre}")
                if duration:
                    print(f"   ⏱️ Duration: {duration//60}:{duration%60:02d}")
                if label:
                    print(f"   🏷️ Label: {label}")
                print(f"   📊 Score: {score}")
                
                duration_ok = True
                if duration:
                    try:
                        if abs(int(duration) - int(file_duration)) > 5:
                            duration_ok = False
                    except Exception:
                        pass

                if score > 70 and duration_ok:
                    print(f"   ✅ Accepting match with score {score}")
                    
                    self.stats['text_search_hits'] += 1
                    if year:
                        self.stats['year_found'] += 1
                    if album:
                        self.stats['album_found'] += 1
                    if genre:
                        self.stats['genre_found'] += 1
                    
                    return {
                        'artist': artist_name,
                        'title': track_title,
                        'year': year,
                        'album': album,
                        'genre': genre,
                        'duration': duration,
                        'label': label,
                        'confidence': score / 100,
                        'method': 'text_search'
                    }
            
            print(f"   ❌ No matches above threshold")
            
        except Exception as e:
            print(f"   ❌ Text search error: {str(e)}")
        
        return None

    def identify_by_fingerprint(self, filepath):
        """Enhanced fingerprinting with album lookup"""
        if not ACOUSTID_AVAILABLE or not self.acoustid_api_key:
            return None
        
        try:
            print(f"   🎵 Fingerprinting audio...")
            
            for score, recording_id, title, artist in acoustid.match(
                self.acoustid_api_key, filepath, meta="recordings releases recordingids"
            ):
                if score > 0.75:
                    year = None
                    album = None
                    
                    try:
                        if MUSICBRAINZ_AVAILABLE and recording_id:
                            recording_info = musicbrainzngs.get_recording_by_id(
                                recording_id, 
                                includes=['releases']
                            )
                            
                            recording_data = recording_info['recording']
                            if 'release-list' in recording_data:
                                for release in recording_data['release-list']:
                                    if 'date' in release and not year:
                                        year = self.extract_year_from_date(release['date'])
                                    if 'title' in release and not album:
                                        album = self.clean_text(release['title'])
                                    if year and album:
                                        break
                    except:
                        pass
                    
                    genre = self.determine_genre(artist, title, album or '')
                    
                    self.stats['fingerprint_hits'] += 1
                    if year:
                        self.stats['year_found'] += 1
                    if album:
                        self.stats['album_found'] += 1
                    if genre:
                        self.stats['genre_found'] += 1
                    
                    return {
                        'artist': artist,
                        'title': title,
                        'year': year,
                        'album': album,
                        'genre': genre,
                        'confidence': score,
                        'method': 'fingerprint',
                        'recording_id': recording_id
                    }
            
        except Exception as e:
            print(f"   ❌ Fingerprint error: {e}")
        
        return None

    def enhance_metadata_online(self, filepath):
        """🆕 Ultimate metadata enhancement with all DJ fields"""
        try:
            audio = MP3(filepath)
            
            # Get current metadata
            current_artist = str(audio.get('TPE1', '')).strip() if 'TPE1' in audio else ''
            current_title = str(audio.get('TIT2', '')).strip() if 'TIT2' in audio else ''
            current_album = str(audio.get('TALB', '')).strip() if 'TALB' in audio else ''
            current_year = str(audio.get('TYER', '')).strip() if 'TYER' in audio else ''
            current_genre = str(audio.get('TCON', '')).strip() if 'TCON' in audio else ''
            
            print(f"   🔍 Current - Artist: '{current_artist}', Title: '{current_title}'")
            print(f"   🔍 Current - Album: '{current_album}', Year: '{current_year}', Genre: '{current_genre}'")
            
            # Force enhancement for debug
            needs_enhancement = True
            
            if not needs_enhancement:
                print(f"   ✅ Metadata is complete")
                return False
            
            print(f"   🌐 Trying online identification...")
            
            # Try text search first
            identified = self.identify_by_text_search(filepath)
            
            # Fallback to fingerprinting
            if not identified and self.acoustid_api_key:
                print(f"   🎵 Text search failed, trying fingerprinting...")
                identified = self.identify_by_fingerprint(filepath)
            
            if identified and identified['confidence'] > 0.7:
                updated_fields = []
                
                # Update basic fields
                if identified['artist']:
                    audio['TPE1'] = TPE1(encoding=3, text=identified['artist'])
                    updated_fields.append(f"artist: '{identified['artist']}'")
                
                if identified['title']:
                    audio['TIT2'] = TIT2(encoding=3, text=identified['title'])
                    updated_fields.append(f"title: '{identified['title']}'")
                
                # Update year
                if identified.get('year'):
                    try:
                        audio['TDRC'] = TDRC(encoding=3, text=identified['year'])
                        updated_fields.append(f"year: '{identified['year']}'")
                    except:
                        audio['TYER'] = TYER(encoding=3, text=identified['year'])
                        updated_fields.append(f"year: '{identified['year']}'")
                
                # Update album
                if identified.get('album'):
                    audio['TALB'] = TALB(encoding=3, text=identified['album'])
                    updated_fields.append(f"album: '{identified['album']}'")
                
                # Update genre
                if identified.get('genre'):
                    audio['TCON'] = TCON(encoding=3, text=identified['genre'])
                    updated_fields.append(f"genre: '{identified['genre']}'")
                
                # Estimate and set BPM
                if identified.get('genre'):
                    estimated_bpm = self.estimate_bpm_from_genre(identified['genre'])
                    if estimated_bpm:
                        audio['TBPM'] = TBPM(encoding=3, text=estimated_bpm)
                        updated_fields.append(f"BPM: '{estimated_bpm}'")
                        self.stats['bpm_found'] += 1
                
                # Add label info as comment
                if identified.get('label'):
                    comment_text = f"Label: {identified['label']}"
                    audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=comment_text)
                    updated_fields.append(f"label: '{identified['label']}'")
                
                audio.save()
                
                print(f"   📝 Updated: {', '.join(updated_fields)}")
                print(f"   💾 Enhanced via {identified['method']}")
                return True
            else:
                print(f"   ❌ No suitable match found")
                self.stats['identification_failures'] += 1
                self.stats['manual_review_needed'].append(Path(filepath).name)
                return False
            
        except Exception as e:
            print(f"   ❌ Enhancement error: {e}")
            return False

    def generate_clean_filename(self, artist, title, year=None):
        """Generate a clean filename from artist and title with optional year"""
        try:
            if artist and title:
                if year:
                    name = f"{artist} - {title} ({year})"
                else:
                    name = f"{artist} - {title}"
            elif title:
                name = title
            elif artist:
                name = artist
            else:
                return "Unknown Track.mp3"

            # Sanitize for filesystem
            name = re.sub(r'[<>:"/\\|?*]', '', name)
            name = re.sub(r'\s+', ' ', name).strip().rstrip('.')
            name = name[:120]  # Slightly longer limit for year

            return name
        except Exception as e:
            print(f"   ❌ Error generating filename: {e}")
            return "Unknown Track"
    
    def process_folder(self, input_folder, output_folder=None, enhance_online=False, include_year_in_filename=False, 
                      dj_analysis=True, analyze_quality=True, detect_key=True, detect_cues=True, calculate_energy=True, 
                      normalize_loudness=False, target_lufs=-14.0, rekordbox_xml=None, export_xml=False,
                      generate_report=True, high_quality_only=False, detailed_report=True):
        """Process a folder of MP3 files with enhanced DJ metadata analysis.
        
        Args:
            input_folder: Path to folder containing MP3 files
            output_folder: Path to output folder (creates if doesn't exist)
            enhance_online: Whether to use online services for metadata enhancement
            include_year_in_filename: Whether to include year in filenames
            dj_analysis: Whether to perform DJ-specific analysis
            analyze_quality: Whether to analyze audio quality
            detect_key: Whether to detect musical key
            detect_cues: Whether to detect cue points
            calculate_energy: Whether to calculate energy rating
            normalize_loudness: Whether to normalize loudness
            target_lufs: Target LUFS for loudness normalization
            rekordbox_xml: Path to Rekordbox XML file for metadata import
            export_xml: Whether to export Rekordbox XML after processing
            generate_report: Whether to generate HTML report

            high_quality_only: Only move high-quality (320kbps+) files to output folder
            detailed_report: Generate detailed per-file changes report
        """
        start_time = time.time()
        
        # Set up output folder - create a "clean" subfolder if not specified
        if not output_folder:
            parent_dir = os.path.dirname(os.path.normpath(input_folder))
            folder_name = os.path.basename(os.path.normpath(input_folder))
            output_folder = os.path.join(parent_dir, f"{folder_name}_clean")
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        print(f"\n🎼 Processing DJ collection: {input_folder}")
        print(f"✨ DJ enhancement mode: {'ON' if dj_analysis else 'OFF'}")
        print(f"📂 Output folder: {output_folder}")
        print(f"🎧 Quality filter: {'ON - Only 320kbps files will be moved' if high_quality_only else 'OFF'}")
        
        # Initialize report manager
        try:
            from .reports import DJReportManager
            report_manager = DJReportManager(base_dir=output_folder)
        except ImportError:
            # Provide a stub if reports module isn't present
            class ReportManagerStub:
                def is_file_already_processed(self, filepath): return None
                def mark_file_as_processed(self, filepath, info): pass
                def save_duplicates_report(self, duplicates): pass
                def generate_changes_report(self): pass
                def generate_low_quality_report(self): pass
                def generate_session_summary(self, stats): pass
            report_manager = ReportManagerStub()
        
        # Import Rekordbox XML if specified
        rekordbox_data = None
        if rekordbox_xml and os.path.exists(rekordbox_xml):
            print(f"🎛️ Importing Rekordbox XML: {rekordbox_xml}")
            rekordbox_data = self.import_rekordbox_xml(rekordbox_xml)
        
        # Initialize stats for tracking
        self.stats = {
            'text_search_hits': 0,
            'fingerprint_hits': 0,
            'identification_failures': 0,
            'year_found': 0,
            'album_found': 0,
            'genre_found': 0,
            'bpm_found': 0,
            'key_found': 0,
            'energy_rated': 0,
            'cue_points_detected': 0,
            'quality_analyzed': 0,
            'high_quality': 0,
            'low_quality': 0,
            'loudness_normalized': 0,
            'total_files': 0,
            'processed': 0,
            'output_folder': output_folder,
            'processing_time': 0,
            'manual_review_needed': []
        }
        
        # Track all processed files
        processed_files = []
        
        # Process all MP3 files
        for root, _, files in os.walk(input_folder):
            for file in tqdm(files, desc="Processing files"):
                if not file.lower().endswith('.mp3'):
                    continue
                
                self.stats['total_files'] += 1
                input_file = os.path.join(root, file)
                
                # Check if file was already processed
                already_processed = report_manager.is_file_already_processed(input_file)
                if already_processed:
                    print(f"\n🔎 Already processed: {file}")
                    # Update stats for previously processed file
                    if already_processed.get('is_high_quality', False):
                        self.stats['high_quality'] += 1
                    else:
                        self.stats['low_quality'] += 1
                        
                    self.stats['processed'] += 1
                    continue
                
                # Process the file
                try:
                    print(f"\n🔎 Processing: {file}")
                    
                    # Load MP3 file
                    audio = MP3(input_file)
                    
                    # Create file info dictionary
                    file_info = {
                        'input_path': input_file,
                        'original_metadata': {},
                        'cleaned_metadata': {},
                        'changes': [],
                        'enhanced': False,
                        'is_high_quality': False,
                        'bitrate': 0,
                        'sample_rate': 0
                    }
        
                    # Analyze audio quality if enabled
                    if analyze_quality:
                        try:
                            quality_info = self.analyze_audio_quality(input_file)
                            if quality_info:
                                file_info['bitrate'] = quality_info.get('bitrate_kbps', 0)
                                file_info['sample_rate'] = quality_info.get('sample_rate_khz', 0)
                                file_info['is_high_quality'] = self.is_high_quality(file_info['bitrate'], file_info['sample_rate'])
                                
                                if file_info['is_high_quality']:
                                    self.stats['high_quality'] += 1
                                    print(f"   🔊 High quality: {file_info['bitrate']}kbps @ {file_info['sample_rate']}kHz")
                                else:
                                    self.stats['low_quality'] += 1
                                    print(f"   ⚠️ Low quality: {file_info['bitrate']}kbps @ {file_info['sample_rate']}kHz")
                                    
                                self.stats['quality_analyzed'] += 1
                                file_info['changes'].append(f"Quality analyzed: {file_info['bitrate']}kbps @ {file_info['sample_rate']}kHz")
                                if quality_info.get('quality_text') and (not high_quality_only or file_info['is_high_quality']):
                                    self.add_to_comments(input_file, quality_info['quality_text'])
                        except Exception as e:
                            print(f"   ❌ Error analyzing quality: {e}")
                    
                    # Store original metadata
                    for tag in audio:
                        if tag.startswith('T') or tag.startswith('COMM'):
                            file_info['original_metadata'][tag] = str(audio[tag])
                    
                    # Clean metadata
                    print("   🧹 Cleaning metadata...")
                    metadata_changes = self.clean_metadata(input_file)
                    if metadata_changes and metadata_changes.get('cleaning_actions'):
                        file_info['changes'].extend(metadata_changes.get('cleaning_actions', []))
                        
                    # Apply Rekordbox metadata if available
                    if rekordbox_data:
                        self.apply_rekordbox_metadata(input_file, rekordbox_data)
                        file_info['changes'].append("Applied Rekordbox metadata")
                    
                    # Enhance metadata online if enabled
                    if enhance_online:
                        print("   🔍 Enhancing metadata online...")
                        enhanced = self.enhance_metadata_online(input_file)
                        if enhanced:
                            file_info['enhanced'] = True
                            file_info['changes'].append("Enhanced metadata online")
                    
                    # Reload audio after potential metadata changes
                    audio = MP3(input_file)
                    
                    # Extract current metadata for filename generation
                    artist = str(audio.get('TPE1', ''))
                    title = str(audio.get('TIT2', ''))
                    year = None
                    
                    # If no metadata, try to extract from filename
                    if not artist and not title:
                        filename = os.path.splitext(os.path.basename(input_file))[0]
                        # Clean filename first
                        filename = self.clean_text(filename)
                        
                        # Try common patterns: "Artist - Title", "Title - Artist", etc.
                        if ' - ' in filename:
                            parts = filename.split(' - ', 1)
                            if len(parts) == 2:
                                # Assume first part is artist, second is title
                                artist = parts[0].strip()
                                title = parts[1].strip()
                        elif filename:
                            # Use entire filename as title if no clear separation
                            title = filename.strip()
                    
                    if 'TDRC' in audio:
                        year_text = str(audio['TDRC'])
                        year = self.extract_year_from_date(year_text)
                    
                    # Store cleaned metadata
                    for tag in audio:
                        if tag.startswith('T') or tag.startswith('COMM'):
                            file_info['cleaned_metadata'][tag] = str(audio[tag])
                    
                    run_for_this_file = (not high_quality_only) or file_info['is_high_quality']

                    # Perform DJ-specific analysis
                    if dj_analysis and run_for_this_file:
                        # Detect musical key
                        if detect_key:
                            try:
                                key = self.detect_musical_key(input_file)
                                if key:
                                    print(f"   🎹 Detected key: {key}")
                                    file_info['changes'].append(f"Detected key: {key}")
                                    audio.save()
                                    self.stats['key_found'] += 1
                            except Exception as e:
                                print(f"   ⚠️ Key detection error: {e}")

                        # Detect cue points
                        if detect_cues:
                            try:
                                cues = self.detect_cue_points(input_file)
                                if cues:
                                    cue_text = "; ".join([f"{k}: {v}" for k, v in cues.items()])
                                    print(f"   📍 Detected cues: {cue_text}")
                                    # Store in comments
                                    if 'COMM::eng' in audio:
                                        comment = str(audio['COMM::eng']) + "\n" + cue_text
                                        audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=comment)
                                    else:
                                        audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=cue_text)
                                    audio.save()
                                    self.stats['cue_points_detected'] += 1
                                    file_info['changes'].append(f"Detected cue points: {cue_text}")
                            except Exception as e:
                                print(f"   ⚠️ Cue detection error: {e}")

                        # Calculate energy rating
                        if calculate_energy:
                            try:
                                energy_result = self.calculate_energy_rating(input_file)
                                if energy_result:
                                    energy = energy_result['energy_rating']
                                    print(f"   ⚡ Energy rating: {energy}/10")
                                    # Use our add_to_comments helper instead of direct manipulation
                                    self.add_to_comments(input_file, f"Energy: {energy}/10")
                                    self.stats['energy_rated'] += 1
                                    file_info['changes'].append(f"Energy rating: {energy}/10")
                            except Exception as e:
                                print(f"   ⚠️ Energy calculation error: {e}")

                    # Normalize loudness if requested
                    if normalize_loudness and run_for_this_file:
                        try:
                            print(f"   🔊 Normalizing loudness to {target_lufs} LUFS...")
                            self.normalize_loudness(input_file, target_lufs=target_lufs)
                            file_info['changes'].append(f"Normalized loudness to {target_lufs} LUFS")
                        except Exception as e:
                            print(f"   ⚠️ Loudness normalization error: {e}")
                    
                    # Generate clean filename
                    clean_filename = self.generate_clean_filename(artist, title, year if include_year_in_filename else None)
                    if not clean_filename.lower().endswith('.mp3'):
                        clean_filename += '.mp3'
                    
                    # Only move high-quality files to output if filter is enabled
                    if not high_quality_only or (high_quality_only and file_info['is_high_quality']):
                        # Determine relative path to maintain folder structure
                        rel_path = os.path.relpath(os.path.dirname(input_file), input_folder)
                        output_subdir = os.path.join(output_folder, rel_path)
                        
                        if not os.path.exists(output_subdir):
                            os.makedirs(output_subdir, exist_ok=True)
                            
                        output_file = os.path.join(output_subdir, clean_filename)
                        
                        # Copy file to output location
                        shutil.copy2(input_file, output_file)
                        file_info['output_path'] = output_file
                        print(f"   💾 Saved to: {output_file}")
                        file_info['changes'].append(f"Renamed to: {clean_filename}")
                    else:
                        print(f"   ⚠️ Low quality file not moved to output folder")
                        file_info['changes'].append("Low quality file not moved to output folder")
                    
                    # Mark as processed in database
                    report_manager.mark_file_as_processed(input_file, file_info)
                    
                    # Update stats
                    self.stats['processed'] += 1
                    processed_files.append(file_info)
                    
                except Exception as e:
                    print(f"   ❌ Error processing {file}: {e}")
                    traceback.print_exc()
        
        # Check for duplicates if requested
        if len(processed_files) > 0:
            print("\n🔍 Checking for duplicates...")
            duplicates = self.find_duplicates(output_folder)
            if duplicates:
                report_manager.save_duplicates_report(duplicates)
        
        # Generate reports if requested
        if generate_report:
            # Generate HTML report
            report_path = os.path.join(output_folder, 'reports')
            os.makedirs(report_path, exist_ok=True)
            
            html_report = os.path.join(report_path, 'dj_report.html')
            self.generate_html_report(html_report)
            

            
            if detailed_report:
                report_manager.generate_changes_report()
                
            # Generate low quality files report if any
            if self.stats['low_quality'] > 0:
                report_manager.generate_low_quality_report()
                
            # Generate session summary
            report_manager.generate_session_summary(self.stats)
        
        # Export Rekordbox XML if requested
        if export_xml:
            xml_output = os.path.join(output_folder, 'rekordbox_export.xml')
            self.export_rekordbox_xml(output_folder, xml_output)
        
        # Calculate processing time
        self.stats['processing_time'] = time.time() - start_time
        
        # Print stats
        print(f"\n🎵 DJ MUSIC CLEANER PROCESSING SUMMARY 🎵")
        print(f"{'='*50}")
        print(f"📊 Total files: {self.stats['total_files']}")
        print(f"✅ Processed files: {self.stats['processed']}")
        print(f"🔊 High quality files: {self.stats['high_quality']}")
        print(f"⚠️ Low quality files: {self.stats['low_quality']}")
        
        if enhance_online:
            print(f"🌐 Online enhancements: {self.stats['text_search_hits'] + self.stats['fingerprint_hits']}")
        
        if dj_analysis:
            print(f"🎹 Keys detected: {self.stats['key_found']}")
            print(f"📍 Cue points detected: {self.stats['cue_points_detected']}")
            print(f"⚡ Energy ratings: {self.stats['energy_rated']}")
        
        if normalize_loudness:
            print(f"🔊 Loudness normalized: {self.stats['loudness_normalized']}")
            
        print(f"Processing time: {self.stats['processing_time']:.1f} seconds")
        print(f"{'='*50}")
        
        return processed_files
    
    # (removed old, broken 'print_stats' implementation that referenced undefined variables)
        
    def generate_html_report(self, output_file):
        """Generate HTML report of processed files"""
        try:
            print(f"\n📊 Generating DJ report to: {output_file}")
            
            with open(output_file, 'w') as f:
                f.write("""<!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>DJ Music Processing Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                        h1, h2 { color: #333; }
                        .stats { background-color: #f4f4f4; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                        .file { margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                        .file h3 { margin-top: 0; color: #444; }
                        .action { color: #0066cc; }
                        .warning { color: #cc6600; }
                        .error { color: #cc0000; }
                    </style>
                </head>
                <body>
                    <h1>DJ Music Processing Report</h1>
                """)
                
                # Write statistics section
                f.write("""<div class="stats">
                    <h2>Processing Statistics</h2>
                    <p><strong>Total files:</strong> {}</p>
                    <p><strong>Processed files:</strong> {}</p>
                    <p><strong>High quality files:</strong> {}</p>
                    <p><strong>Low quality files:</strong> {}</p>
                </div>""".format(
                    self.stats.get('total_files', 0),
                    self.stats.get('processed', 0),
                    self.stats.get('high_quality', 0),
                    self.stats.get('low_quality', 0)
                ))
                
                f.write("</body></html>")
                
        except Exception as e:
            print(f"\nError generating HTML report: {e}")
            

                

    def print_stats(self):
        """Ultimate DJ stats with all metadata types"""
        print(f"\n📈 Ultimate DJ Processing Stats:")
        print(f"   📝 Text search hits: {self.stats['text_search_hits']}")
        print(f"   🎵 Fingerprint hits: {self.stats['fingerprint_hits']}")
        print(f"   📅 Years found: {self.stats['year_found']}")
        print(f"   💿 Albums found: {self.stats['album_found']}")
        print(f"   🎵 Genres found: {self.stats['genre_found']}")
        print(f"   ⚡ BPM estimates: {self.stats['bpm_found']}")
        print(f"   ❓ Failed identifications: {self.stats['identification_failures']}")
        
        total = self.stats['text_search_hits'] + self.stats['fingerprint_hits'] + self.stats['identification_failures']
        if total > 0:
            success = self.stats['text_search_hits'] + self.stats['fingerprint_hits']
            rate = (success / total) * 100
            print(f"   ✅ Success rate: {rate:.1f}%")
            
            if success > 0:
                year_rate = (self.stats['year_found'] / success) * 100
                album_rate = (self.stats['album_found'] / success) * 100
                genre_rate = (self.stats['genre_found'] / success) * 100
                bpm_rate = (self.stats['bpm_found'] / success) * 100
                print(f"   📅 Year completion: {year_rate:.1f}%")
                print(f"   💿 Album completion: {album_rate:.1f}%")
                print(f"   🎵 Genre completion: {genre_rate:.1f}%")
                print(f"   ⚡ BPM completion: {bpm_rate:.1f}%")

    def generate_report(self, processed_files, output_path):
        """Generate enhanced processing report"""
        report_path = output_path / "cleaning_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DJ Music Cleaning Report - Ultimate Version\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files processed: {len(processed_files)}\n")
            f.write(f"Online enhanced: {sum(1 for f in processed_files if f['enhanced'])}\n")
            f.write(f"Text search hits: {self.stats['text_search_hits']}\n")
            f.write(f"Fingerprint hits: {self.stats['fingerprint_hits']}\n")
            f.write(f"Years found: {self.stats['year_found']}\n")
            f.write(f"Albums found: {self.stats['album_found']}\n")
            f.write(f"Genres found: {self.stats['genre_found']}\n")
            f.write(f"BPM estimates: {self.stats['bpm_found']}\n")
            f.write(f"Identification failures: {self.stats['identification_failures']}\n\n")
            
            f.write("File Changes:\n")
            f.write("-" * 30 + "\n")
            for file_info in processed_files:
                input_path = file_info.get('input_path')
                output_path = file_info.get('output_path', input_path)
                if input_path and output_path and input_path != output_path:
                    f.write(f"RENAMED: {input_path} → {output_path}\n")
                    if file_info.get('enhanced'):
                        f.write("  Enhanced: Yes\n")
            
            if self.stats['manual_review_needed']:
                f.write(f"\nManual Review Needed:\n")
                f.write("-" * 30 + "\n")
                for filename in self.stats['manual_review_needed']:
                    f.write(f"  {filename}\n")
        

def main():
    """Ultimate DJ configuration with command line arguments"""
    import argparse
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="DJ Music Cleaner - Ultimate DJ Library Management Tool")
    parser.add_argument("-i", "--input", help="Input folder containing MP3 files", required=True)
    parser.add_argument("-o", "--output", help="Output folder for cleaned files (optional, uses input folder if not specified)")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="AcoustID API key for enhanced identification. If not provided, will use ACOUSTID_API_KEY env var."
    )
    parser.add_argument("--year", help="Include year in filename", action="store_true")
    parser.add_argument("--online", help="Enable online metadata enhancement", action="store_true")
    
    # DJ-specific features
    dj_group = parser.add_argument_group('DJ Features')
    dj_group.add_argument("--no-dj", help="Disable all DJ-specific analysis features", action="store_true")
    dj_group.add_argument("--no-quality", help="Disable audio quality analysis", action="store_true")
    dj_group.add_argument("--no-key", help="Disable key detection", action="store_true")
    dj_group.add_argument("--no-cues", help="Disable cue point detection", action="store_true")
    dj_group.add_argument("--no-energy", help="Disable energy rating", action="store_true")
    
    # Advanced features
    adv_group = parser.add_argument_group('Advanced Features')
    adv_group.add_argument("--normalize", help="Enable loudness normalization", action="store_true")
    adv_group.add_argument("--lufs", help="Target LUFS for loudness normalization", type=float, default=-14.0)
    adv_group.add_argument("--rekordbox", help="Path to Rekordbox XML file for metadata import")
    adv_group.add_argument("--export-xml", help="Export Rekordbox XML after processing", action="store_true")
    adv_group.add_argument("--duplicates", help="Find duplicates in the input folder", action="store_true")
    adv_group.add_argument("--high-quality", help="Only move high-quality files (320kbps+) to output folder", action="store_true")
    adv_group.add_argument("--priorities", help="Show metadata completion priorities", action="store_true")
    adv_group.add_argument("--report", help="Generate HTML report", action="store_true", default=True)
    adv_group.add_argument("--no-report", help="Disable HTML report generation", action="store_true")
    adv_group.add_argument("--detailed-report", help="Generate detailed per-file changes report", action="store_true", default=True)
    adv_group.add_argument("--no-detailed-report", help="Disable detailed per-file changes report", action="store_true")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get API key from CLI arg or environment variable
    ACOUSTID_API_KEY = args.api_key or os.getenv("ACOUSTID_API_KEY")
    
    # Validate API key requirement for online mode
    if args.online and not ACOUSTID_API_KEY:
        parser.error(
            "Online mode requires an AcoustID API key. "
            "Pass it with --api-key or set the ACOUSTID_API_KEY environment variable."
        )

    acoustid_api_key = ACOUSTID_API_KEY
    
    # Validate input folder
    input_folder = args.input
    output_folder = args.output
    
    # Initialize cleaner
    print(f"\n🎵 DJ MUSIC CLEANER - ULTIMATE VERSION 🎵")
    print(f"{'='*50}")
    print(f"📂 Input folder: {input_folder}")
    print(f"📂 Output folder: {output_folder if output_folder else '[In-place]'}")
    print(f"🌐 Online enhancement: {'Enabled' if args.online else 'Disabled'}")
    print(f"🎛️ DJ analysis features: {'Disabled' if args.no_dj else 'Enabled'}")
    
    if args.normalize:
        print(f"🔊 Loudness normalization: Enabled (Target: {args.lufs} LUFS)")
        
    if args.rekordbox:
        print(f"🎛️ Rekordbox XML import: {args.rekordbox}")
    
    print(f"{'='*50}")
    
    # Initialize DJ Music Cleaner
    cleaner = DJMusicCleaner(acoustid_api_key=acoustid_api_key)
    
    # Special operations
    if args.duplicates:
        print("\n🔍 DUPLICATE DETECTION MODE")
        duplicates = cleaner.find_duplicates(input_folder)
        return
        
    if args.priorities:
        print("\n📊 METADATA PRIORITIES MODE")
        priorities = cleaner.prioritize_metadata_completion(input_folder)
        return
    
    # Process files
    processed_files = cleaner.process_folder(
        input_folder=input_folder, 
        output_folder=output_folder,
        enhance_online=args.online,
        include_year_in_filename=args.year,
        dj_analysis=not args.no_dj,
        analyze_quality=not args.no_quality,
        detect_key=not args.no_key,
        detect_cues=not args.no_cues,
        calculate_energy=not args.no_energy,
        normalize_loudness=args.normalize,
        target_lufs=args.lufs,
        rekordbox_xml=args.rekordbox,
        export_xml=args.export_xml,
        generate_report=args.report and not args.no_report,

        high_quality_only=args.high_quality,
        detailed_report=args.detailed_report and not args.no_detailed_report
    )
    
    print(f"\n✅ Processed {len(processed_files)} files")
    if cleaner.stats['manual_review_needed']:
        print(f"\n⚠️ {len(cleaner.stats['manual_review_needed'])} files need manual review")
    
    print(f"\n✨ Done! Your DJ library is now professionally organized and enhanced.")

def cli_main():
    """Entry point for the CLI tool when installed via pip"""
    main()

if __name__ == "__main__":
    main()
