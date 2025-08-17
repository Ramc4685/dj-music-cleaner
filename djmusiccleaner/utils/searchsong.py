import requests
import mutagen
from mutagen.easyid3 import EasyID3
import os

def show_all_tags(file_path):
    """
    Prints all ID3 tags and their values from the MP3 file.
    """
    try:
        audio = EasyID3(file_path)
        if not audio:
            print("No ID3 tags found in the file.")
            return
        print(f"ID3 Tags in '{file_path}':\n")
        for tag, value in audio.items():
            print(f"{tag}: {value}")
    except mutagen.id3.ID3NoHeaderError:
        print("No ID3 header found (no tags present).")
    except Exception as e:
        print(f"Error reading tags: {e}")

def get_mp3_metadata(file_path):
    """
    Extracts title, artist, and album metadata from an MP3 file.
    Returns empty strings if tags are missing or an error occurs.
    """
    try:
        audio = EasyID3(file_path)
        title = audio.get("title", [""])[0]
        artist = audio.get("artist", [""])[0]
        album = audio.get("album", [""])[0]
        return title, artist, album
    except Exception:
        return "", "", ""

def filename_to_title_artist(file_path):
    # Gets 'Yammadi Aathadi' from '10 Yammadi Aathadi.mp3' 
    base = os.path.basename(file_path)  # 10 Yammadi Aathadi.mp3
    name = os.path.splitext(base)[0]    # 10 Yammadi Aathadi
    # Strip leading track numbers if present
    parts = name.split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():  
        potential_title = parts[1]
    else:
        potential_title = name
    return potential_title.strip(), ""  # Only use title for fuzzy cases

def search_musicbrainz(title, artist=None):
    query = f'title:"{title}"'
    if artist and artist.strip():
        query += f' AND artist:"{artist}"'
    params = {
        'query': query,
        'fmt': 'json',
        'limit': 10
    }
    response = requests.get("https://musicbrainz.org/ws/2/recording/", params=params)
    if response.ok:
        return response.json()
    return {}

def filter_exact_matches(results, target_title, target_artist):
    exact_matches = []
    target_title_lower = target_title.strip().lower()
    target_artist_lower = target_artist.strip().lower() if target_artist else ""
    for item in results.get("recordings", []):
        track_title = item.get("title", "").strip().lower()
        found_artist = ""
        if item.get("artist-credit"):
            found_artist = " ".join(
                credit.get("name", "") for credit in item["artist-credit"]
            ).strip().lower()
        if track_title == target_title_lower:
            if target_artist_lower:
                if target_artist_lower in found_artist:
                    exact_matches.append(item)
            else:
                exact_matches.append(item)
    return exact_matches

def extract_album_and_composer(recording):
    album = None
    composer = None
    if recording.get("releases"):
        album = recording["releases"][0].get("title")
    if recording.get("relations"):
        for rel in recording["relations"]:
            if rel.get("type") == "composer" and rel.get("artist"):
                composer = rel["artist"].get("name")
                break
    if not composer and recording.get("artist-credit"):
        composers = []
        for credit in recording["artist-credit"]:
            if credit.get("name"):
                composers.append(credit.get("name"))
        if composers:
            composer = ", ".join(composers)
    return album, composer

def find_album_composer(mp3_file_path):
    print("\n--- Inspecting file tags ---")
    show_all_tags(mp3_file_path)

    title, artist, album = get_mp3_metadata(mp3_file_path)
    if not title:
        title, artist = filename_to_title_artist(mp3_file_path)
        print(f"\n(Tags missing) Fallback to filename: Title='{title}', Artist='{artist}'")
    else:
        print(f"\nTags found: Title='{title}', Artist='{artist}', Album='{album}'")

    search_title = title
    search_artist = artist

    results = search_musicbrainz(search_title, search_artist)
    filtered = filter_exact_matches(results, search_title, search_artist)

    if not filtered:
        print("\nNo exact matches found in MusicBrainz for this file.")
        print("Try manual search on MusicBrainz or verify spelling/language.")
        return

    print("\n--- Match Results ---")
    for idx, rec in enumerate(filtered, 1):
        rec_title = rec.get("title")
        album_name, composer_name = extract_album_and_composer(rec)
        print(f"\nResult {idx}:")
        print(f"  Title   : {rec_title}")
        print(f"  Album   : {album_name}")
        print(f"  Composer: {composer_name}")

if __name__ == "__main__":
    file_path = "/Users/ramc/Documents/Songs/Tamil_Songs/Anirudh-Ravichander-Jolly-O-Gymkhana-(2022).mp3"
    find_album_composer(file_path)
