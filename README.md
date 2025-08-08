# DJ Music Cleaner - Ultimate DJ Library Management Tool

**Author: RamC Venkatasamy**

A comprehensive Python tool for DJ music library management that cleans, organizes, and enhances your DJ collection with professional metadata, audio analysis, and Rekordbox integration. Specifically designed for professional DJs who need high-quality music files with consistent metadata and formatting.

## Features

### Core Features
- **Metadata Cleaning**: Remove promotional text and download site references from file names and tags
- **Aggressive Junk Removal**: Clean all metadata fields including rarely checked ones like Lyricist, Comments, etc.
- **Standardized Formatting**: Proper title casing, consistent naming conventions
- **Online Enhancement**: Integration with MusicBrainz and AcoustID for accurate track identification
- **Smart Filename Parsing**: Extracts artist/title from filenames when metadata is missing
- **High-Quality Filter**: Option to move only 320kbps files to clean output folder
- **Detailed Reporting**: Per-file logs of all actions taken and changes made

### DJ-Specific Enhancements
- **Audio Quality Analysis**: Bit rate, bit depth, sample rate, spectral analysis
- **Dynamic Range Analysis**: EBU R128 loudness metrics for selecting high-quality masters
- **Musical Key Detection**: Librosa-based key detection with Camelot wheel notation for harmonic mixing
- **Cue Point Detection**: Automatic detection of intro, outro, and drop points
- **Energy Rating**: 1-10 energy rating based on audio characteristics
- **Loudness Normalization**: Consistent loudness across your collection with target LUFS

### Collection Management
- **Duplicate Detection**: Find duplicates based on artist and title with remix awareness
- **Persistent Reports**: Keep detailed logs of all processed files and changes made
- **Low Quality Reports**: Generate lists of files below quality threshold (320kbps) for later replacement
- **Metadata Prioritization**: Score and rank files by DJ metadata completeness
- **Rekordbox Integration**: Import, apply, and export Rekordbox XML collection files
- **Clean Output Folder**: Automatically organize high-quality files in a separate folder structure

## Installation

### Prerequisites

```bash
# Required Python packages
pip install mutagen musicbrainzngs

# Optional dependencies for full functionality
pip install librosa numpy soundfile pyloudnorm pyacoustid pyrekordbox tqdm
```

### Quick Setup

```bash
# Clone this repository
git clone https://github.com/yourusername/dj-music-cleaner.git
cd dj-music-cleaner

# Install dependencies
pip install -r requirements.txt

# Run directly
python -m djmusiccleaner.dj_music_cleaner --help
```

### Optional: Install as CLI tool

```bash
# Install as a CLI tool (if setup.py exists)
pip install -e .

# Then use the djcleaner command from anywhere
djcleaner --help
```

```bash
djcleaner -i /path/to/music --online
## Dependencies

### Core Dependencies (Required)
- `mutagen`: MP3 metadata manipulation
- `musicbrainzngs`: MusicBrainz API access

### Optional Dependencies (Recommended)
- `librosa` and `numpy`: Audio analysis, key detection, cue points, energy rating
- `pyloudnorm` and `soundfile`: Loudness normalization
- `pyacoustid`: Audio fingerprinting and identification
- `pyrekordbox`: Rekordbox XML integration
- `tqdm`: Progress bars (cosmetic)

### External Dependencies
- `ffmpeg`: Required for audio conversion during loudness normalization

## Usage

### Basic Usage
```bash
# Basic cleaning (offline mode)
python dj_music_cleaner.py --input /path/to/music --output /path/to/clean

# With online enhancement (recommended)
# Option 1: Pass API key directly
python -m djmusiccleaner.dj_music_cleaner --input /path/to/music --output /path/to/clean --online --api-key YOUR_ACOUSTID_KEY

# Option 2: Set environment variable (preferred)
export ACOUSTID_API_KEY="YOUR_ACOUSTID_KEY"
python -m djmusiccleaner.dj_music_cleaner --input /path/to/music --output /path/to/clean --online

# High-quality filter only (move only 320kbps+ files)
python dj_music_cleaner.py --input /path/to/music --output /path/to/clean --high-quality
```

### Command Line Options

#### Required Options
```
-i, --input INPUT    Input folder containing MP3 files
```

#### Basic Options
```
-o, --output OUTPUT  Output folder for cleaned files (optional, uses input folder if not specified)
--api-key API_KEY    AcoustID API key for enhanced identification (optional if ACOUSTID_API_KEY env var is set)
--year               Include year in filename
--online             Enable online metadata enhancement (MusicBrainz/AcoustID)
```

#### DJ Features
```
--no-dj              Disable all DJ-specific analysis features
--no-quality         Disable audio quality analysis
--no-key             Disable key detection
--no-cues            Disable cue point detection
--no-energy          Disable energy rating
```

#### Advanced Features
```
--normalize          Enable loudness normalization
--lufs LUFS          Target LUFS for loudness normalization (default: -14.0)
--rekordbox REKORDBOX Path to Rekordbox XML file for metadata import
--export-xml         Export Rekordbox XML after processing
--duplicates         Find duplicates in the input folder
--high-quality       Only move high-quality files (320kbps+) to output folder
--priorities         Show metadata completion priorities
--report             Generate HTML report (default: enabled)
--detailed-report    Generate detailed per-file changes report (default: enabled)
```

### Examples

**Basic cleaning with online enhancement:**
```bash
# Option 1: Pass API key directly
djcleaner -i /path/to/music -o /path/to/clean --online --api-key YOUR_ACOUSTID_KEY

# Option 2: Set environment variable (preferred)
export ACOUSTID_API_KEY="YOUR_ACOUSTID_KEY"
djcleaner -i /path/to/music -o /path/to/clean --online
```

**High-quality filter only:**
```bash
python dj_music_cleaner.py --input /path/to/music --output /path/to/clean --high-quality
```

**Full DJ enhancement with loudness normalization:**
```bash
# Option 1: Pass API key directly
djcleaner -i /path/to/music -o /path/to/clean --online --api-key YOUR_ACOUSTID_KEY --normalize --lufs -14.0

# Option 2: Set environment variable (preferred)
export ACOUSTID_API_KEY="YOUR_ACOUSTID_KEY"
djcleaner -i /path/to/music -o /path/to/clean --online --normalize --lufs -14.0
```

**Rekordbox integration:**
```bash
python dj_music_cleaner.py --input /path/to/music --rekordbox /path/to/rekordbox.xml --export-xml
```

**Find duplicates:**
```bash
python dj_music_cleaner.py --input /path/to/music --duplicates
```

### Security Note

> **⚠️ Important:** Never hardcode your API key in code or commit it to Git. Use the `--api-key` flag for temporary use, or store it securely as an environment variable.

For enhanced security, you can also use a `.env` file in your project directory:
```bash
# Create .env file
echo "ACOUSTID_API_KEY=YOUR_ACOUSTID_KEY" > .env

# The tool will automatically load it
djcleaner -i /path/to/music --online
```

**Show metadata completion priorities:**
```bash
python dj_music_cleaner.py --input /path/to/music --priorities
```

## High-Quality Mode

When using the `--high-quality` flag, the tool operates in **strict mode**:

- **Quality Analysis First**: Each file's audio quality (bitrate, sample rate) is analyzed before any processing
- **Skip Low-Quality Files**: Files below 320kbps and 44.1kHz are completely skipped:
  - ❌ No tag modifications or cleaning
  - ❌ No online metadata enhancement
  - ❌ No DJ analysis (key, cues, energy)
  - ❌ No copying to output folder
  - ✅ Only logged in reports under "Skipped (Low Quality)"
- **Process High-Quality Files**: Files ≥320kbps and ≥44.1kHz get full processing:
  - ✅ Complete metadata cleaning and enhancement
  - ✅ All DJ analysis features
  - ✅ Renamed and copied to output folder

This ensures your output folder contains only professional-quality files while preserving the original low-quality files unchanged.

```bash
# High-quality mode example
djcleaner -i /path/to/mixed_quality_music -o /path/to/hq_only --high-quality --online
```

## API Key Configuration

For online metadata enhancement, you need an AcoustID API key:

### Option 1: Command Line (temporary use)
```bash
djcleaner -i /path/to/music --online --api-key YOUR_ACOUSTID_KEY
```

### Option 2: Environment Variable (recommended)
```bash
export ACOUSTID_API_KEY="YOUR_ACOUSTID_KEY"
djcleaner -i /path/to/music --online
```

### Option 3: .env File (development)
```bash
echo "ACOUSTID_API_KEY=YOUR_ACOUSTID_KEY" > .env
djcleaner -i /path/to/music --online
```

**Error Handling**: If `--online` is used without an API key, the tool will show a clear error and exit.

## Metadata Formats

The tool adds DJ-specific metadata using the following fields:

- **ID3 Standard Tags**: Title, Artist, Album, Year, Genre, BPM
- **Comments**: Formatted as `[KEY]:[VALUE]` pairs:
  - `KEY: C major (8B)` - Musical key with Camelot notation
  - `ENERGY: 8/10` - Energy rating
  - `CUE_INTRO_END: 00:15.20` - Detected cue points
  - `CUE_DROP: 00:30.75` - Detected drop points
  - `CUE_OUTRO_START: 03:15.40` - Detected outro points
  - `QUALITY: 320kbps, 44.1kHz` - Audio quality metrics
  - `LOUDNESS: Normalized to -14 LUFS` - Loudness normalization info

## Rekordbox Integration

The tool can:
1. Import DJ-specific metadata from Rekordbox XML exports
2. Apply this metadata to your audio files
3. Export an enhanced Rekordbox XML collection for reimport

This allows for two-way synchronization between your audio files and Rekordbox collection.

## Development

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
