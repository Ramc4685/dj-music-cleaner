# DJ Music Cleaner

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)

A high-performance command-line tool that cleans, organizes, and enhances DJ music libraries with professional metadata and audio analysis. Built for DJs who need clean, consistent metadata and high-quality audio files.

## 🎧 Key Features

### Metadata Optimization
- **Tag Cleaning**: Removes promotional text, URLs, and other junk from all metadata fields
- **Standardized Formatting**: Ensures proper title casing and consistent naming conventions
- **Online Enhancement**: Integrates with MusicBrainz and AcoustID for accurate track identification
- **Audio File Caching**: Dramatically improves performance when batch processing files
- **Batched Tag Updates**: Efficiently updates MP3 tags to reduce disk operations

### DJ-Specific Analysis
- **BPM Detection**: Accurate tempo detection with ensemble analysis
- **Key Detection**: Musical key detection with Camelot wheel notation for harmonic mixing
- **Audio Quality Analysis**: Bit rate verification, sample rate, and spectral analysis
- **Loudness Assessment**: EBU R128 loudness metrics for consistent playback levels
- **Energy Rating**: Calculates energy level based on audio characteristics

### Collection Management
- **Duplicate Detection**: Finds duplicates with awareness of remixes and versions
- **Quality Filtering**: Option to only process high-quality files (320kbps+)
- **Comprehensive Reports**: JSON, HTML, and CSV reports of all processed files and changes
- **Rekordbox Integration**: Import, enhance, and export Rekordbox XML collections

## 📋 Requirements

### Core Dependencies
- Python 3.9 or higher
- `mutagen`: MP3 tag reading/writing
- `numpy`: Numerical processing
- `tqdm`: Progress indicators
- `python-dotenv`: Environment variable management
- `soundfile`: Audio file reading

### Online Features
- `musicbrainzngs`: MusicBrainz API client
- `pyacoustid`: AcoustID audio fingerprinting

### DJ Analysis Features
- `aubio`: Primary audio analysis (BPM detection)
- `librosa`: Enhanced audio feature extraction
- `pyloudnorm`: Loudness normalization

## 🚀 Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/ramc46/dj-music-cleaner.git
cd dj-music-cleaner

# Option 1: Install with all dependencies
pip install -e .[full]

# Option 2: Install minimal version
pip install -e .
```

### Install from PyPI

```bash
# Full installation with all features
pip install dj-music-cleaner[full]

# Minimal installation (core features only)
pip install dj-music-cleaner
```

## 🎛️ Usage

### Basic Usage

```bash
# Basic cleaning (offline mode)
dj-music-cleaner -i /path/to/music -o /path/to/output

# With online enhancement
dj-music-cleaner -i /path/to/music -o /path/to/output --online --api-key YOUR_ACOUSTID_KEY

# Or set the API key as an environment variable
export ACOUSTID_API_KEY="YOUR_ACOUSTID_KEY"
dj-music-cleaner -i /path/to/music -o /path/to/output --online
```

### Advanced Examples

```bash
# Perform a dry run (preview changes without modifying files)
dj-music-cleaner -i /path/to/music --dry-run

# Process with full DJ analysis, but without online lookup
dj-music-cleaner -i /path/to/music -o /path/to/output --detect-key --detect-bpm --calculate-energy

# Only move high-quality (320kbps) files to output
dj-music-cleaner -i /path/to/music -o /path/to/output --high-quality

# Generate detailed HTML and JSON reports
dj-music-cleaner -i /path/to/music -o /path/to/output --report --detailed-report --json-report reports/output.json
```
## 🔍 CLI Options

```
usage: dj-music-cleaner [-h] -i INPUT [-o OUTPUT] [--api-key API_KEY] [--year] [--online] 
                        [--no-dj] [--no-quality] [--no-key] [--no-cues] [--no-energy] 
                        [--normalize] [--lufs LUFS] [--find-duplicates] [--high-quality] 
                        [--detect-bpm] [--detect-key] [--calculate-energy] [--dry-run] 
                        [--workers WORKERS] [--report] [--detailed-report] [--json-report PATH] 
                        [--import-rekordbox PATH] [--export-rekordbox PATH]
```

### Basic Options
| Option | Description |
|--------|-------------|
| `-i, --input PATH` | **Required**. Input folder containing MP3 files |
| `-o, --output PATH` | Output folder for cleaned files (defaults to input folder) |
| `--api-key KEY` | AcoustID API key for online identification |
| `--year` | Include year in filename |
| `--online` | Enable online metadata enhancement |
| `--dry-run` | Preview changes without modifying files |
| `--workers N` | Number of parallel workers (0=auto, defaults to single worker for DJ analysis) |

### DJ Features
| Option | Description |
|--------|-------------|
| `--no-dj` | Disable all DJ-specific analysis features |
| `--detect-bpm` | Enable BPM detection |
| `--detect-key` | Enable musical key detection |
| `--calculate-energy` | Enable energy rating calculation |
| `--no-quality` | Disable audio quality analysis |
| `--normalize` | Enable loudness normalization |
| `--lufs LUFS` | Target LUFS for loudness normalization |
| `--high-quality` | Only move high-quality files (320kbps+) to output |

### Report Options
| Option | Description |
|--------|-------------|
| `--report` | Generate HTML report in output directory |
| `--no-report` | Disable HTML report generation |
| `--detailed-report` | Generate detailed per-file changes report |
| `--json-report PATH` | Path to generate JSON report |
| `--csv-report PATH` | Path to generate CSV report |

### Rekordbox Integration
| Option | Description |
|--------|-------------|
| `--import-rekordbox PATH` | Path to Rekordbox XML for metadata import |
| `--export-rekordbox PATH` | Path to export enhanced Rekordbox XML |
| `--rekordbox-preserve` | Preserve DJ data during processing |

## ⚡ Performance Optimizations

### Audio File Caching
The tool now implements efficient audio file caching to dramatically speed up processing when DJ analysis features are enabled. Files are only loaded once into memory even if they require multiple analysis passes.

### Batched Tag Updates
Metadata updates are batched and written to disk in a single operation, reducing I/O overhead and improving overall processing speed.

## 📈 Benchmark Results

Processing 100 MP3 files with full DJ analysis:

| Configuration | Time (seconds) | Memory Usage |
|---------------|----------------|-------------|
| Without caching | 324s | Normal |
| With caching | 187s | +15% |
| Speedup | 42% faster | - |

## 🔧 Troubleshooting

### Common Issues

- **Multiprocessing Crashes**: When DJ analysis features are enabled, the script automatically uses single worker mode for stability. This is because some audio analysis libraries (librosa/essentia) can cause segmentation faults in parallel mode.

- **Missing AcoustID API Key**: For online enhancement, you need an AcoustID API key. Get one for free at [https://acoustid.org/new-application](https://acoustid.org/new-application).

- **Library Conflicts**: If you encounter segmentation faults, try installing dependencies in a virtual environment:
  ```bash
  python -m venv env
  source env/bin/activate  # On macOS/Linux
  pip install -e .[full]
  ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

RamC Venkatasamy
