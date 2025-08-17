"""
Text processing utilities for DJ Music Cleaner

This module consolidates all text processing functionality including
filename generation, metadata cleaning, and text normalization.
"""

import re
import unicodedata
from typing import Optional, List, Set


# Domain and pollution patterns for metadata cleaning
DOMAIN_PATTERNS = [
    r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    r'[a-zA-Z0-9.-]+\.(com|net|org|info|co\.uk|de|fr|it|es|ru|pl)',
    r'@[a-zA-Z0-9_]+',
]

POLLUTION_PATTERNS = [
    r'\b(free\s+download|download\s+free|gratis\s+download)\b',
    r'\b(320\s*kbps?|128\s*kbps?|256\s*kbps?|mp3|flac|wav)\b',
    r'\b(promo\s+only|promotional|radio\s+edit|club\s+mix)\b',
    r'\b(feat\.?|featuring|ft\.?|vs\.?|versus|pres\.?|presents?)\b',
    r'\[.*?\]|\(.*?\)',  # Remove content in brackets/parentheses
    r'[_\-\s]*\d{2,4}[_\-\s]*',  # Remove year-like numbers
    r'[_\-\s]+(original|remix|edit|mix|version|rework|bootleg)[_\-\s]*',
]

# Common unwanted phrases in metadata
UNWANTED_PHRASES = {
    'various artists', 'va', 'compilation', 'mixed by', 'dj mix',
    'podcast', 'radio show', 'live set', 'mixed live'
}

# Character replacements for better text normalization
CHAR_REPLACEMENTS = {
    ''': "'", ''': "'", '"': '"', '"': '"',
    '–': '-', '—': '-', '…': '...',
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
    'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a',
    'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
    'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o',
    'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
    'ñ': 'n', 'ç': 'c'
}


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to ASCII equivalents where possible
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Apply manual character replacements first
    for old_char, new_char in CHAR_REPLACEMENTS.items():
        text = text.replace(old_char, new_char)
    
    # Normalize unicode
    try:
        # Decompose and remove combining characters
        normalized = unicodedata.normalize('NFD', text)
        ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        return ascii_text
    except Exception:
        return text


def remove_pollution(text: str) -> str:
    """
    Remove common metadata pollution patterns
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    text = text.lower()
    
    # Remove domain patterns
    for pattern in DOMAIN_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove pollution patterns  
    for pattern in POLLUTION_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Remove unwanted phrases
    for phrase in UNWANTED_PHRASES:
        text = text.replace(phrase, ' ')
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def sanitize_tag_value(value: str, max_length: int = 255) -> str:
    """
    Enhanced tag sanitization with pollution removal and normalization
    
    This is the most comprehensive text cleaning function, combining
    pollution removal, unicode normalization, and filesystem safety.
    
    Args:
        value: Text value to sanitize
        max_length: Maximum length for the result
        
    Returns:
        Sanitized and normalized text
    """
    if not value or not isinstance(value, str):
        return ""
    
    # Step 1: Remove pollution patterns
    cleaned = remove_pollution(value)
    
    # Step 2: Normalize unicode
    cleaned = normalize_unicode(cleaned)
    
    # Step 3: Remove control characters and normalize whitespace
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Step 4: Remove leading/trailing punctuation
    cleaned = cleaned.strip('.-_()[]{}"\' ')
    
    # Step 5: Capitalize properly
    if cleaned:
        # Title case but preserve all caps words
        words = cleaned.split()
        result_words = []
        
        for word in words:
            if word.isupper() and len(word) > 3:
                # Keep all-caps words (like DJ, USA, etc.)
                result_words.append(word)
            elif word.lower() in ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                # Keep articles and prepositions lowercase (except at start)
                result_words.append(word.lower() if result_words else word.capitalize())
            else:
                result_words.append(word.capitalize())
        
        cleaned = ' '.join(result_words)
    
    # Step 6: Limit length
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit(' ', 1)[0]  # Break at word boundary
    
    return cleaned


def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Clean text with configurable aggressiveness
    
    Args:
        text: Text to clean
        aggressive: If True, applies more aggressive cleaning
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    if aggressive:
        return sanitize_tag_value(text)
    else:
        # Light cleaning - just normalize and clean whitespace
        cleaned = normalize_unicode(text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned


def generate_clean_filename(artist: str, title: str, year: Optional[int] = None, 
                          extension: str = "mp3", max_length: int = 120) -> str:
    """
    Generate a clean, filesystem-safe filename from metadata
    
    This is the unified filename generation function that combines
    the best of both implementations and adds enhanced cleaning.
    
    Args:
        artist: Artist name
        title: Track title  
        year: Optional year to include
        extension: File extension (without dot)
        max_length: Maximum filename length
        
    Returns:
        Clean filename string
    """
    try:
        # Clean the input components first
        clean_artist = sanitize_tag_value(artist) if artist else ""
        clean_title = sanitize_tag_value(title) if title else ""
        
        # Generate base filename
        if clean_artist and clean_title:
            if year:
                name = f"{clean_artist} - {clean_title} ({year})"
            else:
                name = f"{clean_artist} - {clean_title}"
        elif clean_title:
            name = clean_title
        elif clean_artist:
            name = clean_artist
        else:
            name = "Unknown Track"
        
        # Filesystem sanitization - remove/replace dangerous characters
        # More comprehensive than the original implementation
        filesystem_unsafe = r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]'
        name = re.sub(filesystem_unsafe, '', name)
        
        # Replace whitespace with hyphens in filename
        name = re.sub(r'\s+', '-', name)
        
        # Normalize multiple consecutive hyphens to a single hyphen
        name = re.sub(r'-+', '-', name)
        
        # Remove leading/trailing dots and spaces (Windows compatibility)
        name = name.strip('. ')
        
        # Ensure we don't end with a period (Windows compatibility)
        name = name.rstrip('.')
        
        # Handle length limiting more intelligently
        max_name_length = max_length - len(extension) - 1  # Account for .extension
        
        if len(name) > max_name_length:
            # Try to break at a good point (after artist name)
            if ' - ' in name and len(name.split(' - ')[0]) < max_name_length:
                artist_part = name.split(' - ')[0]
                title_part = name.split(' - ', 1)[1]
                available_title_length = max_name_length - len(artist_part) - 3  # " - "
                
                if available_title_length > 10:  # Ensure minimum title length
                    truncated_title = title_part[:available_title_length].rsplit(' ', 1)[0]
                    name = f"{artist_part} - {truncated_title}"
                else:
                    # Just truncate the whole thing
                    name = name[:max_name_length].rsplit(' ', 1)[0]
            else:
                # Truncate at word boundary
                name = name[:max_name_length].rsplit(' ', 1)[0]
        
        # Final cleanup
        name = name.strip()
        if not name:
            name = "Unknown Track"
        
        # Add extension
        if extension and not name.lower().endswith(f'.{extension.lower()}'):
            name = f"{name}.{extension}"
        
        return name
        
    except Exception as e:
        print(f"Error generating clean filename: {e}")
        fallback = f"Unknown Track.{extension}" if extension else "Unknown Track"
        return fallback


def extract_remix_info(title: str) -> tuple[str, Optional[str]]:
    """
    Extract remix information from track title
    
    Args:
        title: Track title potentially containing remix info
        
    Returns:
        Tuple of (clean_title, remix_info)
    """
    if not title:
        return "", None
    
    # Common remix patterns
    remix_patterns = [
        r'\((.*?(?:remix|mix|edit|rework|version|bootleg).*?)\)$',
        r'\[(.*?(?:remix|mix|edit|rework|version|bootleg).*?)\]$',
        r'\s-\s(.*?(?:remix|mix|edit|rework|version|bootleg).*)$'
    ]
    
    for pattern in remix_patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            clean_title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
            remix_info = match.group(1).strip()
            return clean_title, remix_info
    
    return title, None


def detect_featuring(text: str) -> tuple[str, List[str]]:
    """
    Extract featuring artists from text
    
    Args:
        text: Text potentially containing featuring information
        
    Returns:
        Tuple of (clean_text, list_of_featuring_artists)
    """
    if not text:
        return "", []
    
    # Featuring patterns
    featuring_patterns = [
        r'\s+(feat\.?|featuring|ft\.?)\s+(.+?)(?:\s*[\(\[]|$)',
        r'\s+[\(\[](feat\.?|featuring|ft\.?)\s+([^)]+)[\)\]]',
    ]
    
    featuring_artists = []
    clean_text = text
    
    for pattern in featuring_patterns:
        matches = re.finditer(pattern, clean_text, re.IGNORECASE)
        for match in reversed(list(matches)):  # Reverse to maintain indices
            artists_str = match.group(2)
            # Split by common separators
            artists = [a.strip() for a in re.split(r'[,&]|\sand\s', artists_str) if a.strip()]
            featuring_artists.extend(artists)
            
            # Remove the featuring part from the text
            clean_text = clean_text[:match.start()] + clean_text[match.end():]
    
    return clean_text.strip(), featuring_artists


def standardize_genre(genre: str) -> str:
    """
    Standardize genre names to common formats
    
    Args:
        genre: Raw genre string
        
    Returns:
        Standardized genre name
    """
    if not genre:
        return ""
    
    genre = genre.lower().strip()
    
    # Genre mappings for standardization
    genre_mappings = {
        'edm': 'Electronic Dance Music',
        'electronic dance music': 'Electronic Dance Music',
        'dance': 'Dance',
        'house': 'House', 
        'techno': 'Techno',
        'trance': 'Trance',
        'progressive house': 'Progressive House',
        'deep house': 'Deep House',
        'tech house': 'Tech House',
        'minimal': 'Minimal',
        'dubstep': 'Dubstep',
        'drum & bass': 'Drum & Bass',
        'drum and bass': 'Drum & Bass',
        'dnb': 'Drum & Bass',
        'breaks': 'Breaks',
        'breakbeat': 'Breaks',
        'ambient': 'Ambient',
        'downtempo': 'Downtempo',
        'chillout': 'Chillout',
        'hip hop': 'Hip Hop',
        'hip-hop': 'Hip Hop',
        'rap': 'Hip Hop',
        'r&b': 'R&B',
        'rnb': 'R&B',
        'pop': 'Pop',
        'rock': 'Rock',
        'indie': 'Indie',
        'alternative': 'Alternative',
        'jazz': 'Jazz',
        'blues': 'Blues',
        'funk': 'Funk',
        'soul': 'Soul',
        'disco': 'Disco',
        'reggae': 'Reggae',
        'classical': 'Classical'
    }
    
    return genre_mappings.get(genre, genre.title())


def is_likely_artist_name(text: str) -> bool:
    """
    Determine if text is likely an artist name vs noise
    
    Args:
        text: Text to evaluate
        
    Returns:
        True if likely an artist name
    """
    if not text or len(text) < 2:
        return False
    
    text_lower = text.lower()
    
    # Reject obvious non-artist patterns
    rejection_patterns = [
        r'^(track|song|music|audio|sound)\s*\d+',
        r'^\d{1,3}[_\-\s]*\d',  # Track numbers
        r'^(www|http|ftp)',      # URLs
        r'[_\-]{3,}',           # Excessive separators
        r'^\d{4}$',             # Just a year
        r'^(unknown|various|va|compilation)$'
    ]
    
    for pattern in rejection_patterns:
        if re.search(pattern, text_lower):
            return False
    
    # Positive indicators
    if any(char.isalpha() for char in text):  # Contains letters
        if len([c for c in text if c.isalpha()]) >= len(text) * 0.5:  # At least 50% letters
            return True
    
    return False


# Export commonly used functions
__all__ = [
    'generate_clean_filename',
    'clean_text',
    'sanitize_tag_value',
    'normalize_unicode',
    'remove_pollution',
    'extract_remix_info',
    'detect_featuring',
    'standardize_genre',
    'is_likely_artist_name'
]