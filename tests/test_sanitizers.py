import pytest
from djmusiccleaner.dj_music_cleaner import DJMusicCleaner


@pytest.fixture
def cleaner():
    return DJMusicCleaner()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Test Song", "Test Song"),
        ("masstamilan.com - Song", "Song"),
        ("DJMAZA.COM", None),
        ("", None),
    ],
)
def test_sanitize_tag_value(cleaner, raw, expected):
    assert cleaner.sanitize_tag_value(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Artist1 - Artist2 & Artist3", "Artist1, Artist2, Artist3"),
        ("Artist1, Artist2", "Artist1, Artist2"),
        ("masstamilan.com", None),
        ("", ""),
        (None, None),
    ],
)
def test_normalize_list(cleaner, raw, expected):
    assert cleaner.normalize_list(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2020-05-03", "2020"),
        ("1999", "1999"),
        ("abc 2001 def", "2001"),
        ("2050", None),
        (2015, "2015"),
        ("1899", None),
        (None, None),
    ],
)
def test_parse_year_safely(cleaner, raw, expected):
    assert cleaner.parse_year_safely(raw) == expected
