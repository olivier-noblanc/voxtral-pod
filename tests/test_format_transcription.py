# Tests pour format_transcription

# The function under test is expected to be defined in backend.main
# It should take a single string (the raw transcription line) and return an HTML snippet.
# The exact implementation is not provided here; the tests describe the expected behavior.

from backend.utils import format_transcription


def test_format_timestamp_speaker_text():
    """
    Input  : "[0.50s -> 1.20s] [SPEAKER_00] Bonjour"
    Expected: HTML structure with div.segment, span.segment-time and span.segment-speaker with data-speaker
    """
    raw = "[0.50s -> 1.20s] [SPEAKER_00] Bonjour"
    html = format_transcription(raw)

    assert '<div class="segment">' in html
    assert 'class="segment-time">' in html
    assert '[0.50s → 1.20s]' in html
    assert 'class="segment-speaker"' in html
    assert 'data-speaker="SPEAKER_00"' in html
    assert "Bonjour" in html


def test_format_timestamp_no_speaker():
    """
    Input  : "[0.50s -> 1.20s] Bonjour"
    Expected: HTML structure with time but NO speaker span
    """
    raw = "[0.50s -> 1.20s] Bonjour"
    html = format_transcription(raw)

    assert 'class="segment-time"' in html
    assert 'class="segment-speaker"' not in html
    assert "Bonjour" in html


def test_format_speaker_no_timestamp():
    """
    Input  : "[SPEAKER_01] Hello"
    Expected: HTML structure with speaker but NO time span
    """
    raw = "[SPEAKER_01] Hello"
    html = format_transcription(raw)

    assert 'class="segment-time"' not in html
    assert 'class="segment-speaker"' in html
    assert 'data-speaker="SPEAKER_01"' in html
    assert "Hello" in html


def test_format_plain_text_fallback():
    """
    Input without metadata should be rendered as plain text inside a div.segment
    """
    raw = "Just a plain line without any brackets"
    html = format_transcription(raw)

    assert "Just a plain line without any brackets" in html
    assert 'class="segment-time"' not in html
    assert 'class="segment-speaker"' not in html
    assert 'class="segment-text"' in html


def test_format_escapes_html_injection():
    """
    Ensure HTML tags in text are escaped.
    """
    raw = "[0.00s -> 1.00s] [SPK] <script>alert(1)</script>"
    html = format_transcription(raw)

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>" not in html


def test_format_timestamp_text_speaker():
    """
    Input  : "[5.10s -> 8.00s] Et voici un speaker à la fin. [SPEAKER_02]"
    Expected: HTML structure with correct timestamp and speaker extraction
    """
    raw = "[5.10s -> 8.00s] Et voici un speaker à la fin. [SPEAKER_02]"
    html = format_transcription(raw)

    assert 'class="segment-time"' in html
    assert '[5.10s → 8.00s]' in html
    assert 'class="segment-speaker"' in html
    assert 'data-speaker="SPEAKER_02"' in html
    assert "Et voici un speaker à la fin." in html


def test_format_empty_input():
    """
    Empty string should return an empty string.
    """
    html = format_transcription("")
    assert html == ""