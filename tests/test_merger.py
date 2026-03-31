
from backend.core.merger import (
    assign_speakers_to_words,
    build_speaker_segments,
    smooth_micro_turns,
)


def test_assign_speakers_to_words_basic_overlap() -> None:
    # Words and diarization segments with clear overlap
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]
    diar = [
        (0.0, 0.55, "spk1"),
        (0.55, 1.2, "spk2"),
    ]

    result = assign_speakers_to_words(words, diar)

    assert result[0]["speaker"] == "spk1"
    assert result[1]["speaker"] == "spk2"


def test_assign_speakers_to_words_midpoint_fallback() -> None:
    # No overlap, but midpoint falls inside a segment
    words = [{"word": "test", "start": 1.0, "end": 1.2}]
    diar = [(0.9, 1.5, "spkA")]

    result = assign_speakers_to_words(words, diar)

    assert result[0]["speaker"] == "spkA"


def test_assign_speakers_to_words_nearest_fallback() -> None:
    # Overlap and midpoint miss, nearest turn within 120ms
    words = [{"word": "near", "start": 2.0, "end": 2.1}]
    diar = [(2.3, 2.8, "spkB")]

    result = assign_speakers_to_words(words, diar)

    # Distance from word midpoint (2.05) to diar start (2.3) = 0.25 > 0.12, so fallback to UNKNOWN
    assert result[0]["speaker"] == "UNKNOWN"


def test_smooth_micro_turns_isolated_flip() -> None:
    words = [
        {"word": "a", "speaker": "spk1"},
        {"word": "b", "speaker": "spk2"},
        {"word": "c", "speaker": "spk1"},
    ]
    # The middle word is an isolated micro‑turn and should be relabeled to spk1
    smoothed = smooth_micro_turns(words, max_run_length=1)
    assert [w["speaker"] for w in smoothed] == ["spk1", "spk1", "spk1"]


def test_smooth_micro_turns_no_change_for_long_runs() -> None:
    words = [
        {"word": "x", "speaker": "spk1"},
        {"word": "y", "speaker": "spk1"},
        {"word": "z", "speaker": "spk2"},
        {"word": "w", "speaker": "spk2"},
    ]
    # No isolated single‑word runs, output should be unchanged
    smoothed = smooth_micro_turns(words, max_run_length=1)
    assert [w["speaker"] for w in smoothed] == ["spk1", "spk1", "spk2", "spk2"]


def test_build_speaker_segments_simple() -> None:
    words = [
        {"word": "hello", "speaker": "spk1", "start": 0.0, "end": 0.5},
        {"word": "world", "speaker": "spk1", "start": 0.5, "end": 1.0},
        {"word": "test", "speaker": "spk2", "start": 1.0, "end": 1.5},
    ]

    segments = build_speaker_segments(words)

    assert len(segments) == 2
    assert segments[0]["speaker"] == "spk1"
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 1.0
    assert segments[0]["text"] == "hello world"
    assert segments[1]["speaker"] == "spk2"
    assert segments[1]["start"] == 1.0
    assert segments[1]["end"] == 1.5
    assert segments[1]["text"] == "test"
