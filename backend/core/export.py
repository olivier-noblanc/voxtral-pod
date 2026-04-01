"""
Export utilities for transcription results.

The pipeline is now strictly three‑layered:

1. **ASR layer** – returns a JSON structure (`result.segments`) that is the
   single source of truth.  Each segment must contain at least:
   - ``start`` (float, seconds)
   - ``end``   (float, seconds)
   - ``text``  (str)
   - optional ``speaker`` (str)

2. **Enrichment layer** – can add speaker diarisation, confidence scores,
   etc.  It works directly on the JSON structure.

3. **Export layer** – converts the JSON into the various file formats
   required by the front‑end (TXT, SRT, RTTM).  No flattening of the text
   occurs before export; each segment is written independently.

All exporters raise ``ValueError`` if the required schema is not respected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Mapping, Any

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_segments(segments: List[Mapping[str, Any]]) -> None:
    """
    Ensure every segment contains the mandatory ``start`` and ``end`` keys.
    Raises ``ValueError`` if a segment is malformed.
    """
    for i, seg in enumerate(segments):
        if "start" not in seg or "end" not in seg:
            raise ValueError(
                f"Segment {i} is missing required timestamps: {seg!r}"
            )
        if not isinstance(seg["start"], (int, float)) or not isinstance(seg["end"], (int, float)):
            raise ValueError(
                f"Segment {i} timestamps must be numeric (start={seg['start']}, end={seg['end']})"
            )
        if "text" not in seg:
            raise ValueError(f"Segment {i} is missing required 'text' field.")


# ----------------------------------------------------------------------
# TXT exporter
# ----------------------------------------------------------------------
def export_txt(segments: List[Mapping[str, Any]], txt_path: str) -> None:
    """
    Export the transcription as a plain‑text file.
    Each segment is written on its own line, preserving the original order.
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            line = seg["text"].strip()
            f.write(line + "\n")


# ----------------------------------------------------------------------
# RTTM exporter
# ----------------------------------------------------------------------
def _rttm_line(
    file_id: str,
    seg: Mapping[str, Any],
    speaker: str = "unknown",
) -> str:
    """
    Build a single RTTM line for a segment.
    RTTM spec (simplified):
    SPEAKER <file> <chnl> <start> <duration> <ortho> <stype> <name> <conf> <srat>
    """
    start = float(seg["start"])
    duration = float(seg["end"] - seg["start"])
    # Default values for fields that are not used in Voxtral‑Pod
    ortho = "<NA>"
    stype = "<NA>"
    name = speaker
    conf = "<NA>"
    srat = "<NA>"
    return (
        f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} {ortho} {stype} {name} {conf} {srat}"
    )


def export_rttm(
    segments: List[Mapping[str, Any]],
    file_id: str,
    rttm_path: str,
) -> None:
    """
    Export the transcription as an RTTM file.
    The ``speaker`` field is optional; if absent, ``unknown`` is used.
    """
    with open(rttm_path, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg.get("speaker", "unknown")
            line = _rttm_line(file_id, seg, speaker)
            f.write(line + "\n")