import re
from html import escape as html_escape

def format_transcription(line: str) -> str:
    """
    Convert a raw transcription line into an HTML fragment.
    Robustly extracts timestamps and speakers regardless of their position.
    """
    line = line.strip()
    if not line:
        return ""

    text = line
    start, end, speaker = None, None, None

    # 1. Extraire les timestamps [0.00s -> 5.00s]
    time_match = re.search(r"\[([0-9.]+)s\s*->\s*([0-9.]+)s\]", text)
    if time_match:
        start, end = time_match.groups()
        text = text.replace(time_match.group(0), "").strip()

    # 2. Extraire le speaker [SPEAKER_XX]
    # On cherche un bloc entre crochets restant après extraction du temps
    speaker_match = re.search(r"\[([^\]]+)\]", text)
    if speaker_match:
        speaker = speaker_match.group(1)
        text = text.replace(speaker_match.group(0), "").strip()

    # Nettoyage final du texte
    text = re.sub(r"\s+", " ", text).strip()

    # Construction du HTML
    has_header = (start and end) or speaker
    html = '<div class="segment">'
    if has_header:
        html += '<div class="segment-header">'
        if start and end:
            html += f'<span class="segment-time">[{start}s → {end}s]</span> '
        if speaker:
            html += f'<span class="segment-speaker" data-speaker="{html_escape(speaker)}">{html_escape(speaker)}</span>'
        html += '</div>'
    
    html += f'<div class="segment-text">{html_escape(text)}</div>'
    html += '</div>'
    
    return html
