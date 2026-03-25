def assign_speakers_to_words(words, diarization_segments):
    """
    Assign a speaker to each word based on diarization segments.
    Reference logic from TranscriptionSuite.
    """
    if not words or not diarization_segments:
        return words

    # Pre-sort diarization segments by start time
    diar = sorted(diarization_segments, key=lambda s: s[0])
    result = []
    prev_speaker = None
    prev_end = 0.0

    for w in words:
        w_start = w["start"]
        w_end = w["end"]
        w_mid = (w_start + w_end) / 2.0

        # Inflate word interval by padding (40ms jitter tolerance)
        padded_start = w_start - 0.040
        padded_end = w_end + 0.040

        best_speaker = None
        best_overlap = 0.0
        midpoint_speaker = None
        nearest_speaker = None
        nearest_distance = None

        for ds, de, dspk in diar:
            # Pass 1: overlap
            overlap = max(0.0, min(padded_end, de) - max(padded_start, ds))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dspk

            # Pass 2: midpoint
            if midpoint_speaker is None and ds <= w_mid <= de:
                midpoint_speaker = dspk

            # Pass 3: nearest turn
            if w_mid < ds: dist = ds - w_mid
            elif w_mid > de: dist = w_mid - de
            else: dist = 0.0

            if dist <= 0.120:  # 120ms fallback
                if nearest_distance is None or dist < nearest_distance:
                    nearest_distance = dist
                    nearest_speaker = dspk

        # Fallback chain
        if best_speaker is not None and best_overlap > 0.0: chosen = best_speaker
        elif midpoint_speaker is not None: chosen = midpoint_speaker
        elif nearest_speaker is not None: chosen = nearest_speaker
        elif prev_speaker is not None and (w_start - prev_end) <= 0.200: chosen = prev_speaker
        else: chosen = "UNKNOWN"

        w["speaker"] = chosen
        result.append(w)
        prev_speaker = chosen
        prev_end = w_end

    return result


def smooth_micro_turns(words, max_run_length=1):
    """
    Relabel isolated micro-turns (single word flips) to surrounding speaker.
    Code from TranscriptionSuite.
    """
    if len(words) < 3:
        return words

    # Build runs: [speaker, start_idx, length]
    runs = []
    i = 0
    while i < len(words):
        spk = words[i].get("speaker", "UNKNOWN")
        j = i + 1
        while j < len(words) and words[j].get("speaker", "UNKNOWN") == spk:
            j += 1
        runs.append([spk, i, j - i])
        i = j

    # Identify runs to relabel
    for r_idx in range(1, len(runs) - 1):
        spk, start_idx, length = runs[r_idx]
        if length > max_run_length:
            continue
        prev_spk = runs[r_idx - 1][0]
        next_spk = runs[r_idx + 1][0]
        if prev_spk == next_spk and prev_spk != spk:
            for wi in range(start_idx, start_idx + length):
                words[wi]["speaker"] = prev_spk

    return words


def build_speaker_segments(words_with_speakers):
    """
    Group words into contiguous speaker segments.
    """
    if not words_with_speakers:
        return []

    segments = []
    current_speaker = words_with_speakers[0]["speaker"]
    current_start = words_with_speakers[0]["start"]
    current_words = [words_with_speakers[0]["word"]]

    for i in range(1, len(words_with_speakers)):
        w = words_with_speakers[i]
        if w["speaker"] == current_speaker:
            current_words.append(w["word"])
        else:
            # Fix : " ".join() au lieu de "".join() pour éviter les mots collés
            text = " ".join(current_words).strip()
            segments.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": words_with_speakers[i - 1]["end"],
                "text": text
            })
            current_speaker = w["speaker"]
            current_start = w["start"]
            current_words = [w["word"]]

    # Final flush
    text = " ".join(current_words).strip()
    segments.append({
        "speaker": current_speaker,
        "start": current_start,
        "end": words_with_speakers[-1]["end"],
        "text": text
    })

    return segments
