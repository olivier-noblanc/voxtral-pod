import asyncio
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np

from backend.config import get_vram_gb, setup_gpu, setup_warnings
from backend.core.audio import decode_audio
from backend.core.merger import (
    assign_speakers_to_words,
    build_speaker_segments,
    smooth_micro_turns,
)

# Ensure NNPACK is disabled for engine operations
os.environ.setdefault("DISABLE_NNPACK", "1")

@dataclass
class TranscriptionResult:
    transcript: str
    segments: List[Dict[str, Any]]

    def to_rttm(self, file_id: str) -> str:
        """
        Génère le contenu au format RTTM (Rich Transcription Time Marked).
        Format: SPEAKER <file> <chnl> <start> <duration> <ortho> <stype> <name> <conf> <srat>
        """
        lines = []
        for s in self.segments:
            start = s["start"]
            duration = s["end"] - start
            speaker = s["speaker"]
            # Formatting as required by RTTM spec
            lines.append(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Appel unique à setup_warnings au chargement du module
setup_warnings()
# Optimisations matérielles (incluant l'inhibition NNPACK)
setup_gpu()


class SotaASR:
    def __init__(self, model_id: str = "whisper", hf_token: Optional[str] = None) -> None:
        self.albert_api_key = os.getenv("ALBERT_API_KEY")

        # 1. Hardware detection
        # Lazy import torch for hardware detection
        import torch
        cuda_available = torch.cuda.is_available()
        vram = get_vram_gb()
        self.no_gpu = not cuda_available
        self.low_vram = (vram < 4.0) if cuda_available else True

        if self.no_gpu:
            print("[*] No compatible CUDA device found. Using CPU optimized mode.")
        elif self.low_vram:
            print(f"[*] Low VRAM detected ({vram:.1f}GB). Optimizing for low resources.")

        # 2. Strategy selection
        # Auto-mock if TESTING is enabled OR (VRAM < 7GB and no Albert API key)
        if os.getenv("TESTING") == "1" or (not self.albert_api_key and vram < 7.0):
            print("[*] Testing/Auto-Mock mode active. Using Mock mode.")
            self.model_id = "mock"
        elif self.albert_api_key and (self.low_vram or self.no_gpu or model_id == "albert"):
            print("[*] Fallback: Using Albert API for transcription (No GPU or Low VRAM).")
            self.model_id = "albert"
        else:
            self.model_id = model_id

        self.hf_token = hf_token

        # Diarization engine will be instantiated lazily in load()
        self._use_cpu_diarization = self.no_gpu or self.low_vram
        self.diarization_engine: Any = None

        from backend.core.transcription import TranscriptionEngine
        self.transcription_engine = TranscriptionEngine(model_id=self.model_id)

        self._loaded: bool = False

    def load(self) -> None:
        """
        Charge les sous‑composants (diarisation et transcription) uniquement
        lorsqu'ils sont réellement nécessaires. Cela évite d'importer des
        dépendances lourdes (ex. resemblyzer) pendant les tests qui ne les
        utilisent pas.
        """
        if self._loaded:
            return

        # Instancier le moteur de diarisation si ce n'est pas déjà fait
        if self.diarization_engine is None:
            if self._use_cpu_diarization:
                from backend.core.diarization_cpu import LightDiarizationEngine
                self.diarization_engine = LightDiarizationEngine()  # type: ignore[assignment]
            else:
                from backend.core.diarization import DiarizationEngine
                # Initialise le moteur de diarisation; il gère lui‑même les fallbacks.
                self.diarization_engine = DiarizationEngine(hf_token=self.hf_token, use_cpu=False)  # type: ignore[assignment]

        # ``load`` may raise if heavy deps are missing; we protect against that.
        try:
            if self.diarization_engine is not None:
                # Use Any-casting to call load() on a dynamic engine
                cast(Any, self.diarization_engine).load()
        except Exception as e:
            print(f"[!] Diarization engine load failed: {e}")
            # Fallback to a no‑op engine that returns empty segments.
            class _NoOpEngine:
                def diarize(self, audio_float32: np.ndarray, hook: Any = None) -> list[tuple[float, float, str]]:
                    return []
                def load(self) -> None:
                    pass
            self.diarization_engine = _NoOpEngine()  # type: ignore[assignment]
        self.transcription_engine.load()
        self._loaded = True

    async def process_file(
        self, audio_path: str, progress_callback: Any = None
    ) -> TranscriptionResult:
        """
        Full pipeline: Decode -> Diarize -> Transcribe -> Merge -> Format
        """
        if not self._loaded:
            self.load()

        # 1. Decode
        if progress_callback:
            progress_callback("Décodage audio...", 2)
        audio_np = await asyncio.to_thread(decode_audio, audio_path)

        # 2. Build diarization progress hook
        def diar_hook(step_name: str, *args: Any, **kwargs: Any) -> None:
            if not progress_callback:
                return
            try:
                c_raw: Any = args[0] if len(args) > 0 else kwargs.get("completed", 0)
                t_raw: Any = args[1] if len(args) > 1 else kwargs.get("total")

                def to_scalar(v: Any) -> float:
                    if v is None:
                        return 0.0
                    if hasattr(v, "item"): 
                        try:
                            item = v.item() # type: ignore
                            return float(item)
                        except Exception:
                            pass
                    if isinstance(v, (list, np.ndarray)):
                        if len(v) > 0:
                            return float(v[0])
                        return 0.0
                    try:
                        return float(v) # type: ignore
                    except (TypeError, ValueError):
                        return 0.0

                c_val = to_scalar(c_raw)
                t_val = to_scalar(t_raw)

                display_step = step_name
                if "segmentation" in step_name.lower():
                    display_step = "Détection voix"
                elif "embedding" in step_name.lower():
                    display_step = "Identification speakers"

                if t_val > 0:
                    sub_pct = int((c_val / t_val) * 40)
                    progress_callback(f"Diarisation : {display_step}", 5 + sub_pct)
                else:
                    progress_callback(f"Diarisation : {display_step}...", 5)
            except Exception:
                pass

        # 3. Diarize + Transcribe (parallel pour Albert, séquentiel sinon)
        if self.model_id == "mock":
            # TranscriptionEngine.transcribe returns (words, duration, used_fallback)
            words, _, _ = await asyncio.to_thread(self.transcription_engine.transcribe, audio_np)
            mock_segments = [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": " ".join([w['word'] for w in words])}
            ]
            transcript = "\n".join([f"[{s['start']:.2f}s -> {s['end']:.2f}s] [{s['speaker']}] {s['text']}" for s in mock_segments])
            return TranscriptionResult(transcript=transcript, segments=mock_segments)
        
        parallel = (self.model_id == "albert")

        if parallel:
            if progress_callback:
                progress_callback("Diarisation et transcription en parallèle...", 10)
            if self.diarization_engine is not None:
                diar_task = asyncio.to_thread(self.diarization_engine.diarize, audio_np, hook=None)
                trans_task = asyncio.to_thread(self.transcription_engine.transcribe, audio_np, progress_callback=None)
                # ``asyncio.gather`` renvoie deux résultats ; on désérialise correctement.
                diar_segments_raw, trans_result = await asyncio.gather(diar_task, trans_task)
                diar_segments = cast(list[tuple[float, float, str]], diar_segments_raw)
                words, _, used_fallback = cast(tuple[List[Dict[str, Any]], float, bool], trans_result)
            else:
                words, _, used_fallback = await asyncio.to_thread(self.transcription_engine.transcribe, audio_np, progress_callback=None)
                diar_segments = []  # type: ignore[var-annotated]
            if progress_callback:
                progress_callback("Fusion des résultats...", 80)
        else:
            if progress_callback:
                progress_callback("Diarisation...", 5)
            if self.diarization_engine is not None:
                diar_segments = await asyncio.to_thread(
                    self.diarization_engine.diarize, audio_np, hook=diar_hook
                )
            else:
                diar_segments = []
            if progress_callback:
                progress_callback("Transcription...", 45)
            words, _, used_fallback = await asyncio.to_thread(
                self.transcription_engine.transcribe, audio_np, progress_callback=progress_callback
            )

        # 4. Merge
        print(f"[*] Post-traitement: Alignement de {len(words)} mots avec {len(diar_segments)} segments de locuteurs...")
        words_with_speakers = assign_speakers_to_words(words, diar_segments)
        
        print("[*] Post-traitement: Lissage des micro-tours de parole...")
        words_with_speakers = smooth_micro_turns(words_with_speakers)
        
        print("[*] Post-traitement: Reconstruction des segments par locuteur...")
        segments = build_speaker_segments(words_with_speakers)
        print(f"[*] Post-traitement: {len(segments)} segments finaux générés.")

        # 5. Format final text
        print("[*] Post-traitement: Formattage du texte final...")
        output_lines = []
        if used_fallback:
            output_lines.append("[SYSTÈME] : Transcrit via moteur local (Fallback CPU - Quota Albert atteint)")
            output_lines.append("")
            
        for s in segments:
            output_lines.append(f"[{s['start']:.2f}s -> {s['end']:.2f}s] [{s['speaker']}] {s['text']}")

        print("[*] Post-traitement terminé. Transcript prêt.")
        return TranscriptionResult(transcript="\n".join(output_lines), segments=segments)
