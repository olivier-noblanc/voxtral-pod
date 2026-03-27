import os
import asyncio
import numpy as np
from backend.config import setup_warnings, get_vram_gb, setup_gpu
from backend.core.audio import decode_audio
from backend.core.merger import assign_speakers_to_words, smooth_micro_turns, build_speaker_segments

# Appel unique à setup_warnings au chargement du module
setup_warnings()


class SotaASR:
    def __init__(self, model_id="whisper", hf_token=None):
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
            print(f"[*] Testing/Auto-Mock mode active. Using Mock mode.")
            self.model_id = "mock"
        elif self.albert_api_key and (self.low_vram or self.no_gpu or model_id == "albert"):
            print("[*] Fallback: Using Albert API for transcription (No GPU or Low VRAM).")
            self.model_id = "albert"
        else:
            self.model_id = model_id

        self.hf_token = hf_token

        # Diarization engine will be instantiated lazily in load()
        self._use_cpu_diarization = self.no_gpu or self.low_vram
        self.diarization_engine = None

        from backend.core.transcription import TranscriptionEngine
        self.transcription_engine = TranscriptionEngine(model_id=self.model_id)

        self._loaded = False

    def load(self):
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
                self.diarization_engine = LightDiarizationEngine()
            else:
                from backend.core.diarization import DiarizationEngine
                # Initialise le moteur de diarisation; il gère lui‑même les fallbacks.
                self.diarization_engine = DiarizationEngine(hf_token=self.hf_token, use_cpu=False)

        # ``load`` may raise if heavy deps are missing; we protect against that.
        try:
            self.diarization_engine.load()
        except Exception as e:
            print(f"[!] Diarization engine load failed: {e}")
            # Fallback to a no‑op engine that returns empty segments.
            class _NoOpEngine:
                def diarize(self, audio_float32, hook=None):
                    return []
                def load(self):
                    pass
            self.diarization_engine = _NoOpEngine()
        self.transcription_engine.load()
        self._loaded = True

    async def process_file(self, audio_path, progress_callback=None):
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
        def diar_hook(step_name, *args, **kwargs):
            if not progress_callback:
                return
            try:
                c_raw = args[0] if len(args) > 0 else kwargs.get("completed", 0)
                t_raw = args[1] if len(args) > 1 else kwargs.get("total")

                def to_scalar(v):
                    if v is None: return 0.0
                    if hasattr(v, "item"): return float(v.item())
                    if isinstance(v, (list, np.ndarray)) and len(v) > 0: return float(v[0])
                    return float(v)

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
            return await self.transcription_engine.transcribe(audio_np) # Return directly for mock
        
        parallel = (self.model_id == "albert")

        if parallel:
            if progress_callback:
                progress_callback("Diarisation et transcription en parallèle...", 10)
            diar_task = asyncio.to_thread(self.diarization_engine.diarize, audio_np, hook=None)
            trans_task = asyncio.to_thread(self.transcription_engine.transcribe, audio_np, progress_callback=None)
            # ``asyncio.gather`` renvoie deux résultats ; on désérialise correctement.
            diar_segments, (words, _) = await asyncio.gather(diar_task, trans_task)  # type: ignore[assignment]
            if progress_callback:
                progress_callback("Fusion des résultats...", 80)
        else:
            if progress_callback:
                progress_callback("Diarisation...", 5)
            diar_segments = await asyncio.to_thread(
                self.diarization_engine.diarize, audio_np, hook=diar_hook
            )
            if progress_callback:
                progress_callback("Transcription...", 45)
            words, _ = await asyncio.to_thread(
                self.transcription_engine.transcribe, audio_np, progress_callback=progress_callback
            )

        # 4. Merge
        words_with_speakers = assign_speakers_to_words(words, diar_segments)
        words_with_speakers = smooth_micro_turns(words_with_speakers)
        segments = build_speaker_segments(words_with_speakers)

        # 5. Format final text
        output_lines = []
        for s in segments:
            output_lines.append(f"[{s['start']:.2f}s -> {s['end']:.2f}s] [{s['speaker']}] {s['text']}")

        return "\n".join(output_lines)
