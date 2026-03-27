import os
import numpy as np
import ffmpeg

audio_path = "tests/samples/french_sample.mp3"
sample_rate = 16000

print(f"Testing decode of {audio_path}...")
try:
    process = (
        ffmpeg.input(audio_path)  # type: ignore[attr-defined]
        .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    out, err = process.communicate()
    if process.returncode != 0:
        print(f"FFmpeg failed with return code {process.returncode}")
        print(f"Stderr: {err.decode()}")
    else:
        audio = np.frombuffer(out, dtype=np.float32)
        print(f"Success! Decoded {len(audio)} samples.")
except Exception as e:
    print(f"Exception: {e}")
