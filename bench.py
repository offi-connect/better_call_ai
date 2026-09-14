"""Misura la velocita' di trascrizione sulla macchina corrente.

Uso:  python bench.py <file.wav> [modello]
"""
import sys, time, wave
import numpy as np

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "small"

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / sr
    print(f"File: {path}  durata {duration:.1f}s a {sr} Hz")

    from faster_whisper import WhisperModel
    print(f"Carico {model_name} (cpu/int8)...")
    t0 = time.time()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"Caricato in {time.time() - t0:.1f}s")

    t0 = time.time()
    segments, _ = model.transcribe(
        audio, language="it", beam_size=1, vad_filter=True,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments)
    elapsed = time.time() - t0

    factor = elapsed / duration if duration else 0
    print(f"\nTrascritto in {elapsed:.1f}s  ->  fattore {factor:.2f}x tempo reale")
    print("OK per l'uso live." if factor < 0.7 else
          "Troppo lento per il live: scendi di taglia o passa al cloud.")
    print(f"\nParole: {len(text.split())}\n")
    print(text[:600] + ("..." if len(text) > 600 else ""))
    return 0

if __name__ == "__main__":
    sys.exit(main())
