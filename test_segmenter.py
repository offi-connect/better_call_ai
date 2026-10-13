"""Verifica la segmentazione sui silenzi senza hardware audio ne' GUI."""
import sys, types
import numpy as np

# Stub di tkinter: nel container non esiste, ma serve solo alla GUI.
# L'elenco dei sottomoduli deve coprire tutto cio' che verbale.py importa:
# aggiungendo un "from tkinter import X" senza toccare questa riga, i test
# del segmentatore si rompono con un ImportError che non c'entra con loro.
for name in ("tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.filedialog"):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
sys.modules["tkinter"].Tk = object
sys.modules["tkinter"].Text = object
sys.modules["tkinter"].Canvas = object
sys.modules["tkinter"].StringVar = object
sys.modules["tkinter"].ttk = sys.modules["tkinter.ttk"]
sys.modules["tkinter"].messagebox = sys.modules["tkinter.messagebox"]
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]

from verbale import Segmenter, TARGET_SR, resample_linear, fmt_hms  # noqa: E402

SR = TARGET_SR
BLOCK = 1024


def blocks(seconds: float, amplitude: float):
    """Genera blocchi: rumore di ampiezza data (0 = silenzio digitale)."""
    total = int(seconds * SR)
    rng = np.random.default_rng(0)
    made = 0
    while made < total:
        n = min(BLOCK, total - made)
        yield (rng.standard_normal(n).astype(np.float32) * amplitude)
        made += n


def run(pattern, **kw):
    seg = Segmenter(samplerate=SR, **kw)
    chunks = []
    for seconds, amp in pattern:
        for b in blocks(seconds, amp):
            chunks.extend(seg.feed(b))
    chunks.extend(seg.flush())
    return seg, chunks


print("--- 1. parlato 12s, silenzio 1.5s, parlato 5s ---")
seg, chunks = run([(12, 0.08), (1.5, 0.0005), (5, 0.08)])
for c in chunks:
    print(f"   chunk @ {fmt_hms(c.start_sec)}  durata {c.audio.shape[0]/SR:5.1f}s")
assert len(chunks) == 2, f"attesi 2 chunk, ottenuti {len(chunks)}"
assert 12 < chunks[0].audio.shape[0] / SR < 15, "il primo chunk dovrebbe chiudere sul silenzio"
print("   OK: taglio avvenuto sul silenzio, non a tempo fisso")

print("\n--- 2. solo silenzio per 40s ---")
seg, chunks = run([(40, 0.0005)])
print(f"   chunk prodotti: {len(chunks)}")
assert chunks == [], "il silenzio non deve mai raggiungere il modello"
print("   OK: nessuna inferenza sul silenzio (anti-allucinazione)")

print("\n--- 3. parlato continuo 60s senza pause (taglio forzato) ---")
seg, chunks = run([(60, 0.08)], max_chunk_sec=25.0)
durate = [round(c.audio.shape[0] / SR, 1) for c in chunks]
print(f"   durate: {durate}")
assert all(d <= 25.6 for d in durate), "nessun chunk deve superare il massimo"
assert sum(durate) > 55, "non si deve perdere audio"
print("   OK: taglio forzato rispettato, nessun audio perso")

print("\n--- 4. continuita' della timeline ---")
seg, chunks = run([(10, 0.08), (1.0, 0.0005), (10, 0.08), (1.0, 0.0005), (10, 0.08)])
prev_end = 0.0
for c in chunks:
    assert abs(c.start_sec - prev_end) < 0.2, f"salto nella timeline a {c.start_sec}"
    prev_end = c.start_sec + c.audio.shape[0] / SR
print(f"   {len(chunks)} chunk contigui, durata totale {fmt_hms(seg.elapsed_sec)}")
print("   OK: i timestamp non derivano")

print("\n--- 5. ricampionamento 48k -> 16k ---")
t = np.linspace(0, 1, 48000, endpoint=False)
tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
out = resample_linear(tone, 48000, 16000)
print(f"   {tone.shape[0]} campioni -> {out.shape[0]}, ampiezza max {out.max():.3f}")
assert out.shape[0] == 16000 and 0.9 < out.max() <= 1.0
print("   OK")

print("\n--- 6. soglia adattiva al rumore di fondo ---")
seg = Segmenter(samplerate=SR)
for b in blocks(3.0, 0.003):  # stanza rumorosa, nessun parlato
    seg.feed(b)
print(f"   noise floor stimato {seg._noise_floor:.5f}, soglia {seg.threshold:.5f}")
assert seg.threshold > 0.003, "la soglia deve salire sopra il rumore ambientale"
print("   OK: la soglia si alza in ambiente rumoroso")

print("\nTutti i test superati.")
