# Verbale — prototipo

Registra il microfono (riunione in vivavoce o audio della stanza), trascrive a
blocchi e scrive tutto in un file Markdown locale, mostrandolo man mano.

## Installazione (Windows, CPU)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python verbale.py
```

Al primo avvio faster-whisper scarica il modello (`small` ≈ 250 MB in int8) in
`%USERPROFILE%\.cache\huggingface`. Il pulsante Avvia resta disabilitato fino a
caricamento completato — pochi secondi dalla seconda volta in poi.

## Output

Ogni sessione crea `sessioni/AAAA-MM-GG_hh-mm-ss/` con:

- `verbale.md` — trascrizione con timestamp, note e marcatori di pausa
- `audio.wav` — audio integrale a 16 kHz mono, per ri-trascrivere con un
  modello migliore senza rifare la riunione

Il file viene scritto e sincronizzato a ogni blocco: se il programma si chiude
a metà riunione, il verbale fino a quel punto è già su disco.

## Configurazione

`config.json` viene creato al primo avvio. Le voci che contano:

| Chiave | Effetto |
|---|---|
| `engine` | `local` (faster-whisper) o `cloud` (endpoint compatibile OpenAI) |
| `local.model` | `tiny` → `large-v3`. Su CPU: `small` è il compromesso sensato, `medium` è nettamente migliore ma può non stare in tempo reale |
| `local.beam_size` | `1` è il più veloce; `5` guadagna un po' di accuratezza a caro prezzo |
| `language` | `it`. Metti `null` per la rilevazione automatica |
| `segmentation.max_chunk_sec` | Abbassalo per vedere il testo comparire prima, alzalo per una trascrizione più accurata |
| `cloud.api_key_env` | Nome della variabile d'ambiente con la chiave, non la chiave stessa |

Per passare al cloud: `"engine": "cloud"` e `set OPENAI_API_KEY=...` prima di
avviare.

## Taratura del modello sulla tua macchina

Prima di usarlo in una riunione vera, misura quanto ci mette:

```bat
python bench.py sessioni\<una-sessione>\audio.wav
```

Se il fattore riportato è sopra 1.0, il modello è più lento del tempo reale e
la coda dei blocchi crescerà durante la riunione: scendi di taglia o passa al
cloud.

## Limiti noti di questo prototipo

- **Solo microfono.** Va bene per il vivavoce; l'audio pulito direttamente da
  Zoom richiede il loopback WASAPI, che si aggiunge cambiando l'apertura dello
  stream in `RecordingSession.start`.
- **Nessuna distinzione dei parlanti.** Il testo è un flusso unico.
- **Il testo non è modificabile** nella finestra, per non dover sincronizzare
  vista e file. Le annotazioni si aggiungono dal campo note.
- **Ritardo di 20-40 secondi** tra parlato e testo: è il costo della
  segmentazione a blocchi, non un bug.
