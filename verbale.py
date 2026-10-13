"""
Verbale - registratore e trascrittore di riunioni (prototipo).

Cattura il microfono (riunione in vivavoce o audio della stanza), trascrive a
blocchi tagliati sui silenzi e scrive il testo in un file Markdown locale,
mostrandolo man mano nella finestra.

Avvio:  python verbale.py
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Le dipendenze pesanti sono importate in modo tollerante, cosi' il modulo resta
# importabile (e testabile) anche su una macchina senza audio o senza modello.
try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"

DEFAULT_CONFIG = {
    "engine": "local",  # "local" | "cloud"
    "language": "it",
    "local": {
        # Misurato su 120s di telefonata vera (i7-1165G7, cpu/int8):
        # small 0,17x tempo reale, medium 0,53x, large-v3 0,89x.
        # medium tiene il passo con margine e capisce frasi che small perde
        # ("ganci traino" contro "canci trinos"). large-v3 e' piu' accurato
        # ancora ma con l'11% di margine non e' sicuro per la diretta.
        "model": "medium",  # tiny | base | small | medium | large-v3
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 1,
        "vad_filter": True,
    },
    "cloud": {
        "base_url": "https://api.openai.com/v1",
        "model": "whisper-1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_sec": 120,
    },
    "audio": {
        "device": None,  # indice del dispositivo di input, None = predefinito
        "samplerate": 16000,
        "blocksize": 1024,
    },
    "segmentation": {
        "min_chunk_sec": 8.0,
        "max_chunk_sec": 25.0,
        "silence_hold_sec": 0.7,
        "min_speech_sec": 0.6,
        "noise_multiplier": 3.0,
        "absolute_floor": 0.004,
    },
    "summary": {
        # "auto" sceglie in base alla lunghezza: il modello locale se la
        # trascrizione e' corta, il cloud se e' lunga. Vedi scegli_motore().
        "engine": "auto",  # "auto" | "local" | "cloud"
        "soglia_parole": 700,
        "local": {
            # gemma3:4b al posto di qwen2.5:3b: su trascrizioni vere qwen
            # storpiava le parole italiane ("Planifica", "maretta") e emetteva
            # le intestazioni di sezione come voci di elenco. Costa 8 secondi
            # in piu' per blocco, ne vale la pena.
            "model": "gemma3:4b",
            "host": "http://127.0.0.1:11434",
            "num_ctx": 4096,
            "parole_per_blocco": 550,
        },
        "cloud": {
            # Quale servizio remoto usare. La tendina dell'interfaccia resta a
            # tre posizioni (auto/locale/remoto): il fornitore si scegli qui.
            "provider": "gemini",  # "gemini" | "anthropic"
            "gemini": {
                "model": "gemini-3.5-flash-lite",
                "api_key_env": "GEMINI_API_KEY",
                "max_tokens": 1024,
                # L'estrazione dei punti salienti non ha bisogno di ragionare a
                # lungo, e sui modelli Gemini il ragionamento consuma token di
                # uscita: tenerlo basso costa meno senza perdere qualita'.
                "thinking_level": "low",
                # Sul piano gratuito Google usa i contenuti per migliorare i
                # propri prodotti; su quello a pagamento no. Se sei sul piano
                # gratuito lascia True e il banner lo dichiarera'.
                "piano_gratuito": True,
            },
            "anthropic": {
                "model": "claude-haiku-4-5-20251001",
                "api_key_env": "ANTHROPIC_API_KEY",
                "max_tokens": 1024,
            },
        },
    },
    "output_dir": "sessioni",
}

TARGET_SR = 16000  # Whisper lavora a 16 kHz


# --------------------------------------------------------------------------- #
# Configurazione
# --------------------------------------------------------------------------- #

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config() -> dict:
    """Legge config.json, creandolo con i valori predefiniti se manca."""
    if CONFIG_PATH.exists():
        try:
            user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return _deep_merge(DEFAULT_CONFIG, user_cfg)
        except Exception:
            pass
    CONFIG_PATH.write_text(
        json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dict(DEFAULT_CONFIG)


# --------------------------------------------------------------------------- #
# Segmentazione: taglia il flusso audio sui silenzi
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    start_sec: float
    audio: np.ndarray


@dataclass
class Segmenter:
    """
    Accumula blocchi audio e chiude un segmento quando trova un silenzio
    abbastanza lungo, purche' il segmento abbia raggiunto la durata minima.
    Tagliare sui silenzi invece che a tempo fisso evita di spezzare le parole.

    I segmenti che non contengono abbastanza parlato vengono scartati: questo
    e' cio' che impedisce a Whisper di allucinare frasi inventate sul silenzio.
    """

    samplerate: int = TARGET_SR
    min_chunk_sec: float = 8.0
    max_chunk_sec: float = 25.0
    silence_hold_sec: float = 0.7
    min_speech_sec: float = 0.6
    noise_multiplier: float = 3.0
    absolute_floor: float = 0.004

    _buffer: list[np.ndarray] = field(default_factory=list, init=False)
    _buffer_frames: int = field(default=0, init=False)
    _speech_frames: int = field(default=0, init=False)
    _silence_frames: int = field(default=0, init=False)
    _noise_floor: float = field(default=0.002, init=False)
    _timeline_frames: int = field(default=0, init=False)
    _chunk_start_frame: int = field(default=0, init=False)
    last_rms: float = field(default=0.0, init=False)

    @property
    def threshold(self) -> float:
        return max(self.absolute_floor, self._noise_floor * self.noise_multiplier)

    def feed(self, block: np.ndarray) -> list[Chunk]:
        """Aggiunge un blocco e restituisce gli eventuali segmenti chiusi."""
        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
        self.last_rms = rms

        is_speech = rms > self.threshold
        if not is_speech:
            # Il rumore di fondo si adatta lentamente, solo durante i silenzi.
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * rms

        n = block.shape[0]
        self._buffer.append(block)
        self._buffer_frames += n
        self._timeline_frames += n
        if is_speech:
            self._speech_frames += n
            self._silence_frames = 0
        else:
            self._silence_frames += n

        buffer_sec = self._buffer_frames / self.samplerate
        silence_sec = self._silence_frames / self.samplerate

        closed = buffer_sec >= self.min_chunk_sec and silence_sec >= self.silence_hold_sec
        forced = buffer_sec >= self.max_chunk_sec

        if closed or forced:
            chunk = self._close()
            return [chunk] if chunk else []
        return []

    def flush(self) -> list[Chunk]:
        """Chiude il segmento in corso (su pausa o stop)."""
        chunk = self._close()
        return [chunk] if chunk else []

    def _close(self) -> Chunk | None:
        if not self._buffer:
            return None
        audio = np.concatenate(self._buffer)
        speech_sec = self._speech_frames / self.samplerate
        start_sec = self._chunk_start_frame / self.samplerate

        self._buffer = []
        self._buffer_frames = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._chunk_start_frame = self._timeline_frames

        if speech_sec < self.min_speech_sec:
            return None  # praticamente silenzio: non vale una inferenza
        return Chunk(start_sec=start_sec, audio=audio)

    @property
    def elapsed_sec(self) -> float:
        return self._timeline_frames / self.samplerate


# --------------------------------------------------------------------------- #
# Motori di trascrizione
# --------------------------------------------------------------------------- #

class Transcriber:
    name = "base"

    def load(self) -> None:
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        raise NotImplementedError


class LocalWhisper(Transcriber):
    """faster-whisper su CPU. Nessuna rete, nessuna chiave."""

    def __init__(self, cfg: dict, language: str):
        self.cfg = cfg
        self.language = language
        self.model = None
        self.name = f"faster-whisper {cfg['model']} ({cfg['device']}/{cfg['compute_type']})"

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.cfg["model"],
            device=self.cfg["device"],
            compute_type=self.cfg["compute_type"],
        )

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        if self.model is None:
            raise RuntimeError("Modello non caricato")
        segments, _info = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=self.cfg.get("beam_size", 1),
            vad_filter=self.cfg.get("vad_filter", True),
            # Fondamentale a blocchi: senza questo il modello ripete l'ultima
            # frase all'infinito quando un blocco e' poco intelligibile.
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class CloudWhisper(Transcriber):
    """Endpoint compatibile OpenAI /audio/transcriptions."""

    def __init__(self, cfg: dict, language: str):
        self.cfg = cfg
        self.language = language
        self.name = f"cloud {cfg['model']}"
        self._key = None

    def load(self) -> None:
        self._key = os.environ.get(self.cfg["api_key_env"], "").strip()
        if not self._key:
            raise RuntimeError(
                f"Variabile d'ambiente {self.cfg['api_key_env']} non impostata."
            )

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        import io

        import requests

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(float_to_pcm16(audio).tobytes())
        buf.seek(0)

        resp = requests.post(
            f"{self.cfg['base_url'].rstrip('/')}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self._key}"},
            files={"file": ("chunk.wav", buf, "audio/wav")},
            data={"model": self.cfg["model"], "language": self.language},
            timeout=self.cfg.get("timeout_sec", 120),
        )
        resp.raise_for_status()
        return (resp.json().get("text") or "").strip()


def build_transcriber(cfg: dict) -> Transcriber:
    engine = cfg.get("engine", "local")
    if engine == "cloud":
        return CloudWhisper(cfg["cloud"], cfg["language"])
    return LocalWhisper(cfg["local"], cfg["language"])


# --------------------------------------------------------------------------- #
# Punti salienti: estrazione di decisioni e azioni dal verbale
# --------------------------------------------------------------------------- #

# Il formato e' imposto in modo rigido perche' i modelli piccoli, lasciati
# liberi, scivolano nell'elenco numerato e nelle deduzioni inventate.
SUMMARY_SYSTEM_PROMPT = (
    "Sei un verbalista. Ricevi la trascrizione di una riunione o conversazione "
    "in italiano e ne estrai i punti salienti.\n\n"
    "Regole ferree:\n"
    "- Riporta SOLO cio' che e' stato detto esplicitamente. Non dedurre, non "
    "interpretare, non aggiungere raccomandazioni.\n"
    "- Non attribuire mai un compito a qualcuno il cui nome non e' stato "
    "pronunciato, e non inventare mai una scadenza. Se non sono stati detti, "
    "scrivi solo l'azione e basta.\n"
    "- Conserva numeri, date, giorni della settimana, importi e nomi esattamente "
    "come compaiono: se il testo dice martedi' non scrivere mercoledi'.\n"
    "- Non introdurre nomi propri che non siano nella trascrizione: niente nomi "
    "di aziende, prodotti, servizi o fornitori che tu abbia dedotto o "
    "completato. Se la trascrizione e' confusa o tronca, riporta le parole che "
    "ci sono anche se incomplete, senza ricostruire il nome che immagini.\n"
    "- Scrivi in italiano corretto. Non inventare parole.\n"
    "- Se una sezione non ha contenuti, ometti del tutto la sezione.\n"
    "- Non scrivere mai una voce che commenta l'assenza di contenuto.\n\n"
    "Sezioni possibili, nell'ordine, usando solo quelle che servono:\n\n"
    "Decisioni\n"
    "- cosa e' stato deciso\n\n"
    "Da decidere\n"
    "- questione lasciata aperta\n\n"
    "Azioni\n"
    "- cosa fare. Aggiungi il responsabile e la scadenza solo se dichiarati.\n"
    "  Metti qui solo i compiti che qualcuno si e' impegnato a svolgere. Una\n"
    "  funzionalita' descritta o un'ipotesi discussa non e' un'azione: quella\n"
    "  va fra i temi. Nel dubbio, non e' un'azione.\n\n"
    "Temi discussi\n"
    "- argomento\n\n"
    "Usa il trattino come segno di elenco. Niente titoli in grassetto, niente "
    "numerazione, niente premessa, niente conclusione."
)

# Prompt per la lettura a blocchi (solo motore locale). Attenzione: qui non va
# mai offerta una via d'uscita del tipo "se non trovi nulla rispondi NIENTE".
# Misurato: con quella scorciatoia qwen2.5:3b la imbocca anche su blocchi che
# contengono tre decisioni esplicite, e il richiamo crolla a zero.
CHUNK_EXTRACT_PROMPT = (
    "Ricevi un pezzo di trascrizione di una riunione in italiano.\n"
    "Trascrivi ogni decisione presa, ogni compito assegnato e ogni scadenza, "
    "data o importo nominato.\n"
    "Copia numeri, date, importi e nomi esattamente come compaiono nel testo.\n"
    "Non riassumere le opinioni e le chiacchiere, ma non scartare mai un fatto.\n"
    "Rispondi solo con righe che iniziano con un trattino."
)

CHUNK_MERGE_PROMPT = (
    "Ricevi appunti presi da spezzoni diversi della stessa riunione.\n"
    "Uniscili in un verbale unico, eliminando i doppioni.\n"
    "Non aggiungere nulla che non sia negli appunti.\n"
    "Conserva numeri, date, importi e nomi.\n\n"
    "Formato esatto:\n\n"
    "Decisioni\n- <decisione>\n\nAzioni\n- <cosa> - <chi> - <entro quando>\n\n"
    "Usa il trattino. Niente premessa, niente conclusione."
)


# Una voce che si limita a dire "qui non c'e' niente" va tolta: il posto giusto
# per dirlo e' l'assenza della sezione. I motivi sono elencati per esteso invece
# che con un "non" generico, perche' "Non si procede con la migrazione" e' una
# decisione legittima e non va buttata via.
_VOCE_VUOTA = re.compile(r"^[-*\s]*$")
_VOCE_COMMENTO = re.compile(
    r"non (e'|è|e) stat[oa] (detto|deciso|indicato|stabilito|preso)"
    r"|non sono stat[ei] (fatte|prese|presi|indicate|definite)"
    r"|nessun[ao]? (decisione|azione|tema|punto|elemento|compito)"
    r"|non ci sono (stat[ei] )?(decisioni|azioni|temi|punti)"
    r"|non (e'|è|e) emers[oa]"
    r"|nulla di (specifico|rilevante|particolare|concreto)"
    r"|non risulta (nulla|niente)"
    r"|^\s*[-*]?\s*(niente|nulla|nessun[ao]?|non pervenut[oa]|n\.?d\.?)\s*\.?\s*$",
    re.IGNORECASE,
)

# Le intestazioni previste dal prompt. Un modello puo' emetterle come voci di
# elenco ("- Decisioni**"): vanno riconosciute come titoli, altrimenti la
# struttura collassa in una lista piatta.
_INTESTAZIONI = {"decisioni", "da decidere", "azioni", "temi discussi"}

# Righe che ricopiano alla lettera i segnaposto descrittivi del prompt invece
# di sostituirli con il contenuto.
_ECO_MODELLO = re.compile(
    r"^(cosa (e'|è|e) stato deciso"
    r"|questione lasciata aperta"
    r"|cosa fare"
    r"|argomento"
    r"|decisione"
    r"|tema)\s*[:.]?\s*$",
    re.IGNORECASE,
)


# Parole alfabetiche, accenti italiani compresi.
_TOKEN_PAROLA = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’]*")
# Fine di frase: dopo questi segni la parola seguente torna maiuscola per
# ragioni grammaticali, non perche' sia un nome proprio.
_FINE_FRASE = re.compile(r"(?<=[.!?:;])\s+")


# Scioglimento di un acronimo fra parentesi: "FOMO (Fear Of Missing Out)".
# Le maiuscole la' dentro traducono una sigla, non nominano un'entita'.
_GLOSSA_ACRONIMO = re.compile(r"\b[A-Z]{2,}\s*\(([^)]*)\)")


def _candidati_nomi(testo: str) -> list[str]:
    """I token che si comportano da nome proprio, in ordine di comparsa.

    Il criterio e' la POSIZIONE, non un elenco di parole italiane. Una prima
    versione raccoglieva tutte le maiuscole e segnalava "Definizione",
    "Evoluzione", "Sostenibilita'" — parole comuni maiuscole solo perche'
    aprivano una voce di elenco: sei falsi positivi su sette segnalazioni.

    Quindi: una maiuscola conta solo se NON apre una frase, mentre un
    acronimo tutto maiuscolo conta sempre, anche in apertura.

    Gli scioglimenti fra parentesi vengono ignorati: su una generazione vera
    "FOMO (Fear Of Missing Out)" faceva segnalare "Fear", che non e' un
    fornitore inventato ma la traduzione di una sigla presente nel testo.
    """
    # Le glosse si svuotano prima dello spoglio, non si cancellano: cosi' le
    # posizioni di inizio frase restano quelle del testo originale.
    testo = _GLOSSA_ACRONIMO.sub(lambda m: m.group().replace(m.group(1), ""), testo)
    candidati: list[str] = []
    for riga in testo.splitlines():
        riga = riga.lstrip("-*• \t")
        for frase in _FINE_FRASE.split(riga):
            primo = True
            for trovato in _TOKEN_PAROLA.finditer(frase):
                parola = trovato.group()
                if len(parola) >= 2 and parola.isupper():
                    candidati.append(parola)
                elif not primo and len(parola) >= 3 and parola[:1].isupper():
                    candidati.append(parola)
                primo = False
    return candidati


def marca_nomi_sospetti(testo: str, trascrizione: str) -> tuple[str, list[str]]:
    """Segna con [?] i nomi propri che non compaiono nella trascrizione.

    Serve contro un difetto misurato e non eliminabile col prompt: davanti a
    un passaggio di trascrizione rovinato il modello completa il nome che
    immagina. Su una telefonata vera ha coniato prima "Targasys" e poi
    "Infotarga" da un unico "targata" biascicato, come se fossero fornitori
    reali. Il marcatore non impedisce l'invenzione: la rende visibile prima
    che qualcuno agisca sul verbale.

    Restituisce il testo marcato e l'elenco dei nomi sospetti.
    """
    if not testo.strip() or not trascrizione.strip():
        return testo, []

    fonte = trascrizione.lower()
    sospetti = [
        parola
        for parola in dict.fromkeys(_candidati_nomi(testo))
        if parola.lower() not in fonte
    ]

    marcato = testo
    for parola in sospetti:
        # Il lookahead evita di marcare due volte la stessa occorrenza.
        marcato = re.sub(
            rf"\b{re.escape(parola)}\b(?!\s*\[\?\])",
            f"{parola} [?]",
            marcato,
        )
    return marcato, sospetti


def pulisci_elenco(testo: str) -> str:
    """Toglie voci vuote, commenti sull'assenza di contenuto e sezioni rimaste
    senza voci.

    Serve perche' i modelli piccoli riempiono comunque lo schema di risposta:
    producono l'intestazione seguita da un trattino nudo, oppure una voce che
    commenta il fatto di non aver trovato niente. Chiederglielo nel prompt non
    basta, va ripulito qui.
    """
    sezioni: list[tuple[str, list[str]]] = []
    intestazione = ""
    voci: list[str] = []

    def chiudi() -> None:
        nonlocal intestazione, voci
        if voci:
            sezioni.append((intestazione, voci))
        intestazione, voci = "", []

    for riga in testo.splitlines():
        # Il grassetto Markdown non serve a niente qui e sporca il confronto
        # con i nomi delle intestazioni.
        spoglia = riga.replace("**", "").replace("__", "").strip()
        if not spoglia:
            continue

        if spoglia.startswith(("-", "*", "•")):
            contenuto = spoglia.lstrip("-*• ").strip()

            # Un'intestazione travestita da voce resta un'intestazione.
            if contenuto.rstrip(":").strip().lower() in _INTESTAZIONI:
                chiudi()
                intestazione = contenuto.rstrip(":").strip()
                continue

            # Tolti i segnaposto del formato (<chi>, <entro quando>...) la voce
            # puo' restare vuota: e' il modello che ha ricopiato lo schema
            # invece di riempirlo.
            nuda = re.sub(r"<[^>]*>", "", contenuto)
            if (
                _VOCE_VUOTA.match(nuda)
                or _VOCE_COMMENTO.search(contenuto)
                or _ECO_MODELLO.match(contenuto)
            ):
                continue
            voci.append("- " + contenuto)
        else:
            chiudi()
            intestazione = spoglia.rstrip(":").strip()
    chiudi()

    blocchi = [
        "\n".join([titolo, *elenco]) if titolo else "\n".join(elenco)
        for titolo, elenco in sezioni
    ]
    return "\n\n".join(blocchi).strip()


@dataclass
class Highlights:
    """Il testo estratto piu' l'indicazione di chi lo ha prodotto."""

    text: str
    engine: str  # "local" | "cloud"
    detail: str  # descrizione leggibile, finisce nella barra di stato
    # Nomi propri comparsi nell'uscita ma assenti dalla trascrizione: nel testo
    # sono marcati con [?]. Vedi marca_nomi_sospetti().
    sospetti: list[str] = field(default_factory=list)

    @property
    def is_cloud(self) -> bool:
        return self.engine == "cloud"

    def avviso_sospetti(self) -> str:
        """Riga da mettere accanto al verbale quando ci sono nomi da controllare."""
        if not self.sospetti:
            return ""
        elenco = ", ".join(self.sospetti)
        quanti = (
            "Un nome non compare" if len(self.sospetti) == 1
            else f"{len(self.sospetti)} nomi non compaiono"
        )
        return (
            f"[DA CONTROLLARE] {quanti} nella trascrizione e potrebbe essere "
            f"inventato: {elenco}. Nel testo e' marcato con [?]."
        )

    def banner(self) -> str:
        """Riga di intestazione: deve dire a colpo d'occhio dove e' finito il testo."""
        if self.is_cloud:
            return f"[CLOUD] Elaborato da {self.detail} - la trascrizione e' stata inviata a un servizio esterno"
        return f"[LOCALE] Elaborato da {self.detail} - nessun dato ha lasciato questo PC"


class RemoteSummarizer:
    """Base comune ai motori remoti.

    Esiste per i controlli di tipo: il codice e i test devono poter chiedere
    "questo motore manda il testo fuori dal PC?" senza elencare i fornitori.
    """


def _testo_gemini(interazione) -> str:
    """Estrae il testo dalla risposta di Gemini.

    output_text e' la via diretta, ma per contratto restituisce solo gli ultimi
    blocchi di testo e salta quelli separati da contenuti non testuali (pensieri,
    chiamate a strumenti). Se torna vuoto si ricompone a mano dai passi.
    """
    diretto = getattr(interazione, "output_text", None)
    if diretto:
        return diretto
    pezzi: list[str] = []
    for passo in getattr(interazione, "steps", None) or []:
        for blocco in getattr(passo, "content", None) or []:
            testo = getattr(blocco, "text", None)
            if testo:
                pezzi.append(testo)
    return "".join(pezzi)


class GeminiSummarizer(RemoteSummarizer):
    """Manda il verbale all'API Gemini di Google."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg.get("model", "gemini-3.5-flash-lite")
        self._client = None

    def _key_env(self) -> str:
        return self.cfg.get("api_key_env", "GEMINI_API_KEY")

    def available(self) -> bool:
        return bool(os.environ.get(self._key_env(), "").strip())

    def _client_ready(self):
        if self._client is None:
            from google import genai

            api_key = os.environ.get(self._key_env(), "").strip()
            if not api_key:
                raise RuntimeError(
                    f"Variabile d'ambiente {self._key_env()} non impostata."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _etichetta(self) -> str:
        """Sul piano gratuito l'avviso fa parte dell'identita' del motore."""
        if self.cfg.get("piano_gratuito", False):
            return (
                f"{self.model} (piano gratuito: Google puo' usare il testo "
                "per addestrare i suoi modelli)"
            )
        return self.model

    def _configurazione(self) -> dict:
        """Il dizionario che interactions.create si aspetta.

        Un dizionario e non un oggetto tipizzato, per tre motivi verificati
        contro l'SDK installato:

        - ne esistono DUE di GenerationConfig. Quello pubblico in
          google.genai.types viene rifiutato: l'unmarshaller vuole la sua
          classe interna in _gaos, che e' privata e non va importata. Il
          dizionario e' l'unica forma pubblica che accetta.
        - thinking_level e' una chiave di primo livello, non annidata dentro
          un thinking_config come nell'altra API.
        - temperature esiste ma e' marcata deprecata, quindi non la passiamo:
          a tenere il formato ci pensano il prompt e pulisci_elenco().

        thinking_level va in minuscolo: l'enum Python dell'SDK ha i nomi in
        maiuscolo, ma il protocollo accetta solo 'minimal', 'low', 'medium',
        'high' e un 'LOW' viene respinto con 400.
        """
        opzioni: dict = {"max_output_tokens": self.cfg.get("max_tokens", 1024)}
        livello = self.cfg.get("thinking_level")
        if livello:
            opzioni["thinking_level"] = str(livello).strip().lower()
        return opzioni

    def plan(self, transcript_text: str) -> tuple[str, bool]:
        return f"invio a Gemini ({self._etichetta()})", True

    def summarize(self, transcript_text: str) -> Highlights:
        """Chiede i punti salienti, riprovando sui guasti dichiarati temporanei.

        Il 503 "high demand" capita e passa da solo: senza un riprova, un
        verbale lungo andrebbe perso per un singolo picco di traffico.
        """
        client = self._client_ready()
        tentativi = max(1, int(self.cfg.get("tentativi", 3)))
        attesa = 4.0
        ultimo: Exception | None = None

        for numero in range(1, tentativi + 1):
            try:
                interazione = client.interactions.create(
                    model=self.model,
                    input=transcript_text,
                    system_instruction=SUMMARY_SYSTEM_PROMPT,
                    generation_config=self._configurazione(),
                )
                testo = pulisci_elenco(_testo_gemini(interazione))
                return Highlights(testo, "cloud", self._etichetta())
            except Exception as exc:
                ultimo = exc
                if numero < tentativi and self._e_temporaneo(exc):
                    time.sleep(attesa)
                    attesa *= 2
                    continue
                raise RuntimeError(self._traduci_errore(exc)) from exc

        raise RuntimeError(self._traduci_errore(ultimo or Exception("ignoto")))

    @staticmethod
    def _e_temporaneo(exc: Exception) -> bool:
        """Vale la pena riprovare? Solo per i guasti che passano da soli."""
        basso = str(exc).lower()
        return (
            "503" in basso
            or "service_unavailable" in basso
            or "high demand" in basso
            or "overloaded" in basso
            or "unavailable" in basso
        )

    @staticmethod
    def _traduci_errore(exc: Exception) -> str:
        """Il piano gratuito ha limiti di frequenza: devono leggersi, non esplodere."""
        messaggio = str(exc)
        basso = messaggio.lower()
        if "429" in messaggio or "resource_exhausted" in basso or "quota" in basso:
            return (
                "Limite del piano Gemini raggiunto. Riprova piu' tardi, oppure "
                "scegli il modello locale dalla tendina."
            )
        if GeminiSummarizer._e_temporaneo(exc):
            return (
                "Gemini e' sovraccarico in questo momento (503). Ho riprovato "
                "senza successo: ritenta fra qualche minuto, oppure scegli il "
                "modello locale dalla tendina."
            )
        if any(
            s in basso for s in ("api key", "unauthenticated", "permission_denied")
        ) or any(c in messaggio for c in ("401", "403")):
            return "Chiave Gemini rifiutata: controlla GEMINI_API_KEY."
        return f"Gemini non ha risposto: {messaggio}"


class CloudSummarizer(RemoteSummarizer):
    """Manda il verbale all'API Claude. Veloce, ma il testo esce dal PC."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg.get("model", "claude-haiku-4-5-20251001")
        self._client = None

    def available(self) -> bool:
        key_env = self.cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        return bool(os.environ.get(key_env, "").strip())

    def _client_ready(self):
        if self._client is None:
            import anthropic

            key_env = self.cfg.get("api_key_env", "ANTHROPIC_API_KEY")
            api_key = os.environ.get(key_env, "").strip()
            if not api_key:
                raise RuntimeError(f"Variabile d'ambiente {key_env} non impostata.")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def plan(self, transcript_text: str) -> tuple[str, bool]:
        return f"invio al cloud ({self.model})", True

    def summarize(self, transcript_text: str) -> Highlights:
        client = self._client_ready()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.cfg.get("max_tokens", 1024),
            system=SUMMARY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": transcript_text}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return Highlights(pulisci_elenco(text), "cloud", self.model)


class LocalSummarizer:
    """Interroga un modello servito da Ollama sulla macchina locale.

    Sopra una certa lunghezza legge a blocchi: su CPU la velocita' di lettura
    degrada con la dimensione del contesto (misurati 23 tok/s a 900 token contro
    10 tok/s a 6400), e un modello da 3B annega comunque in un contesto lungo.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg.get("model", "gemma3:4b")
        self.host = cfg.get("host", "http://127.0.0.1:11434").rstrip("/")

    def available(self) -> bool:
        """Ollama attivo E il modello configurato presente.

        Controllare solo che il server risponda non basta: dopo una
        reinstallazione di Ollama il server c'e' ma la lista dei modelli e'
        vuota. L'indicatore direbbe "pronto" e il pulsante fallirebbe poi in
        fase di generazione, che e' il momento peggiore per scoprirlo.
        """
        return self.model in self.modelli_disponibili()

    def modelli_disponibili(self) -> set[str]:
        """I modelli che Ollama ha davvero, insieme vuoto se non risponde."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as resp:
                dati = json.load(resp)
        except Exception:
            return set()
        nomi: set[str] = set()
        for voce in dati.get("models") or []:
            nome = voce.get("name") or voce.get("model") or ""
            if not nome:
                continue
            nomi.add(nome)
            # Ollama elenca "gemma3:4b" e "nomic-embed-text:latest": una
            # configurazione senza tag deve combaciare con ":latest".
            if nome.endswith(":latest"):
                nomi.add(nome[: -len(":latest")])
        return nomi

    def _ask(self, system: str, prompt: str, num_ctx: int) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": num_ctx},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                return json.load(resp)["response"].strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama non raggiungibile su {self.host} ({exc.reason}). "
                "Avvialo con 'ollama serve'."
            ) from exc

    def _split(self, text: str) -> list[str]:
        """Taglia sulle righe, cosi' un intervento non finisce mai a meta'.

        Una riga piu' lunga del limite viene comunque spezzata sulle parole:
        senza questo ripiego una trascrizione priva di a capo resterebbe un
        blocco unico e sfonderebbe num_ctx senza dare errore.
        """
        limite = self.cfg.get("parole_per_blocco", 550)
        blocchi: list[str] = []
        corrente: list[str] = []
        conta = 0

        def chiudi() -> None:
            nonlocal corrente, conta
            if corrente:
                blocchi.append("\n".join(corrente))
                corrente, conta = [], 0

        for riga in text.splitlines():
            parole = riga.split()
            if len(parole) > limite:
                chiudi()
                for i in range(0, len(parole), limite):
                    blocchi.append(" ".join(parole[i : i + limite]))
                continue
            if conta + len(parole) > limite:
                chiudi()
            corrente.append(riga)
            conta += len(parole)

        chiudi()
        return blocchi

    def plan(self, transcript_text: str) -> tuple[str, bool]:
        n = len(self._split(transcript_text))
        blocchi = "" if n <= 1 else f", {n} blocchi"
        return f"elaborazione locale ({self.model}{blocchi})", False

    def summarize(self, transcript_text: str) -> Highlights:
        num_ctx = self.cfg.get("num_ctx", 4096)
        blocchi = self._split(transcript_text)

        if len(blocchi) <= 1:
            text = self._ask(SUMMARY_SYSTEM_PROMPT, transcript_text, num_ctx)
            return Highlights(pulisci_elenco(text), "local", self.model)

        appunti = [
            self._ask(CHUNK_EXTRACT_PROMPT, blocco, num_ctx) for blocco in blocchi
        ]
        text = self._ask(CHUNK_MERGE_PROMPT, "\n".join(appunti), num_ctx)
        return Highlights(
            pulisci_elenco(text), "local", f"{self.model} su {len(blocchi)} blocchi"
        )


class AutoSummarizer:
    """Sceglie il motore in base alla lunghezza, con ripiego se uno manca.

    Il locale e' preferito finche' l'attesa resta ragionevole: su questo tipo di
    macchina un 3B macina all'incirca 550 parole al minuto e mezzo, quindi oltre
    qualche centinaio di parole conviene il cloud.
    """

    def __init__(self, cfg: dict):
        self.soglia = cfg.get("soglia_parole", 700)
        self.local = LocalSummarizer(cfg["local"])
        self.cloud = build_cloud_summarizer(cfg["cloud"])

    def _scegli(self, transcript_text: str):
        """Restituisce (motore, e_un_ripiego)."""
        lungo = len(transcript_text.split()) > self.soglia
        preferito, ripiego = (
            (self.cloud, self.local) if lungo else (self.local, self.cloud)
        )
        if preferito.available():
            return preferito, False
        if ripiego.available():
            return ripiego, True
        raise RuntimeError(
            f"Nessun motore disponibile. Il locale richiede Ollama attivo con "
            f"il modello {self.local.model} (scaricalo con "
            f"'ollama pull {self.local.model}'); il remoto richiede la chiave "
            "API nella variabile d'ambiente."
        )

    def plan(self, transcript_text: str) -> tuple[str, bool]:
        motore, ripiego = self._scegli(transcript_text)
        etichetta, e_cloud = motore.plan(transcript_text)
        if ripiego:
            etichetta += " - ripiego, il motore preferito non risponde"
        return etichetta, e_cloud

    def summarize(self, transcript_text: str) -> Highlights:
        motore, ripiego = self._scegli(transcript_text)
        risultato = motore.summarize(transcript_text)
        if ripiego:
            risultato.detail += " (ripiego: il motore preferito non e' disponibile)"
        return risultato


# Voci del menu a tendina: (etichetta mostrata, valore in config.json).
# Restano tre: "remoto" e' remoto a prescindere dal fornitore, che si scegli con
# summary.cloud.provider in config.json.
MOTORI_SINTESI = [
    ("Automatico (locale se corto, remoto se lungo)", "auto"),
    ("Solo modello locale (niente esce dal PC)", "local"),
    ("Solo modello remoto (servizio esterno)", "cloud"),
]

FORNITORI_REMOTI = {
    "gemini": GeminiSummarizer,
    "anthropic": CloudSummarizer,
}


def build_cloud_summarizer(cloud_cfg: dict) -> RemoteSummarizer:
    """Costruisce il motore remoto indicato da summary.cloud.provider."""
    provider = cloud_cfg.get("provider", "gemini")
    classe = FORNITORI_REMOTI.get(provider)
    if classe is None:
        ammessi = ", ".join(sorted(FORNITORI_REMOTI))
        raise RuntimeError(
            f"Fornitore remoto sconosciuto in config.json: {provider!r}. "
            f"Ammessi: {ammessi}."
        )
    if provider not in cloud_cfg:
        raise RuntimeError(
            f"Manca la sezione summary.cloud.{provider} in config.json."
        )
    return classe(cloud_cfg[provider])


def build_summarizer(cfg: dict):
    summary = cfg["summary"]
    engine = summary.get("engine", "auto")
    if engine == "local":
        return LocalSummarizer(summary["local"])
    if engine == "cloud":
        return build_cloud_summarizer(summary["cloud"])
    return AutoSummarizer(summary)


# --------------------------------------------------------------------------- #
# Utilita'
# --------------------------------------------------------------------------- #

def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Ricampionamento lineare: sufficiente per il parlato in un prototipo."""
    if src_sr == dst_sr:
        return audio
    n_out = int(round(audio.shape[0] * dst_sr / src_sr))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    x_in = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
    x_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_out, x_in, audio).astype(np.float32)


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def dbfs(rms: float) -> float:
    return 20.0 * np.log10(max(rms, 1e-9))


# --------------------------------------------------------------------------- #
# Sessione di registrazione: audio, segmentazione, trascrizione
# --------------------------------------------------------------------------- #

STOP = object()
FLUSH = object()


class RecordingSession:
    """
    Tre thread, ruoli separati:
      - callback audio  -> mette i blocchi in coda, non fa nient'altro
      - collector       -> scrive il WAV, segmenta, produce i chunk
      - worker          -> trascrive un chunk per volta, in ordine
    I risultati finiscono in results, che la GUI svuota dal thread principale.
    """

    def __init__(self, cfg: dict, transcriber: Transcriber, session_dir: Path):
        self.cfg = cfg
        self.transcriber = transcriber
        self.dir = session_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.dir / "verbale.md"
        self.audio_path = self.dir / "audio.wav"

        seg = cfg["segmentation"]
        self.segmenter = Segmenter(
            samplerate=TARGET_SR,
            min_chunk_sec=seg["min_chunk_sec"],
            max_chunk_sec=seg["max_chunk_sec"],
            silence_hold_sec=seg["silence_hold_sec"],
            min_speech_sec=seg["min_speech_sec"],
            noise_multiplier=seg["noise_multiplier"],
            absolute_floor=seg["absolute_floor"],
        )

        self.audio_q: queue.Queue = queue.Queue()
        self.chunk_q: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()

        self.paused = threading.Event()
        self._stream = None
        self._wav = None
        self._collector = None
        self._worker = None
        self._device_sr = TARGET_SR
        self.level_rms = 0.0
        self.pending = 0
        self._pending_lock = threading.Lock()

    # -- ciclo di vita ----------------------------------------------------- #

    def start(self, device_index: int | None) -> None:
        if sd is None:
            raise RuntimeError("sounddevice non disponibile: pip install sounddevice")

        blocksize = self.cfg["audio"]["blocksize"]
        try:
            self._stream = sd.InputStream(
                samplerate=TARGET_SR, channels=1, dtype="float32",
                blocksize=blocksize, device=device_index, callback=self._on_audio,
            )
            self._device_sr = TARGET_SR
        except Exception:
            # Alcuni dispositivi Windows non accettano 16 kHz: apriamo alla
            # frequenza nativa e ricampioniamo nel collector.
            info = sd.query_devices(device_index, "input")
            self._device_sr = int(info["default_samplerate"])
            self._stream = sd.InputStream(
                samplerate=self._device_sr, channels=1, dtype="float32",
                blocksize=blocksize, device=device_index, callback=self._on_audio,
            )

        self._wav = wave.open(str(self.audio_path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(TARGET_SR)

        self._collector = threading.Thread(target=self._collect, daemon=True)
        self._worker = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._collector.start()
        self._worker.start()
        self._stream.start()

    def pause(self) -> None:
        self.paused.set()
        self.audio_q.put(FLUSH)

    def resume(self) -> None:
        self.paused.clear()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.audio_q.put(STOP)

    # -- thread ------------------------------------------------------------ #

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if self.paused.is_set():
            self.level_rms = 0.0
            return
        block = indata[:, 0].copy()
        self.level_rms = float(np.sqrt(np.mean(np.square(block))))
        self.audio_q.put(block)

    def _collect(self) -> None:
        while True:
            item = self.audio_q.get()

            if item is STOP:
                for chunk in self.segmenter.flush():
                    self._enqueue_chunk(chunk)
                self.chunk_q.put(STOP)
                if self._wav is not None:
                    self._wav.close()
                    self._wav = None
                return

            if item is FLUSH:
                for chunk in self.segmenter.flush():
                    self._enqueue_chunk(chunk)
                continue

            block = resample_linear(item, self._device_sr, TARGET_SR)
            if self._wav is not None:
                self._wav.writeframes(float_to_pcm16(block).tobytes())
            for chunk in self.segmenter.feed(block):
                self._enqueue_chunk(chunk)

    def _enqueue_chunk(self, chunk: Chunk) -> None:
        with self._pending_lock:
            self.pending += 1
        self.chunk_q.put(chunk)

    def _transcribe_loop(self) -> None:
        while True:
            item = self.chunk_q.get()
            if item is STOP:
                self.results.put(("done", None))
                return
            try:
                text = self.transcriber.transcribe(item.audio, TARGET_SR)
                if text:
                    self.results.put(("text", (item.start_sec, text)))
            except Exception as exc:
                self.results.put(("error", f"Trascrizione fallita: {exc}"))
            finally:
                with self._pending_lock:
                    self.pending -= 1


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #

import tkinter as tk  # noqa: E402
from tkinter import messagebox, ttk  # noqa: E402

# Palette azzurra. Tenuta in un posto solo: i widget non hanno colori scritti
# a mano, cosi' cambiare tinta e' una modifica sola.
COLORI = {
    "sfondo": "#EAF3FA",       # fondo della finestra
    "superficie": "#FFFFFF",   # aree di contenuto (testo, campi)
    "bordo": "#B7D5EA",
    "blu": "#1F6FB2",          # accento principale
    "blu_vivo": "#2A85CE",     # accento sotto il mouse
    "blu_scuro": "#0E3F63",    # accento premuto, fondo del vumetro
    "blu_chiaro": "#D3E7F5",   # selezione
    "testo": "#16394F",
    "testo_tenue": "#5B7B90",
    "spento": "#C9DCEA",       # controlli disabilitati
    "spento_testo": "#8FAEC4",
    "ok": "#1B7A4B",
    "allarme": "#B03A2E",
}


class App(tk.Tk):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.session: RecordingSession | None = None
        self.transcriber: Transcriber | None = None
        self.summarizer: LocalSummarizer | CloudSummarizer | AutoSummarizer | None = None
        self.model_ready = False
        self.generating_highlights = False
        self._hl_inizio: float | None = None  # avvio dell'elaborazione, per il cronometro
        self._hl_etichetta = ""
        self.state_name = "idle"  # idle | recording | paused | finishing
        self.peak = 0.0

        self.title("Verbale — registrazione e trascrizione")
        self.geometry("980x680")
        self.minsize(760, 520)
        self._build_ui()
        self._load_model_async()
        self._tick_meter()
        self._tick_results()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- costruzione interfaccia ------------------------------------------ #

    def _applica_tema(self) -> None:
        """Palette azzurra su tutti i widget ttk.

        Serve il tema 'clam': quello nativo di Windows ignora quasi tutte le
        opzioni di colore, quindi senza questa riga il resto non ha effetto.
        """
        c = COLORI
        self.configure(bg=c["sfondo"])
        stile = ttk.Style(self)
        stile.theme_use("clam")

        stile.configure(
            ".",
            background=c["sfondo"],
            foreground=c["testo"],
            fieldbackground=c["superficie"],
            bordercolor=c["bordo"],
            font=("Segoe UI", 10),
        )
        stile.configure("TFrame", background=c["sfondo"])
        stile.configure("TLabel", background=c["sfondo"], foreground=c["testo"])
        stile.configure(
            "TButton",
            background=c["blu"],
            foreground=c["superficie"],
            bordercolor=c["blu"],
            focuscolor=c["blu"],
            relief="flat",
            padding=(12, 6),
        )
        stile.map(
            "TButton",
            background=[
                ("disabled", c["spento"]),
                ("pressed", c["blu_scuro"]),
                ("active", c["blu_vivo"]),
            ],
            foreground=[("disabled", c["spento_testo"])],
            bordercolor=[("disabled", c["spento"])],
        )
        stile.configure(
            "TCombobox",
            fieldbackground=c["superficie"],
            background=c["superficie"],
            foreground=c["testo"],
            arrowcolor=c["blu"],
            bordercolor=c["bordo"],
            selectbackground=c["blu_chiaro"],
            selectforeground=c["testo"],
            padding=4,
        )
        stile.map(
            "TCombobox",
            fieldbackground=[("readonly", c["superficie"])],
            bordercolor=[("focus", c["blu"])],
        )
        stile.configure(
            "TEntry",
            fieldbackground=c["superficie"],
            foreground=c["testo"],
            bordercolor=c["bordo"],
            insertcolor=c["blu"],
            padding=4,
        )
        stile.map("TEntry", bordercolor=[("focus", c["blu"])])
        stile.configure(
            "TScrollbar",
            background=c["blu_chiaro"],
            troughcolor=c["sfondo"],
            bordercolor=c["sfondo"],
            arrowcolor=c["blu"],
        )

        # La tendina aperta della Combobox e' una Listbox Tk, non un widget ttk:
        # si colora solo per opzioni.
        self.option_add("*TCombobox*Listbox.background", c["superficie"])
        self.option_add("*TCombobox*Listbox.foreground", c["testo"])
        self.option_add("*TCombobox*Listbox.selectBackground", c["blu"])
        self.option_add("*TCombobox*Listbox.selectForeground", c["superficie"])

    def _build_ui(self) -> None:
        self._applica_tema()
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Microfono").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(
            top, textvariable=self.device_var, state="readonly", width=46
        )
        self.device_box.pack(side="left", padx=(6, 16))
        self._populate_devices()

        ttk.Button(top, text="Aggiorna elenco", command=self._populate_devices).pack(
            side="left"
        )
        self.engine_label = ttk.Label(top, text="", foreground=COLORI["testo_tenue"])
        self.engine_label.pack(side="right")

        meter_row = ttk.Frame(self)
        meter_row.pack(fill="x", **pad)
        ttk.Label(meter_row, text="Livello").pack(side="left")
        self.meter = tk.Canvas(
            meter_row, height=20, bg=COLORI["blu_scuro"], highlightthickness=0
        )
        self.meter.pack(side="left", fill="x", expand=True, padx=8)
        self.clock_label = ttk.Label(meter_row, text="00:00:00", font=("Consolas", 11))
        self.clock_label.pack(side="right")

        # Le righe sotto vengono impacchettate PRIMA del riquadro di testo (e
        # ancorate con side="bottom") cosi' da avere sempre la loro altezza
        # naturale garantita: e' il testo a restringersi/scrollare su finestre
        # basse, mai i pulsanti a sparire.
        self.status = ttk.Label(self, text="Caricamento del modello…", anchor="w")
        self.status.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        # Scelta del motore per i punti salienti. Impacchettata dopo la barra di
        # stato e prima dei pulsanti, cosi' finisce fra i due.
        engine_row = ttk.Frame(self)
        engine_row.pack(side="bottom", fill="x", **pad)
        ttk.Label(engine_row, text="Punti salienti con").pack(side="left")
        self.summary_engine_var = tk.StringVar()
        self.summary_engine_box = ttk.Combobox(
            engine_row,
            textvariable=self.summary_engine_var,
            state="readonly",
            width=44,
            values=[etichetta for etichetta, _ in MOTORI_SINTESI],
        )
        self.summary_engine_box.pack(side="left", padx=(6, 10))
        self.summary_engine_box.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_summary_engine_changed()
        )
        self.summary_engine_status = ttk.Label(engine_row, text="", foreground=COLORI["testo_tenue"])
        self.summary_engine_status.pack(side="left")

        scelto = self.cfg["summary"].get("engine", "auto")
        self.summary_engine_var.set(
            next(
                (e for e, v in MOTORI_SINTESI if v == scelto), MOTORI_SINTESI[0][0]
            )
        )
        self._check_summary_engine()

        buttons = ttk.Frame(self)
        buttons.pack(side="bottom", fill="x", **pad)
        self.btn_start = ttk.Button(buttons, text="Avvia registrazione", command=self.start)
        self.btn_pause = ttk.Button(buttons, text="Pausa", command=self.toggle_pause)
        self.btn_stop = ttk.Button(buttons, text="Termina", command=self.stop)
        self.btn_copy = ttk.Button(buttons, text="Copia tutto", command=self.copy_all)
        self.btn_open = ttk.Button(buttons, text="Apri con Notepad", command=self.open_notepad)
        self.btn_highlights = ttk.Button(
            buttons, text="Punti salienti", command=self.generate_highlights
        )
        for b in (
            self.btn_start, self.btn_pause, self.btn_stop,
            self.btn_copy, self.btn_open, self.btn_highlights,
        ):
            b.pack(side="left", padx=(0, 8))

        note_row = ttk.Frame(self)
        note_row.pack(side="bottom", fill="x", **pad)
        self.note_entry = ttk.Entry(note_row, font=("Segoe UI", 10))
        self.note_entry.pack(side="left", fill="x", expand=True)
        self.note_entry.bind("<Return>", lambda _e: self.add_note())
        self.btn_note = ttk.Button(note_row, text="Aggiungi nota", command=self.add_note)
        self.btn_note.pack(side="left", padx=(8, 0))

        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, **pad)
        self.text = tk.Text(
            text_frame, wrap="word", font=("Segoe UI", 11), state="disabled",
            spacing1=2, spacing3=4, padx=10, pady=8,
            bg=COLORI["superficie"], fg=COLORI["testo"],
            insertbackground=COLORI["blu"],
            selectbackground=COLORI["blu_chiaro"],
            selectforeground=COLORI["blu_scuro"],
            relief="flat", highlightthickness=1,
            highlightbackground=COLORI["bordo"], highlightcolor=COLORI["bordo"],
        )
        scroll = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure(
            "ts", foreground=COLORI["testo_tenue"], font=("Consolas", 9)
        )
        self.text.tag_configure(
            "note", foreground=COLORI["blu"], font=("Segoe UI", 11, "bold")
        )
        self.text.tag_configure(
            "meta", foreground=COLORI["testo_tenue"], font=("Segoe UI", 9, "italic")
        )
        # Rosso su giallo: quando il testo esce dal PC deve saltare all'occhio.
        self.text.tag_configure(
            "cloud",
            foreground="#8a2b06",
            background="#fff3cd",
            font=("Segoe UI", 9, "bold"),
        )

        self._refresh_buttons()

    def _populate_devices(self) -> None:
        self.devices: list[tuple[int, str]] = []
        if sd is not None:
            try:
                for idx, dev in enumerate(sd.query_devices()):
                    if dev.get("max_input_channels", 0) > 0:
                        self.devices.append((idx, f"[{idx}] {dev['name']}"))
            except Exception:
                pass
        labels = [label for _idx, label in self.devices] or ["Nessun ingresso rilevato"]
        self.device_box["values"] = labels
        preferred = self.cfg["audio"].get("device")
        chosen = 0
        for pos, (idx, _label) in enumerate(self.devices):
            if idx == preferred:
                chosen = pos
                break
        self.device_box.current(chosen)

    def _selected_device(self) -> int | None:
        pos = self.device_box.current()
        if 0 <= pos < len(self.devices):
            return self.devices[pos][0]
        return None

    # -- caricamento modello ---------------------------------------------- #

    def _load_model_async(self) -> None:
        def work():
            try:
                tr = build_transcriber(self.cfg)
                tr.load()
                self.after(0, lambda: self._on_model_ready(tr))
            except Exception as exc:
                self.after(0, lambda: self._on_model_failed(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_model_ready(self, tr: Transcriber) -> None:
        self.transcriber = tr
        self.model_ready = True
        self.engine_label.config(text=tr.name)
        self.status.config(text="Pronto. Premi Avvia registrazione.")
        self._refresh_buttons()

    def _on_model_failed(self, exc: Exception) -> None:
        self.status.config(text=f"Motore non disponibile: {exc}")
        messagebox.showerror(
            "Motore non disponibile",
            f"{exc}\n\nControlla config.json e le dipendenze installate.",
        )

    # -- azioni ------------------------------------------------------------ #

    def start(self) -> None:
        if not self.model_ready or self.transcriber is None:
            return
        stamp = datetime.now()
        session_dir = (
            APP_DIR / self.cfg["output_dir"] / stamp.strftime("%Y-%m-%d_%H-%M-%S")
        )
        try:
            self.session = RecordingSession(self.cfg, self.transcriber, session_dir)
            self.session.start(self._selected_device())
        except Exception as exc:
            self.session = None
            messagebox.showerror("Avvio non riuscito", str(exc))
            self.status.config(text=f"Avvio non riuscito: {exc}")
            return

        header = (
            f"# Verbale — {stamp.strftime('%d/%m/%Y %H:%M')}\n\n"
            f"- Sorgente: {self.device_var.get()}\n"
            f"- Motore: {self.transcriber.name}, lingua {self.cfg['language']}\n\n"
            "## Trascrizione\n\n"
        )
        self.session.transcript_path.write_text(header, encoding="utf-8")

        self._clear_text()
        self._append(f"Sessione avviata — {session_dir.name}\n\n", "meta")
        self.state_name = "recording"
        self.status.config(text=f"Registrazione in corso → {self.session.transcript_path}")
        self._refresh_buttons()

    def toggle_pause(self) -> None:
        if self.session is None:
            return
        if self.state_name == "recording":
            self.session.pause()
            self.state_name = "paused"
            self._write_and_append(f"\n_— pausa {self._clock()} —_\n\n", "meta")
            self.status.config(text="In pausa. L'audio non viene catturato.")
        elif self.state_name == "paused":
            self.session.resume()
            self.state_name = "recording"
            self._write_and_append(f"_— ripresa {self._clock()} —_\n\n", "meta")
            self.status.config(text="Registrazione in corso.")
        self._refresh_buttons()

    def stop(self) -> None:
        if self.session is None:
            return
        self.session.stop()
        self.state_name = "finishing"
        self.status.config(text="Registrazione conclusa. Trascrivo l'ultima parte…")
        self._refresh_buttons()

    def add_note(self) -> None:
        note = self.note_entry.get().strip()
        if not note:
            return
        self._write_and_append(f"\n> **NOTA {self._clock()}** — {note}\n\n", "note")
        self.note_entry.delete(0, "end")

    def copy_all(self) -> None:
        content = self._current_transcript_text()
        if not content:
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self.update()  # necessario perche' Windows riceva davvero il contenuto
        self.status.config(text=f"Copiati {len(content)} caratteri negli appunti.")

    def open_notepad(self) -> None:
        if self.session is None:
            self.status.config(text="Nessun verbale da aprire: avvia una sessione.")
            return
        path = str(self.session.transcript_path)
        try:
            subprocess.Popen(["notepad.exe", path])
        except Exception:
            try:
                os.startfile(path)  # type: ignore[attr-defined]
            except Exception as exc:
                messagebox.showerror("Apertura non riuscita", str(exc))

    # -- scelta del motore per i punti salienti ---------------------------- #

    def _on_summary_engine_changed(self) -> None:
        etichetta = self.summary_engine_var.get()
        valore = next((v for e, v in MOTORI_SINTESI if e == etichetta), "auto")
        self.cfg["summary"]["engine"] = valore
        # Il sommarizzatore e' costruito una volta sola e messo in cache: va
        # buttato, altrimenti la scelta nuova non ha effetto.
        self.summarizer = None
        self._salva_preferenza_motore(valore)
        self._check_summary_engine()

    def _salva_preferenza_motore(self, valore: str) -> None:
        """Scrive la scelta in config.json, cosi' sopravvive al riavvio."""
        try:
            dati = (
                json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if CONFIG_PATH.exists()
                else {}
            )
            if not isinstance(dati.get("summary"), dict):
                dati["summary"] = {}
            dati["summary"]["engine"] = valore
            CONFIG_PATH.write_text(
                json.dumps(dati, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            # Non e' un errore fatale: la scelta vale per questa sessione.
            self.summary_engine_status.config(
                text=f"scelta non salvata su disco ({exc})",
                foreground=COLORI["allarme"],
            )

    def _check_summary_engine(self) -> None:
        """Dice subito se il motore scelto e' raggiungibile.

        Sondare Ollama costa un paio di secondi, quindi gira in un thread:
        scoprire che manca la chiave o che Ollama e' spento deve succedere
        adesso, non dopo aver premuto il pulsante su un verbale lungo.
        """
        self.summary_engine_status.config(text="verifico…", foreground=COLORI["testo_tenue"])
        scelto = self.cfg["summary"].get("engine", "auto")
        summary = dict(self.cfg["summary"])

        def work():
            motore_locale = LocalSummarizer(summary["local"])
            modelli = motore_locale.modelli_disponibili()
            locale = motore_locale.model in modelli
            # Distinguere i due guasti del locale: Ollama spento, oppure attivo
            # ma senza il modello configurato. Le soluzioni sono diverse.
            if locale:
                guasto_locale = ""
            elif modelli:
                guasto_locale = f"manca il modello {motore_locale.model}"
            else:
                guasto_locale = "Ollama non risponde"

            try:
                motore_remoto = build_cloud_summarizer(summary["cloud"])
                cloud = motore_remoto.available()
                guasto_remoto = "" if cloud else "chiave API non impostata"
            except RuntimeError as exc:
                cloud, guasto_remoto = False, str(exc)

            if scelto == "local":
                esito = ("pronto", True) if locale else (guasto_locale, False)
            elif scelto == "cloud":
                esito = ("pronto", True) if cloud else (guasto_remoto, False)
            elif locale and cloud:
                esito = ("entrambi disponibili", True)
            elif locale:
                esito = (f"solo locale: {guasto_remoto}", False)
            elif cloud:
                esito = (f"solo remoto: {guasto_locale}", False)
            else:
                esito = (f"nessun motore: {guasto_locale}, {guasto_remoto}", False)
            self.after(0, lambda: self._mostra_stato_motore(*esito))

        threading.Thread(target=work, daemon=True).start()

    def _mostra_stato_motore(self, testo: str, ok: bool) -> None:
        self.summary_engine_status.config(
            text=testo, foreground=COLORI["ok"] if ok else COLORI["allarme"]
        )

    def generate_highlights(self) -> None:
        if self.generating_highlights:
            return
        content = self._current_transcript_text()
        if not content.strip():
            self.status.config(text="Nessun testo da analizzare.")
            return

        # Il riscontro va dove l'utente sta guardando. Su un verbale lungo col
        # motore remoto l'attesa e' di minuti: con il solo testo nella barra di
        # stato in fondo alla finestra sembra che il clic non abbia fatto nulla.
        self.generating_highlights = True
        self._hl_inizio = time.monotonic()
        self._hl_etichetta = "Scelgo il motore"
        self._refresh_buttons()
        self._append("\nGenero i punti salienti…\n", "meta")
        self._tick_highlights()

        def work():
            try:
                if self.summarizer is None:
                    self.summarizer = build_summarizer(self.cfg)
                # La scelta del motore va annunciata prima di partire: se il
                # testo sta per uscire dal PC l'utente deve saperlo adesso, non
                # a cose fatte.
                etichetta, e_cloud = self.summarizer.plan(content)
                self.after(0, lambda: self._on_highlights_planned(etichetta, e_cloud))
                risultato = self.summarizer.summarize(content)
                # Il confronto coi nomi della trascrizione vale per qualsiasi
                # motore, quindi sta qui e non dentro i singoli motori.
                risultato.text, risultato.sospetti = marca_nomi_sospetti(
                    risultato.text, content
                )
                self.after(0, lambda: self._on_highlights_ready(risultato))
            except Exception as exc:
                self.after(0, lambda: self._on_highlights_failed(exc))

        threading.Thread(target=work, daemon=True).start()

    def _tick_highlights(self) -> None:
        """Cronometro che scorre finche' il motore sta elaborando.

        Serve a distinguere "sta lavorando" da "non e' partito niente": senza
        un numero che cambia, tre minuti di attesa sono indistinguibili da un
        clic andato perso.
        """
        if not self.generating_highlights or self._hl_inizio is None:
            return
        trascorso = time.monotonic() - self._hl_inizio
        puntini = "." * (1 + int(trascorso * 2) % 3)
        self.status.config(
            text=f"{self._hl_etichetta}{puntini}  {fmt_hms(trascorso)}  "
            "(il pulsante torna attivo a elaborazione finita)"
        )
        self.after(400, self._tick_highlights)

    def _on_highlights_planned(self, etichetta: str, e_cloud: bool) -> None:
        prefisso = "CLOUD - " if e_cloud else ""
        self._hl_etichetta = f"{prefisso}{etichetta}"
        # Anche nella finestra, non solo nella barra: se il testo sta per
        # uscire dal PC deve vedersi dove l'utente sta leggendo.
        self._append(f"{etichetta}\n", "cloud" if e_cloud else "meta")

    def _durata_highlights(self) -> str:
        """Quanto e' durata l'elaborazione, da aggiungere al messaggio finale."""
        if self._hl_inizio is None:
            return ""
        durata = fmt_hms(time.monotonic() - self._hl_inizio)
        self._hl_inizio = None
        return f" in {durata}"

    def _on_highlights_ready(self, risultato: Highlights) -> None:
        self.generating_highlights = False
        durata = self._durata_highlights()
        banner = risultato.banner()

        # Dopo la ripulitura puo' non restare niente: e' il caso di una
        # trascrizione troppo breve o senza contenuto decisionale. Meglio dirlo
        # che mostrare intestazioni vuote.
        if not risultato.text.strip():
            self._append("\nPunti salienti\n", "note")
            self._append(f"{banner}\n", "cloud" if risultato.is_cloud else "note")
            self._append("Nessun punto saliente individuato nel testo.\n\n", "meta")
            self.status.config(
                text=f"Nessun punto saliente individuato{durata}."
            )
            self._refresh_buttons()
            return

        avviso = risultato.avviso_sospetti()
        if self.session is not None:
            with open(self.session.transcript_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n## Punti salienti\n\n_{banner}_\n\n")
                if avviso:
                    fh.write(f"_{avviso}_\n\n")
                fh.write(f"{risultato.text}\n\n")
                fh.flush()
        self._append("\nPunti salienti\n", "note")
        self._append(f"{banner}\n", "cloud" if risultato.is_cloud else "note")
        if avviso:
            self._append(f"{avviso}\n", "cloud")
        self._append(f"{risultato.text}\n\n")
        quanti = len(risultato.sospetti)
        coda = f" {quanti} nome da controllare." if quanti == 1 else (
            f" {quanti} nomi da controllare." if quanti else ""
        )
        self.status.config(
            text=("Punti salienti generati nel CLOUD" if risultato.is_cloud
                  else "Punti salienti generati in locale") + durata + "." + coda
        )
        self._refresh_buttons()

    def _on_highlights_failed(self, exc: Exception) -> None:
        self.generating_highlights = False
        durata = self._durata_highlights()
        self.status.config(text=f"Punti salienti non generati{durata}: {exc}")
        # Anche nella finestra: chi guarda il testo deve capire che il "Genero…"
        # di prima e' finito male, non che sta ancora lavorando.
        self._append(f"Punti salienti non generati: {exc}\n\n", "cloud")
        messagebox.showerror("Punti salienti non riusciti", str(exc))
        self._refresh_buttons()

    # -- polling ----------------------------------------------------------- #

    def _tick_meter(self) -> None:
        rms = self.session.level_rms if self.session else 0.0
        level = max(0.0, min(1.0, (dbfs(rms) + 60.0) / 60.0))
        self.peak = max(level, self.peak * 0.92)

        c = self.meter
        c.delete("all")
        width = max(c.winfo_width(), 1)
        height = max(c.winfo_height(), 1)
        segments = 40
        gap = 2
        seg_w = (width - gap * (segments - 1)) / segments
        lit = int(level * segments)
        for i in range(segments):
            frac = i / segments
            if frac > 0.9:
                color = "#e04b3a"
            elif frac > 0.72:
                color = "#e0b23a"
            else:
                color = "#3ac06a"
            x0 = i * (seg_w + gap)
            c.create_rectangle(
                x0, 3, x0 + seg_w, height - 3,
                fill=color if i < lit else "#1D5177", outline="",
            )
        px = self.peak * width
        c.create_line(px, 1, px, height - 1, fill=COLORI["sfondo"], width=1)

        if self.session:
            self.clock_label.config(text=self._clock())
        self.after(50, self._tick_meter)

    def _tick_results(self) -> None:
        if self.session is not None:
            while True:
                try:
                    kind, payload = self.session.results.get_nowait()
                except queue.Empty:
                    break
                if kind == "text":
                    start_sec, text = payload
                    self._write_and_append_segment(start_sec, text)
                elif kind == "error":
                    self._append(f"[{payload}]\n\n", "meta")
                elif kind == "done":
                    self._finish()
            if self.state_name in ("recording", "paused"):
                pending = self.session.pending
                base = "Registrazione in corso" if self.state_name == "recording" else "In pausa"
                suffix = f" — {pending} blocco/i in coda" if pending else ""
                self.status.config(text=f"{base}{suffix} → {self.session.transcript_path}")
        self.after(120, self._tick_results)

    def _finish(self) -> None:
        self.state_name = "idle"
        if self.session is not None:
            with open(self.session.transcript_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n_Fine sessione — durata {self._clock()}._\n")
            self.status.config(text=f"Verbale completo: {self.session.transcript_path}")
            self._append("\nTrascrizione completata.\n", "meta")
        self._refresh_buttons()

    # -- testo e file ------------------------------------------------------ #

    def _clock(self) -> str:
        if self.session is None:
            return "00:00:00"
        return fmt_hms(self.session.segmenter.elapsed_sec)

    def _clear_text(self) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")

    def _append(self, content: str, tag: str | None = None) -> None:
        self.text.config(state="normal")
        self.text.insert("end", content, tag or ())
        self.text.config(state="disabled")
        self.text.see("end")

    def _write_and_append(self, markdown: str, tag: str | None = None) -> None:
        if self.session is not None:
            with open(self.session.transcript_path, "a", encoding="utf-8") as fh:
                fh.write(markdown)
                fh.flush()
        self._append(markdown.replace("**", "").replace("_", ""), tag)

    def _write_and_append_segment(self, start_sec: float, text: str) -> None:
        stamp = fmt_hms(start_sec)
        if self.session is not None:
            with open(self.session.transcript_path, "a", encoding="utf-8") as fh:
                fh.write(f"[{stamp}] {text}\n\n")
                fh.flush()
        self.text.config(state="normal")
        self.text.insert("end", f"[{stamp}] ", "ts")
        self.text.insert("end", f"{text}\n\n")
        self.text.config(state="disabled")
        self.text.see("end")

    def _current_transcript_text(self) -> str:
        if self.session is not None and self.session.transcript_path.exists():
            return self.session.transcript_path.read_text(encoding="utf-8")
        return self.text.get("1.0", "end").strip()

    # -- stato dei pulsanti ------------------------------------------------ #

    def _refresh_buttons(self) -> None:
        s = self.state_name
        recording = s in ("recording", "paused")
        self.btn_start.config(state="normal" if (s == "idle" and self.model_ready) else "disabled")
        self.btn_pause.config(state="normal" if recording else "disabled")
        self.btn_pause.config(text="Riprendi" if s == "paused" else "Pausa")
        self.btn_stop.config(state="normal" if recording else "disabled")
        self.btn_note.config(state="normal" if recording else "disabled")
        self.device_box.config(state="disabled" if recording else "readonly")
        has_file = self.session is not None and self.session.transcript_path.exists()
        self.btn_copy.config(state="normal" if has_file else "disabled")
        self.btn_open.config(state="normal" if has_file else "disabled")
        # L'etichetta del pulsante e' il riscontro piu' immediato: il controllo
        # che hai premuto cambia da solo, senza dover cercare la barra di stato.
        self.btn_highlights.config(
            text="Genero…" if self.generating_highlights else "Punti salienti",
            state="normal" if (has_file and not self.generating_highlights) else "disabled",
        )

    def _on_close(self) -> None:
        if self.state_name in ("recording", "paused"):
            if not messagebox.askokcancel(
                "Registrazione in corso",
                "La sessione e' attiva. Chiudere adesso interrompe la trascrizione "
                "dell'ultimo blocco. Continuare?",
            ):
                return
            try:
                self.session.stop()  # type: ignore[union-attr]
            except Exception:
                pass
        self.destroy()


def main() -> int:
    cfg = load_config()
    app = App(cfg)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
