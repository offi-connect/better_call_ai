"""
Test della scelta automatica del motore per i punti salienti.

Non apre la GUI e non richiede una chiave API: le parti che dipendono da
Ollama o dal cloud vengono saltate se quel motore non e' disponibile.

Avvio:  python test_summarizer.py
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest import mock

# Stub di tkinter: serve solo a rendere importabile verbale.py senza display.
# L'elenco deve coprire ogni sottomodulo importato da verbale.py.
for nome in (
    "tkinter", "tkinter.ttk", "tkinter.messagebox",
    "tkinter.scrolledtext", "tkinter.filedialog",
):
    modulo = types.ModuleType(nome)
    modulo.__getattr__ = lambda _n: mock.MagicMock()
    sys.modules.setdefault(nome, modulo)

import verbale  # noqa: E402

CORTO = "Marco: approvato, Davide sulla fatturazione da lunedi'. " * 5
LUNGO = "Marco: approvato, Davide sulla fatturazione da lunedi'. " * 200

esiti: list[bool] = []


def verifica(descrizione: str, condizione: bool) -> None:
    esiti.append(bool(condizione))
    print(f"   {'OK:' if condizione else 'FALLITO:'} {descrizione}")


def test_configurazione() -> None:
    print("--- 1. la chiave summary sopravvive al merge con config.json ---")
    cfg = verbale.load_config()
    verifica("summary presente", "summary" in cfg)
    # Il predefinito si controlla su DEFAULT_CONFIG: nel config.json c'e' la
    # preferenza scelta dalla tendina, che e' legittimamente diversa.
    verifica(
        "predefinito di fabbrica = auto",
        verbale.DEFAULT_CONFIG["summary"]["engine"] == "auto",
    )
    scelto = cfg["summary"].get("engine")
    verifica(
        f"preferenza salvata ({scelto!r}) e' una voce valida del menu",
        scelto in [v for _, v in verbale.MOTORI_SINTESI],
    )
    verifica("soglia definita", cfg["summary"].get("soglia_parole", 0) > 0)


def test_fabbrica() -> None:
    print("\n--- 2. build_summarizer rispetta engine ---")
    base = verbale.load_config()["summary"]
    for motore, atteso in [
        ("local", verbale.LocalSummarizer),
        ("cloud", verbale.RemoteSummarizer),
        ("auto", verbale.AutoSummarizer),
    ]:
        costruito = verbale.build_summarizer({"summary": dict(base, engine=motore)})
        verifica(f"engine={motore} -> {atteso.__name__}", isinstance(costruito, atteso))


def test_fornitori_remoti() -> None:
    print("\n--- 2b. la scelta del fornitore remoto ---")
    cloud = verbale.load_config()["summary"]["cloud"]

    verifica(
        "gemini -> GeminiSummarizer",
        isinstance(
            verbale.build_cloud_summarizer(dict(cloud, provider="gemini")),
            verbale.GeminiSummarizer,
        ),
    )
    verifica(
        "anthropic -> CloudSummarizer",
        isinstance(
            verbale.build_cloud_summarizer(dict(cloud, provider="anthropic")),
            verbale.CloudSummarizer,
        ),
    )
    verifica(
        "entrambi sono motori remoti",
        all(
            issubclass(c, verbale.RemoteSummarizer)
            for c in verbale.FORNITORI_REMOTI.values()
        ),
    )

    # Un refuso nel config deve dare un errore che si legge, non un KeyError.
    for rotto, atteso in [("gemnini", "sconosciuto"), ("", "sconosciuto")]:
        try:
            verbale.build_cloud_summarizer(dict(cloud, provider=rotto))
            verifica(f"provider {rotto!r} rifiutato", False)
        except RuntimeError as exc:
            verifica(f"provider {rotto!r} rifiutato con spiegazione", atteso in str(exc))
        except Exception as exc:
            verifica(f"provider {rotto!r} -> RuntimeError, non {type(exc).__name__}", False)

    # Sezione mancante: altrettanto esplicito.
    try:
        verbale.build_cloud_summarizer({"provider": "gemini"})
        verifica("sezione mancante rifiutata", False)
    except RuntimeError as exc:
        verifica("sezione mancante segnalata", "summary.cloud.gemini" in str(exc))


def test_richiesta_gemini() -> None:
    print("\n--- 2c. la richiesta Gemini usa i nomi veri dell'SDK ---")
    cfg = verbale.load_config()["summary"]["cloud"]["gemini"]
    motore = verbale.GeminiSummarizer(cfg)

    # La configurazione e' un dizionario, e ogni dettaglio qui sotto e' una
    # lezione pagata con una chiamata rifiutata:
    #
    # 1. un types.GenerateContentConfig si costruiva benissimo e il test
    #    passava, ma interactions.create lo rifiutava;
    # 2. anche types.GenerationConfig veniva rifiutato — ne esiste una seconda
    #    classe interna in _gaos, e la forma pubblica che funziona e' il dict;
    # 3. thinking_level e' una chiave di primo livello, non dentro un
    #    thinking_config;
    # 4. il valore va minuscolo: 'LOW' torna 400 dal server.
    conf = motore._configurazione()
    verifica("la configurazione e' un dizionario", isinstance(conf, dict))
    verifica(
        "max_output_tokens presente",
        conf.get("max_output_tokens") == cfg.get("max_tokens", 1024),
    )
    verifica("nessun thinking_config annidato", "thinking_config" not in conf)
    livello = conf.get("thinking_level")
    verifica("thinking_level di primo livello", livello is not None)
    verifica(
        "thinking_level minuscolo e fra i valori del protocollo",
        livello in {"minimal", "low", "medium", "high"},
    )
    verifica("temperature non passata (deprecata)", "temperature" not in conf)

    # Anche in maiuscolo nel config, deve uscire minuscolo.
    urlante = verbale.GeminiSummarizer(dict(cfg, thinking_level="LOW"))
    verifica(
        "un 'LOW' nel config viene normalizzato",
        urlante._configurazione().get("thinking_level") == "low",
    )

    # I guasti temporanei vanno distinti da quelli definitivi.
    verifica(
        "503 riconosciuto come temporaneo",
        verbale.GeminiSummarizer._e_temporaneo(Exception("Error code: 503 service_unavailable")),
    )
    verifica(
        "chiave sbagliata NON e' temporanea",
        not verbale.GeminiSummarizer._e_temporaneo(Exception("API key not valid")),
    )

    verifica(
        "available() segue la variabile d'ambiente",
        motore.available()
        == bool(os.environ.get(cfg["api_key_env"], "").strip()),
    )
    verifica(
        "il piano gratuito e' dichiarato nell'etichetta",
        "piano gratuito" in motore._etichetta(),
    )
    verifica("plan() dichiara che e' remoto", motore.plan("ciao")[1] is True)

    print("   errori tradotti in italiano leggibile:")
    casi = [
        ("429 RESOURCE_EXHAUSTED: quota exceeded", "Limite del piano Gemini"),
        ("API key not valid. Please pass a valid API key.", "Chiave Gemini rifiutata"),
        ("403 PERMISSION_DENIED", "Chiave Gemini rifiutata"),
        ("connessione interrotta", "Gemini non ha risposto"),
    ]
    for grezzo, atteso in casi:
        tradotto = verbale.GeminiSummarizer._traduci_errore(Exception(grezzo))
        verifica(f"{grezzo[:34]!r}", atteso in tradotto)

    print("   estrazione del testo dalla risposta:")

    class _Blocco:
        def __init__(self, testo):
            self.text = testo

    class _Passo:
        def __init__(self, testi):
            self.content = [_Blocco(t) for t in testi]

    class _Diretta:
        output_text = "Decisioni\n- rinnovo approvato"

    class _APassi:
        output_text = ""
        steps = [_Passo(["Azioni\n", "- mandare la mail"])]

    class _Vuota:
        output_text = ""
        steps = []

    verifica("output_text usato quando c'e'", "rinnovo" in verbale._testo_gemini(_Diretta()))
    verifica("ripiego sui passi quando manca", "mail" in verbale._testo_gemini(_APassi()))
    verifica("risposta vuota -> stringa vuota", verbale._testo_gemini(_Vuota()) == "")


def test_disponibilita_locale() -> None:
    print("\n--- 2d. available() guarda il modello, non solo il server ---")
    import io
    import json as _json

    class _Risposta(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _finto_urlopen(corpo):
        def _apri(url, timeout=None):
            return _Risposta(_json.dumps(corpo).encode())

        return _apri

    originale = verbale.urllib.request.urlopen
    try:
        elencati = {"models": [{"name": "gemma3:latest"}, {"name": "altro:7b"}]}
        verbale.urllib.request.urlopen = _finto_urlopen(elencati)
        verifica(
            "'gemma3' combacia con 'gemma3:latest'",
            verbale.LocalSummarizer({"model": "gemma3"}).available(),
        )
        verifica(
            "'gemma3:latest' combacia",
            verbale.LocalSummarizer({"model": "gemma3:latest"}).available(),
        )
        verifica(
            "'gemma3:4b' non combacia con 'gemma3:latest'",
            not verbale.LocalSummarizer({"model": "gemma3:4b"}).available(),
        )

        # Il caso che ha motivato la modifica: Ollama risponde, lista vuota.
        verbale.urllib.request.urlopen = _finto_urlopen({"models": []})
        vuoto = verbale.LocalSummarizer({"model": "gemma3:4b"})
        verifica("server attivo ma senza modelli -> non disponibile", not vuoto.available())
        verifica("lista vuota -> insieme vuoto", vuoto.modelli_disponibili() == set())
    finally:
        verbale.urllib.request.urlopen = originale

    morto = verbale.LocalSummarizer({"model": "x", "host": "http://127.0.0.1:1"})
    verifica("host irraggiungibile -> non disponibile", not morto.available())
    verifica("host irraggiungibile -> insieme vuoto", morto.modelli_disponibili() == set())


def test_blocchi() -> None:
    print("\n--- 3. la divisione in blocchi non lascia scoperto nessun caso ---")
    locale = verbale.LocalSummarizer({"model": "x", "parole_per_blocco": 10})

    righe = "\n".join(f"riga {i} con qualche parola in piu" for i in range(20))
    blocchi = locale._split(righe)
    verifica("testo su piu' righe spezzato", len(blocchi) > 1)
    verifica(
        "nessuna parola persa",
        len(" ".join(blocchi).split()) == len(righe.split()),
    )

    # Caso che in passato sfuggiva: una riga sola piu' lunga del limite.
    unica = " ".join(f"parola{i}" for i in range(95))
    blocchi = locale._split(unica)
    verifica("riga unica lunghissima spezzata", len(blocchi) > 1)
    verifica(
        "ogni blocco rispetta il limite",
        all(len(b.split()) <= 10 for b in blocchi),
    )
    verifica(
        "nessuna parola persa nel taglio forzato",
        len(" ".join(blocchi).split()) == 95,
    )


def test_voci_menu() -> None:
    print("\n--- 3b. le voci del menu corrispondono ai motori reali ---")
    base = verbale.load_config()["summary"]
    valori = [v for _, v in verbale.MOTORI_SINTESI]

    verifica("tre voci previste", len(verbale.MOTORI_SINTESI) == 3)
    verifica("nessun valore duplicato", len(set(valori)) == len(valori))
    verifica(
        "le etichette sono in italiano e non vuote",
        all(e.strip() for e, _ in verbale.MOTORI_SINTESI),
    )

    # Ogni valore del menu deve produrre un sommarizzatore, non un KeyError.
    attesi = {
        "auto": verbale.AutoSummarizer,
        "local": verbale.LocalSummarizer,
        "cloud": verbale.RemoteSummarizer,
    }
    for valore in valori:
        costruito = verbale.build_summarizer({"summary": dict(base, engine=valore)})
        verifica(
            f"'{valore}' -> {attesi[valore].__name__}",
            isinstance(costruito, attesi[valore]),
        )

    verifica(
        "il valore predefinito compare fra le voci",
        base.get("engine", "auto") in valori,
    )


def test_pulizia() -> None:
    print("\n--- 4. ripulitura di cio' che i modelli piccoli aggiungono da soli ---")

    # Uscita reale di qwen2.5:3b su una trascrizione di due righe.
    sporco = (
        "Decisioni\n"
        "-\n\n"
        "Azioni\n"
        "- Non c'e' stato detto nulla di specifico da decidere\n\n"
        "Temi discussi\n"
        "- Vacanza: mare e montagna\n"
    )
    pulito = verbale.pulisci_elenco(sporco)
    verifica("sezione vuota rimossa", "Decisioni" not in pulito)
    verifica("commento sull'assenza rimosso", "nulla di specifico" not in pulito)
    verifica("sezione rimasta senza voci rimossa", "Azioni" not in pulito)
    verifica("il contenuto vero sopravvive", "mare e montagna" in pulito)

    # I segnaposto del formato, ricopiati invece che riempiti.
    segnaposto = "Azioni\n- <nessuno> - <chi non specificato> - <entro oggi>\n"
    verifica("voce fatta solo di segnaposto rimossa", verbale.pulisci_elenco(segnaposto) == "")

    # Da non buttare via: una decisione negativa e' comunque una decisione.
    legittima = "Decisioni\n- Non si procede con la migrazione del database"
    verifica(
        "decisione negativa conservata",
        "migrazione" in verbale.pulisci_elenco(legittima),
    )

    # Un importo con il minore non deve far sparire la voce.
    importo = "Decisioni\n- Approvata la spesa < 2000 euro"
    verifica("voce con < conservata", "2000" in verbale.pulisci_elenco(importo))

    verifica("elenco interamente vuoto -> stringa vuota", verbale.pulisci_elenco("Decisioni\n- \n") == "")

    # Casi presi da un'uscita reale su una chiamata di 76 minuti: il modello
    # aveva emesso le intestazioni come voci di elenco, col grassetto Markdown.
    intestazioni_travestite = (
        "- Decisioni**\n"
        "- Da decidere**\n"
        "- Come implementare l'automatizzazione.\n"
        "- Temi discussi**\n"
        "- Ticketing"
    )
    r = verbale.pulisci_elenco(intestazioni_travestite)
    verifica("grassetto Markdown rimosso", "**" not in r)
    verifica("intestazione travestita da voce riconosciuta", "- Decisioni" not in r)
    verifica("sezione vuota cosi' scoperta viene rimossa", "Decisioni" not in r)
    verifica("le voci vere restano sotto la loro sezione", "Temi discussi" in r and "Ticketing" in r)

    # Eco letterale dei segnaposto descrittivi del prompt.
    eco = (
        "Decisioni\n"
        "- cosa è stato deciso: Non sono state fatte decisioni specifiche.\n"
        "Da decidere\n"
        "- questione lasciata aperta\n"
        "Azioni\n"
        "- prepararsi per Power Automate"
    )
    r = verbale.pulisci_elenco(eco)
    verifica("plurale 'non sono state fatte' intercettato", "Non sono state" not in r)
    verifica("segnaposto descrittivo ricopiato rimosso", "questione lasciata aperta" not in r)
    verifica("l'azione vera sopravvive", "Power Automate" in r)


def test_nomi_sospetti() -> None:
    print("\n--- 4b. marcatura dei nomi propri inventati ---")

    # Il caso reale: da un unico "targata" biascicato il modello ha coniato
    # prima "Targasys" poi "Infotarga", come se fossero fornitori esistenti.
    trascrizione = (
        "[00:18:15] i clienti di fatto sono una targata in veicolo per noi.\n"
        "[00:22:03] poi quello che vede con Power BI, ti fanno statistiche.\n"
        "[00:31:40] uno come reseller poi vendi la cosa alle officine.\n"
        "[00:40:12] chiamano sistemi RAG, praticamente alimentano chat.\n"
        "[00:44:01] mi concordi l'export in CSV, io lo rimporto senza problemi.\n"
        "[00:51:30] le rigiro a Cristian le slide.\n"
        "[00:55:18] e' la FOMO, la paura di restare fuori dalla rete.\n"
    )
    uscita = (
        "Decisioni\n"
        "- Utilizzare il servizio Infotarga per la gestione delle targhe.\n"
        "Da decidere\n"
        "- Definizione della politica di visibilita' delle officine.\n"
        "- Scelta delle funzionalita' da implementare subito.\n"
        "Azioni\n"
        "- Rigirare le slide a Cristian.\n"
        "- Verificare il collegamento con il CSV.\n"
        "Temi discussi\n"
        "- Evoluzione verso la diagnostica con sistemi RAG.\n"
        "- Corsi di formazione su Power BI.\n"
        "- Sostenibilita' del modello a reseller.\n"
    )
    marcato, sospetti = verbale.marca_nomi_sospetti(uscita, trascrizione)

    verifica("il nome inventato e' segnalato", "Infotarga" in sospetti)
    verifica("nel testo compare marcato", "Infotarga [?]" in marcato)
    verifica("segnalato una volta sola", sospetti.count("Infotarga") == 1)

    # Le sei parole che il primo euristico sbagliava: comuni, maiuscole solo
    # perche' aprono una voce di elenco.
    print("   parole comuni a inizio voce, non piu' confuse con nomi propri:")
    for comune in ("Definizione", "Scelta", "Evoluzione", "Sostenibilita'", "Rigirare", "Corsi"):
        verifica(f"{comune} non segnalata", comune not in sospetti)

    print("   nomi e acronimi che SONO nella trascrizione:")
    for vero in ("Cristian", "CSV", "RAG", "Power", "BI"):
        verifica(f"{vero} non segnalato", vero not in sospetti)

    print("   casi limite:")
    verifica("in totale un solo sospetto", len(sospetti) == 1)
    verifica(
        "testo vuoto -> nessun sospetto",
        verbale.marca_nomi_sospetti("", trascrizione) == ("", []),
    )
    verifica(
        "trascrizione vuota -> non marca nulla",
        verbale.marca_nomi_sospetti(uscita, "")[1] == [],
    )
    verifica(
        "confronto insensibile alle maiuscole",
        not verbale.marca_nomi_sospetti("- parla con Reseller", trascrizione)[1],
    )
    doppio, _ = verbale.marca_nomi_sospetti(
        "- Infotarga serve. Poi Infotarga di nuovo.", trascrizione
    )
    verifica("tutte le occorrenze marcate", doppio.count("[?]") == 2)
    verifica(
        "una marcatura non si somma a se stessa",
        "[?] [?]" not in verbale.marca_nomi_sospetti(doppio, trascrizione)[0],
    )

    print("   scioglimenti di sigle fra parentesi (caso osservato dal vero):")
    glossa = "- logiche di FOMO (Fear Of Missing Out) per le officine."
    _, sosp_glossa = verbale.marca_nomi_sospetti(glossa, trascrizione)
    verifica("'Fear' non segnalato", "Fear" not in sosp_glossa)
    verifica("nessuna parola della glossa segnalata", sosp_glossa == [])
    # Ma una parentesi che NON scioglie una sigla resta sotto esame.
    _, sosp_paren = verbale.marca_nomi_sospetti(
        "- il modulo (gestito da Acme) va rifatto.", trascrizione
    )
    verifica("un nome in parentesi normale e' ancora segnalato", "Acme" in sosp_paren)

    print("   l'avviso che finisce nel verbale:")
    uno = verbale.Highlights("x", "cloud", "m", ["Infotarga"])
    due = verbale.Highlights("x", "cloud", "m", ["Infotarga", "Acme"])
    nessuno = verbale.Highlights("x", "cloud", "m")
    verifica("singolare", "Un nome non compare" in uno.avviso_sospetti())
    verifica("plurale", "2 nomi non compaiono" in due.avviso_sospetti())
    verifica("senza sospetti nessun avviso", nessuno.avviso_sospetti() == "")


def test_ritrascrizione() -> None:
    print("\n--- 4c. ritrascrizione differita ---")
    cfg = verbale.load_config()
    valori = [v for _, v in verbale.MOTORI_RITRASCRIZIONE]

    verifica("quattro motori nel menu", len(verbale.MOTORI_RITRASCRIZIONE) == 4)
    verifica("nessun valore duplicato", len(set(valori)) == len(valori))
    verifica(
        "il predefinito e' fra le voci",
        cfg["ritrascrizione"]["motore"] in valori,
    )
    # Il predefinito deve essere locale: un menu che manda l'audio a un
    # servizio esterno senza una scelta esplicita e' una trappola.
    verifica(
        "il predefinito e' locale",
        verbale.DEFAULT_CONFIG["ritrascrizione"]["motore"].startswith("whisper"),
    )
    verifica(
        "ogni valore ha la forma tipo:modello",
        all(len(v.split(":")) == 2 and all(v.split(":")) for v in valori),
    )
    verifica(
        "i tipi sono solo whisper e gemini",
        {v.split(":")[0] for v in valori} == {"whisper", "gemini"},
    )

    print("   nomi dei file prodotti:")
    nomi = [verbale.nome_file_ritrascrizione(v) for v in valori]
    for valore, nome in zip(valori, nomi):
        print(f"     {valore:30s} -> {nome}")
    verifica("tutti distinti", len(set(nomi)) == len(nomi))
    verifica("nessuno sovrascrive verbale.md", "verbale.md" not in nomi)
    vietati = set('<>:"/\\|?*')
    verifica(
        "nessun carattere vietato su Windows",
        all(not set(n) & vietati for n in nomi),
    )

    try:
        verbale.ritrascrivi(Path("x.wav"), "inesistente:1", cfg)
        verifica("motore ignoto rifiutato", False)
    except RuntimeError as exc:
        verifica("motore ignoto rifiutato con spiegazione", "sconosciuto" in str(exc))
    except Exception as exc:
        verifica(f"motore ignoto -> RuntimeError, non {type(exc).__name__}", False)

    print("   lettura del wav (mono, stereo, ricampionato):")
    import tempfile
    import wave as _wave

    def scrivi(percorso, canali, sr, campioni):
        with _wave.open(str(percorso), "wb") as wf:
            wf.setnchannels(canali)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(campioni.astype("<i2").tobytes())

    import numpy as _np

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        uno = (_np.ones(verbale.TARGET_SR) * 1000)
        scrivi(base / "mono.wav", 1, verbale.TARGET_SR, uno)
        letto = verbale.leggi_wav_mono16k(base / "mono.wav")
        verifica("mono 16k: un secondo resta un secondo", len(letto) == verbale.TARGET_SR)
        verifica("ampiezza normalizzata sotto 1", float(_np.max(_np.abs(letto))) < 1.0)

        stereo = _np.repeat(uno, 2)
        scrivi(base / "stereo.wav", 2, verbale.TARGET_SR, stereo)
        letto = verbale.leggi_wav_mono16k(base / "stereo.wav")
        verifica("stereo ridotto a mono", len(letto) == verbale.TARGET_SR)

        scrivi(base / "48k.wav", 1, 48000, _np.ones(48000) * 1000)
        letto = verbale.leggi_wav_mono16k(base / "48k.wav")
        verifica(
            "48 kHz ricampionato a 16 kHz",
            abs(len(letto) - verbale.TARGET_SR) <= 1,
        )


def test_banner() -> None:
    print("\n--- 5. il banner dichiara sempre dove e' finito il testo ---")
    locale = verbale.Highlights("x", "local", "qwen2.5:3b").banner()
    cloud = verbale.Highlights("x", "cloud", "claude-haiku-4-5").banner()
    verifica("locale marcato [LOCALE]", locale.startswith("[LOCALE]"))
    verifica("locale rassicura sulla privacy", "nessun dato" in locale)
    verifica("cloud marcato [CLOUD]", cloud.startswith("[CLOUD]"))
    verifica("cloud avvisa dell'invio esterno", "servizio esterno" in cloud)


def test_scelta() -> None:
    print("\n--- 6. scelta automatica in base alla lunghezza ---")
    # Costruito a mano, non via build_summarizer: quello segue la preferenza
    # salvata dalla tendina, che l'utente puo' avere messo su "local" o
    # "cloud". Il test della scelta automatica deve provare l'automatico.
    auto = verbale.AutoSummarizer(verbale.load_config()["summary"])
    ollama_ok = auto.local.available()
    cloud_ok = auto.cloud.available()
    print(f"   (Ollama: {ollama_ok}, chiave cloud: {cloud_ok})")

    if not ollama_ok and not cloud_ok:
        print("   saltato: nessun motore disponibile")
        return

    if ollama_ok:
        _, e_cloud = auto.plan(CORTO)
        verifica("testo corto elaborato in locale", not e_cloud)

    etichetta, e_cloud = auto.plan(LUNGO)
    if cloud_ok:
        verifica("testo lungo mandato al cloud", e_cloud)
    else:
        verifica("senza chiave il lungo ripiega sul locale", not e_cloud)
        verifica("il ripiego viene dichiarato", "ripiego" in etichetta)
        verifica("il lungo viene spezzato in blocchi", "blocchi" in etichetta)


def main() -> int:
    test_configurazione()
    test_fabbrica()
    test_fornitori_remoti()
    test_richiesta_gemini()
    test_disponibilita_locale()
    test_blocchi()
    test_voci_menu()
    test_pulizia()
    test_nomi_sospetti()
    test_ritrascrizione()
    test_banner()
    test_scelta()

    print()
    if all(esiti):
        print(f"Tutti i test superati ({len(esiti)}).")
        return 0
    print(f"{esiti.count(False)} test falliti su {len(esiti)}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
