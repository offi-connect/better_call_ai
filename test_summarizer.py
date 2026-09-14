"""
Test della scelta automatica del motore per i punti salienti.

Non apre la GUI e non richiede una chiave API: le parti che dipendono da
Ollama o dal cloud vengono saltate se quel motore non e' disponibile.

Avvio:  python test_summarizer.py
"""

from __future__ import annotations

import sys
import types
from unittest import mock

# Stub di tkinter: serve solo a rendere importabile verbale.py senza display.
for nome in ("tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.scrolledtext"):
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
    verifica("engine predefinito = auto", cfg["summary"].get("engine") == "auto")
    verifica("soglia definita", cfg["summary"].get("soglia_parole", 0) > 0)


def test_fabbrica() -> None:
    print("\n--- 2. build_summarizer rispetta engine ---")
    base = verbale.load_config()["summary"]
    for motore, atteso in [
        ("local", verbale.LocalSummarizer),
        ("cloud", verbale.CloudSummarizer),
        ("auto", verbale.AutoSummarizer),
    ]:
        costruito = verbale.build_summarizer({"summary": dict(base, engine=motore)})
        verifica(f"engine={motore} -> {atteso.__name__}", isinstance(costruito, atteso))


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
        "cloud": verbale.CloudSummarizer,
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
    auto = verbale.build_summarizer(verbale.load_config())
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
    test_blocchi()
    test_voci_menu()
    test_pulizia()
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
