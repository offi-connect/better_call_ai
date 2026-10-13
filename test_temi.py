"""
Test dei temi dell'interfaccia.

Verifica che ogni tema sia completo e, soprattutto, leggibile: il contrasto
viene misurato con la formula WCAG invece di essere giudicato a occhio.
Scegliendo le tinte a mano quattro combinazioni erano sotto il minimo e
nessuna di esse sembrava sbagliata guardando il codice.

Non apre la GUI e non richiede nulla installato oltre a verbale.py.

Avvio:  python test_temi.py
"""

from __future__ import annotations

import sys
import types
from unittest import mock

for nome in (
    "tkinter", "tkinter.ttk", "tkinter.messagebox",
    "tkinter.scrolledtext", "tkinter.filedialog",
):
    modulo = types.ModuleType(nome)
    modulo.__getattr__ = lambda _n: mock.MagicMock()
    sys.modules.setdefault(nome, modulo)

import verbale  # noqa: E402

esiti: list[bool] = []


def verifica(descrizione: str, condizione: bool) -> None:
    esiti.append(bool(condizione))
    print(f"   {'OK:' if condizione else 'FALLITO:'} {descrizione}")


def _luminanza(colore: str) -> float:
    """Luminanza relativa secondo WCAG 2.1."""
    canali = [int(colore[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lineari = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in canali
    ]
    return 0.2126 * lineari[0] + 0.7152 * lineari[1] + 0.0722 * lineari[2]


def contrasto(a: str, b: str) -> float:
    la, lb = _luminanza(a), _luminanza(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# Soglie: 4,5 e' il minimo WCAG per il testo normale, 3,0 per testo grande e
# componenti dell'interfaccia. Il testo disabilitato deve restare leggibile
# quanto basta a capire cosa dice il pulsante, non di piu'.
COPPIE = [
    ("testo su sfondo", "testo", "sfondo", 4.5),
    ("testo su superficie", "testo", "superficie", 4.5),
    ("testo del pulsante", "testo_accento", "accento", 4.5),
    ("pulsante sotto il mouse", "testo_accento", "accento_vivo", 4.0),
    ("pulsante premuto", "testo_accento", "accento_scuro", 4.0),
    ("didascalie su sfondo", "testo_tenue", "sfondo", 3.0),
    ("bordo su superficie", "bordo", "superficie", 1.4),
    ("esito positivo", "ok", "sfondo", 3.0),
    ("esito negativo", "allarme", "sfondo", 3.0),
    ("testo disabilitato", "spento_testo", "spento", 1.8),
    ("selezione leggibile", "testo", "selezione", 4.5),
    # Il vumetro e' un fondo scuro su cui si accendono segmenti colorati: deve
    # staccare dalla finestra, non ospitare testo.
    ("vumetro stacca dallo sfondo", "vumetro", "sfondo", 3.0),
]


def test_completezza() -> None:
    print("--- 1. ogni tema definisce tutti i ruoli ---")
    ruoli = set(verbale.TEMI["azzurro"])
    verifica("il tema di riferimento ha dei ruoli", len(ruoli) > 5)
    for nome, palette in verbale.TEMI.items():
        mancanti = ruoli - set(palette)
        in_piu = set(palette) - ruoli
        verifica(f"{nome}: nessun ruolo mancante {sorted(mancanti) or ''}", not mancanti)
        verifica(f"{nome}: nessun ruolo di troppo {sorted(in_piu) or ''}", not in_piu)


def test_forma_dei_colori() -> None:
    print("\n--- 2. i valori sono colori validi ---")
    for nome, palette in verbale.TEMI.items():
        cattivi = [
            f"{k}={v}"
            for k, v in palette.items()
            if not (
                isinstance(v, str)
                and len(v) == 7
                and v[0] == "#"
                and all(c in "0123456789abcdefABCDEF" for c in v[1:])
            )
        ]
        verifica(f"{nome}: tutti #rrggbb {cattivi or ''}", not cattivi)


def test_menu() -> None:
    print("\n--- 3. il menu e i temi non divergono ---")
    chiavi = [v for _, v in verbale.TEMI_INTERFACCIA]
    verifica("ogni voce del menu esiste", all(k in verbale.TEMI for k in chiavi))
    verifica("ogni tema e' raggiungibile dal menu", set(chiavi) == set(verbale.TEMI))
    verifica("nessuna voce duplicata", len(set(chiavi)) == len(chiavi))
    verifica(
        "le etichette non sono vuote",
        all(e.strip() for e, _ in verbale.TEMI_INTERFACCIA),
    )
    verifica(
        "il predefinito di fabbrica e' selezionabile",
        verbale.DEFAULT_CONFIG["interfaccia"]["tema"] in chiavi,
    )
    verifica(
        "la palette in uso parte dal predefinito",
        set(verbale.COLORI) == set(verbale.TEMI["azzurro"]),
    )


def test_leggibilita() -> None:
    print("\n--- 4. contrasto misurato, non giudicato a occhio ---")
    for nome, palette in verbale.TEMI.items():
        print(f"   {nome}:")
        for descrizione, primo, secondo, minimo in COPPIE:
            rapporto = contrasto(palette[primo], palette[secondo])
            ok = rapporto >= minimo
            esiti.append(ok)
            segno = "OK: " if ok else "FALLITO:"
            print(f"     {segno} {descrizione:26s} {rapporto:5.2f} (min {minimo})")


def main() -> int:
    test_completezza()
    test_forma_dei_colori()
    test_menu()
    test_leggibilita()
    print()
    if all(esiti):
        print(f"Tutti i test superati ({len(esiti)}).")
        return 0
    print(f"{esiti.count(False)} test falliti su {len(esiti)}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
