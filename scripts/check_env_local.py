"""
Vérifie que l'environnement local (Environnement A - WSL2, 5 Go RAM)
est correctement configuré pour la préparation des données.

Aucune dépendance GPU n'est testée ici : ce script est volontairement
léger pour fonctionner sur un poste contraint en RAM.

Usage :
    python scripts/check_env_local.py
"""

from __future__ import annotations

import importlib
import shutil
import sys

import psutil


REQUIRED_MODULES = [
    "pandas",
    "datasets",
    "ydata_profiling",
    "presidio_analyzer",
    "presidio_anonymizer",
    "spacy",
    "pydantic",
    "transformers",
    "huggingface_hub",
    "mlflow",
]

MIN_RAM_GB = 4.0  # marge de sécurité sous les 5 Go annoncés


def verifier_version_python() -> bool:
    ok = sys.version_info >= (3, 10)
    statut = "OK" if ok else "ECHEC"
    print(f"  Version Python.......: {sys.version.split()[0]:<15} [{statut}]")
    return ok


def verifier_ram_disponible() -> bool:
    ram_totale_gb = psutil.virtual_memory().total / (1024 ** 3)
    ok = ram_totale_gb >= MIN_RAM_GB
    statut = "OK" if ok else "ATTENTION"
    print(f"  RAM totale...........: {ram_totale_gb:.1f} Go        [{statut}]")
    if not ok:
        print(
            "    -> RAM insuffisante pour le profilage de gros corpus. "
            "Traiter par échantillons (df.sample())."
        )
    return True  # avertissement seulement, ne bloque pas


def verifier_modules() -> bool:
    tous_ok = True
    for nom_module in REQUIRED_MODULES:
        try:
            importlib.import_module(nom_module)
            print(f"  Module {nom_module:<20}: installé   [OK]")
        except ImportError:
            print(f"  Module {nom_module:<20}: MANQUANT   [ECHEC]")
            tous_ok = False
    return tous_ok


def verifier_modeles_spacy() -> bool:
    tous_ok = True
    try:
        import spacy

        for nom_modele in ["fr_core_news_md", "en_core_web_sm"]:
            try:
                spacy.load(nom_modele)
                print(f"  Modèle spaCy {nom_modele:<15}: chargé  [OK]")
            except OSError:
                print(f"  Modèle spaCy {nom_modele:<15}: absent  [ECHEC]")
                print(f"    -> python -m spacy download {nom_modele}")
                tous_ok = False
    except ImportError:
        print("  spaCy non installé, impossible de vérifier les modèles [ECHEC]")
        tous_ok = False
    return tous_ok


def verifier_cli_huggingface() -> bool:
    present = shutil.which("huggingface-cli") is not None
    statut = "OK" if present else "ATTENTION"
    print(f"  CLI huggingface-cli..: {'présent' if present else 'absent':<15} [{statut}]")
    return True


def main() -> None:
    print("=" * 80)
    print("VERIFICATION DE L'ENVIRONNEMENT LOCAL (Environnement A - WSL2)")
    print("=" * 80)

    resultats = [
        verifier_version_python(),
        verifier_ram_disponible(),
        verifier_modules(),
        verifier_modeles_spacy(),
        verifier_cli_huggingface(),
    ]

    print("=" * 80)
    if all(resultats):
        print("Environnement local pret pour la preparation des donnees.")
    else:
        print("Des elements sont manquants : voir le detail ci-dessus.")
        sys.exit(1)


if __name__ == "__main__":
    main()
