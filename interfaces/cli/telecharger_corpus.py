"""
Point d'entree CLI -- telecharger un corpus brut depuis Hugging Face
Hub et l'exporter en JSONL local (data/raw/).

Ferme le vide identifie le 03/09/2026 : l'adaptateur
`LecteurCorpusHuggingFace` existait deja et etait teste, mais aucun
script ne l'invoquait -- tous les CLI de l'Etape 1 supposaient que
`data/raw/*.jsonl` existait deja.

Usage :
    uv run python interfaces/cli/telecharger_corpus.py \
        --identifiant-hub ANR-MALADES/MediQAl \
        --sortie data/raw/mediqal.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chsa_triage.infrastructure.adapters import LecteurCorpusHuggingFace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identifiant-hub",
        required=True,
        help="Chemin du dataset sur le Hub, ex. ANR-MALADES/MediQAl",
    )
    parser.add_argument(
        "--configuration",
        default=None,
        help="Sous-configuration HF si le dataset en definit plusieurs",
    )
    parser.add_argument("--split", default="train", help="Split a telecharger (defaut: train)")
    parser.add_argument("--sortie", required=True, help="Chemin JSONL de sortie (ex. data/raw/mediqal.jsonl)")
    arguments = parser.parse_args()

    lecteur = LecteurCorpusHuggingFace(
        identifiant_hub=arguments.identifiant_hub,
        configuration=arguments.configuration,
        split=arguments.split,
    )

    chemin_sortie = Path(arguments.sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    nombre_enregistrements = 0
    with chemin_sortie.open("w", encoding="utf-8") as fichier_sortie:
        for enregistrement in lecteur.lire_enregistrements():
            fichier_sortie.write(json.dumps(enregistrement, ensure_ascii=False) + "\n")
            nombre_enregistrements += 1

    print(f"{nombre_enregistrements} enregistrements ecrits dans {chemin_sortie}")


if __name__ == "__main__":
    main()
