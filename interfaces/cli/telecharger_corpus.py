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
from tools.rafael.log_tool import LogTool

log = LogTool(origin="telecharger_corpus")


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

    log.START_ACTION("telecharger_corpus", "main", "telechargement d'un corpus depuis le Hub")
    log.PARAMETER_VALUE("identifiant-hub", arguments.identifiant_hub)
    log.PARAMETER_VALUE("configuration", arguments.configuration or "(aucune)")
    log.PARAMETER_VALUE("split", arguments.split)
    log.PARAMETER_VALUE("sortie", arguments.sortie)

    log.STEP(1, "Connexion au Hugging Face Hub", "chargement paresseux, premiere iteration a suivre")
    lecteur = LecteurCorpusHuggingFace(
        identifiant_hub=arguments.identifiant_hub,
        configuration=arguments.configuration,
        split=arguments.split,
    )

    chemin_sortie = Path(arguments.sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    log.STEP(2, "Ecriture JSONL", f"vers {chemin_sortie}")
    nombre_enregistrements = 0
    try:
        with chemin_sortie.open("w", encoding="utf-8") as fichier_sortie:
            for enregistrement in lecteur.lire_enregistrements():
                fichier_sortie.write(json.dumps(enregistrement, ensure_ascii=False) + "\n")
                nombre_enregistrements += 1
                if nombre_enregistrements % 500 == 0:
                    log.LEVEL_7_INFO(
                        "telecharger_corpus",
                        f"{nombre_enregistrements} enregistrements ecrits jusqu'ici...",
                    )
    except ValueError as erreur:
        log.LEVEL_4_ERROR(
            "telecharger_corpus",
            f"echec du telechargement -- split '{arguments.split}' invalide pour "
            f"'{arguments.identifiant_hub}' (configuration={arguments.configuration!r}) : {erreur}",
        )
        raise

    log.PARAMETER_VALUE("enregistrements ecrits", nombre_enregistrements)
    log.FINISH_ACTION("telecharger_corpus", "main", f"{nombre_enregistrements} enregistrements ecrits dans {chemin_sortie}")

    print(f"{nombre_enregistrements} enregistrements ecrits dans {chemin_sortie}")


if __name__ == "__main__":
    main()
