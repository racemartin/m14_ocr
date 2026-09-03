"""
Point d'entree CLI — Etape 1, action "profiler".

Injecte les adaptateurs concrets (lecteur + profileur) dans le cas
d'usage `ProfilerCorpusUseCase`. Aucune logique metier ici : ce
fichier ne fait qu'assembler les briques hexagonales.

Usage :
    uv run python interfaces/cli/profiler_corpus.py \
        --source data/raw/mediqal.jsonl \
        --nom MediQAl
"""

from __future__ import annotations

import argparse

from chsa_triage.application.use_cases import ProfilerCorpusUseCase
from chsa_triage.infrastructure.adapters import (
    LecteurCorpusFichierLocal,
    YdataProfileur,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="profiler_corpus")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Chemin du fichier corpus brut (.csv/.jsonl)")
    parser.add_argument("--nom", required=True, help="Nom du corpus (pour le rapport)")
    arguments = parser.parse_args()

    log.START_ACTION("profiler_corpus", "main", "profilage d'un corpus")
    log.PARAMETER_VALUE("source", arguments.source)
    log.PARAMETER_VALUE("nom", arguments.nom)

    lecteur   = LecteurCorpusFichierLocal(arguments.source)
    profileur = YdataProfileur()

    log.STEP(1, "Lecture + profilage ydata-profiling", "peut prendre plusieurs minutes sur un gros corpus")
    try:
        cas_usage = ProfilerCorpusUseCase(lecteur=lecteur, profileur=profileur)
        rapport = cas_usage.executer(arguments.nom)
    except Exception as erreur:
        log.LEVEL_4_ERROR("profiler_corpus", f"echec du profilage de {arguments.source} : {erreur}")
        raise

    log.PARAMETER_VALUE("enregistrements", rapport.nombre_enregistrements)
    log.PARAMETER_VALUE("taux de doublons", f"{rapport.taux_doublons:.2%}")
    log.PARAMETER_VALUE("rapport detaille", rapport.chemin_rapport_detaille)
    log.FINISH_ACTION("profiler_corpus", "main", f"rapport ecrit pour {arguments.nom}")

    print("=" * 80)
    print(f"RAPPORT DE PROFILAGE — {arguments.nom}")
    print("=" * 80)
    print(f"  Enregistrements.......: {rapport.nombre_enregistrements}")
    print(f"  Taux de doublons......: {rapport.taux_doublons:.2%}")
    print(f"  Rapport detaille......: {rapport.chemin_rapport_detaille}")
    print("  Valeurs manquantes par colonne :")
    for colonne, taux in rapport.taux_valeurs_manquantes.items():
        print(f"    {colonne:<30}: {taux:.2%}")


if __name__ == "__main__":
    main()
