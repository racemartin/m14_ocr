"""
Point d'entree CLI — Etape 1, action "profiler".

Injecte les adaptateurs concrets (lecteur + profileur) dans le cas
d'usage `ProfilerCorpusUseCase`. Aucune logique metier ici : ce
fichier ne fait qu'assembler les briques hexagonales.

Usage :
    uv run python interfaces/cli/profiler_corpus.py \
        --source data/raw/mediqal.jsonl \
        --nom MediQAl

Option --bloque (03/09/2026, ajoutee suite a un OOM reel sur
ultramedical_preference.jsonl, 966 Mo, 5.8 Go de RAM disponibles) :
sans --bloque, tout le corpus est charge en memoire d'un coup (comme
avant) -- inchange, c'est toujours le chemin par defaut. Avec
--bloque N, le corpus est lu par blocs de N enregistrements
(LecteurCorpusFichierLocal(taille_bloc=N), lecture pandas chunksize)
et CHAQUE bloc produit son propre rapport ydata-profiling complet
(<nom>_blocNNN) -- aucun enregistrement n'est ignore ni echantillonne,
seule la memoire de pointe est bornee. Contrepartie assumee : les
correlations/doublons sont calcules par bloc, pas globalement sur tout
le corpus (un doublon a cheval sur deux blocs n'est pas detecte).
"""

from __future__ import annotations

import argparse
import itertools

from chsa_triage.application.use_cases import ProfilerCorpusUseCase
from chsa_triage.infrastructure.adapters import (
    LecteurCorpusFichierLocal,
    YdataProfileur,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="profiler_corpus")


def _afficher_rapport(nom: str, rapport) -> None:
    print("=" * 80)
    print(f"RAPPORT DE PROFILAGE — {nom}")
    print("=" * 80)
    print(f"  Enregistrements.......: {rapport.nombre_enregistrements}")
    print(f"  Taux de doublons......: {rapport.taux_doublons:.2%}")
    print(f"  Rapport detaille......: {rapport.chemin_rapport_detaille}")
    print("  Valeurs manquantes par colonne :")
    for colonne, taux in rapport.taux_valeurs_manquantes.items():
        print(f"    {colonne:<30}: {taux:.2%}")


def _profiler_par_blocs(source: str, nom: str, taille_bloc: int) -> None:
    """Profile le corpus par blocs de taille_bloc enregistrements, memoire bornee."""
    lecteur   = LecteurCorpusFichierLocal(source, taille_bloc=taille_bloc)
    profileur = YdataProfileur()

    log.STEP(1, "Lecture + profilage par blocs", f"taille_bloc={taille_bloc}, peut prendre du temps")
    iterateur = iter(lecteur.lire_enregistrements())
    numero_bloc = 0
    total_enregistrements = 0
    try:
        while True:
            bloc = list(itertools.islice(iterateur, taille_bloc))
            if not bloc:
                break
            nom_bloc = f"{nom}_bloc{numero_bloc:03d}"
            rapport = profileur.profiler(bloc, nom_bloc)
            log.PARAMETER_VALUE(
                f"bloc {numero_bloc}",
                f"{rapport.nombre_enregistrements} enregistrements -> {rapport.chemin_rapport_detaille}",
            )
            _afficher_rapport(nom_bloc, rapport)
            total_enregistrements += rapport.nombre_enregistrements
            numero_bloc += 1
    except Exception as erreur:
        log.LEVEL_4_ERROR("profiler_corpus", f"echec du profilage par blocs de {source} (bloc {numero_bloc}) : {erreur}")
        raise

    log.PARAMETER_VALUE("total enregistrements", total_enregistrements)
    log.PARAMETER_VALUE("nombre de blocs", numero_bloc)
    log.FINISH_ACTION(
        "profiler_corpus", "main",
        f"{total_enregistrements} enregistrements profiles en {numero_bloc} blocs pour {nom}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Chemin du fichier corpus brut (.csv/.jsonl)")
    parser.add_argument("--nom", required=True, help="Nom du corpus (pour le rapport)")
    parser.add_argument(
        "--bloque",
        type=int,
        default=None,
        help="Profiler par blocs de N enregistrements (evite l'OOM sur un gros corpus ; "
             "un rapport par bloc, correlations/doublons calcules par bloc, pas globalement)",
    )
    arguments = parser.parse_args()

    log.START_ACTION("profiler_corpus", "main", "profilage d'un corpus")
    log.PARAMETER_VALUE("source", arguments.source)
    log.PARAMETER_VALUE("nom", arguments.nom)
    log.PARAMETER_VALUE("bloque", arguments.bloque or "(desactive -- lecture complete)")

    if arguments.bloque:
        _profiler_par_blocs(arguments.source, arguments.nom, arguments.bloque)
        return

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

    _afficher_rapport(arguments.nom, rapport)


if __name__ == "__main__":
    main()
