"""
STEP 03
Point d'entree CLI — Etape 1, action "anonymiser".

Usage :
    uv run python interfaces/cli/anonymiser_dataset.py \
        --dataset data/processed/dataset_pivot.jsonl \
        --sortie data/processed/dataset_pivot_anonymise.jsonl \
        --strategie replace \
        --limite 5000

Design source/sortie separes (08/09/2026, decision du capitaine --
remplace un design precedent qui mutait le pivot en place) :
--dataset (le pivot original) n'est JAMAIS modifie par ce script --
lu seul. Le resultat anonymise est ecrit dans --sortie, un fichier
SEPARE (defaut data/processed/dataset_pivot_anonymise.jsonl). "Deja
anonymise" se determine desormais par la presence de l'identifiant
dans --sortie, pas par le champ `anonymise` du pivot original (qui
reste `False` pour toujours sur le pivot source). Ce design permet
(a) de toujours pouvoir regenerer le pivot sans perdre le texte
original d'exemples deja anonymises, et (b) un controle qualite a
posteriori par simple comparaison --dataset vs --sortie (cf.
`controler_qualite_anonymisation.py`), sans enganche en direct dans la
boucle d'anonymisation.

Option --limite (08/09/2026, decision produit suite a la mesure reelle
d'un temps d'anonymisation complet de ~19h sur le dataset pivot --
cf. docs/02_etape1_donnees/00_couverture_exigences_officielles.md) :
`--limite N` (defaut 5000, l'objectif de la mission) anonymise une
SOUS-ECHANTILLON stratifie par (type_exemple, source) de taille N
parmi les exemples du pivot source dont l'identifiant n'est pas encore
dans --sortie -- le reste attend un appel ulterieur avec un N plus
grand (les exemples deja presents dans --sortie ne sont jamais
retraites). `--limite full` (ou toute valeur >= au nombre restant)
traite tout ce qui reste en une seule fois.

Rapport RGPD automatique (demande du capitaine : que le pipeline
genere lui-meme ses indicateurs RGPD, de facon reproductible, plutot
qu'un calcul manuel ponctuel) : CHAQUE execution ecrit un rapport RGPD
CUMULE (JSON + Markdown, cf. `application/use_cases/rapport_anonymisation.py`)
qui FUSIONNE ses resultats avec ceux des executions precedentes.
Ecrit par defaut sous `data/processed/` (gitignore, comme les fichiers
pivot/anonymise) ; --rapport-json / --rapport-markdown pour surcharger
les chemins.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from chsa_triage.application.use_cases import (
    AnonymiserDatasetUseCase,
    ExecutionAnonymisation,
    RapportAnonymisationCumule,
    formater_rapport_anonymisation_markdown,
    formater_resume_anonymisation_console,
    fusionner_execution,
    rapport_depuis_dict,
    rapport_vers_dict,
)
from chsa_triage.infrastructure.adapters import (
    JsonlDatasetRepository,
    PresidioAnonymiseur,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="anonymiser_dataset")

CHEMIN_SORTIE_DEFAUT = "data/processed/dataset_pivot_anonymise.jsonl"
CHEMIN_RAPPORT_JSON_DEFAUT = "data/processed/rapport_anonymisation_rgpd.json"
CHEMIN_RAPPORT_MARKDOWN_DEFAUT = "data/processed/rapport_anonymisation_rgpd.md"


def _parser_limite(valeur: str) -> int | None:
    """Parse --limite : un entier positif, ou "full"/"illimite" pour tout traiter."""
    if valeur.strip().lower() in ("full", "illimite", "none"):
        return None
    nombre = int(valeur)
    if nombre <= 0:
        raise argparse.ArgumentTypeError("--limite doit etre un entier positif, ou 'full'")
    return nombre


def _charger_rapport_cumule(chemin: Path) -> RapportAnonymisationCumule:
    """Charge le rapport RGPD cumule existant, ou un rapport vide si c'est la premiere execution."""
    if not chemin.exists():
        return RapportAnonymisationCumule()
    with chemin.open("r", encoding="utf-8") as f:
        contenu = f.read().strip()
    if not contenu:
        return RapportAnonymisationCumule()
    return rapport_depuis_dict(json.loads(contenu))


def _sauvegarder_rapport_cumule(
    rapport: RapportAnonymisationCumule,
    chemin_json: Path,
    chemin_markdown: Path,
    total_dataset: int,
) -> None:
    chemin_json.parent.mkdir(parents=True, exist_ok=True)
    with chemin_json.open("w", encoding="utf-8") as f:
        json.dump(rapport_vers_dict(rapport), f, ensure_ascii=False, indent=2)

    chemin_markdown.parent.mkdir(parents=True, exist_ok=True)
    chemin_markdown.write_text(formater_rapport_anonymisation_markdown(rapport, total_dataset), encoding="utf-8")


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",   required=True,     help="Chemin du fichier pivot JSONL ORIGINAL (jamais modifie)")
    parser.add_argument(
        "--sortie",
        default=CHEMIN_SORTIE_DEFAUT,
        help=f"Chemin du fichier JSONL de sortie anonymise, SEPARE du pivot original (defaut {CHEMIN_SORTIE_DEFAUT})",
    )
    parser.add_argument("--strategie", default="replace", choices=["replace", "mask", "redact"])
    parser.add_argument(
        "--limite",
        type=_parser_limite,
        default=5000,
        help="Taille de l'echantillon stratifie (type_exemple, source) a anonymiser parmi les "
             "exemples du pivot source pas encore presents dans --sortie (defaut 5000) ; "
             "'full' pour tout traiter d'un coup",
    )
    parser.add_argument(
        "--rapport-json",
        default=CHEMIN_RAPPORT_JSON_DEFAUT,
        help=f"Chemin du rapport RGPD cumule, JSON (defaut {CHEMIN_RAPPORT_JSON_DEFAUT})",
    )
    parser.add_argument(
        "--rapport-markdown",
        default=CHEMIN_RAPPORT_MARKDOWN_DEFAUT,
        help=f"Chemin du rapport RGPD cumule, Markdown (defaut {CHEMIN_RAPPORT_MARKDOWN_DEFAUT})",
    )
    arguments = parser.parse_args()

    log.START_ACTION("anonymiser_dataset", "main", "anonymisation Presidio du dataset pivot")
    log.PARAMETER_VALUE("dataset (source, jamais modifie)", arguments.dataset)
    log.PARAMETER_VALUE("sortie", arguments.sortie)
    log.PARAMETER_VALUE("strategie", arguments.strategie)
    log.PARAMETER_VALUE("limite", arguments.limite if arguments.limite is not None else "full (aucune limite)")

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS
    # -------------------------------------------------------------------------
    repository_source  = JsonlDatasetRepository(arguments.dataset)
    repository_sortie   = JsonlDatasetRepository(arguments.sortie)
    anonymiseur          = PresidioAnonymiseur(strategie=arguments.strategie)

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(1, "Anonymisation Presidio", "peut prendre du temps selon le nombre d'exemples")
    try:
        cas_usage = AnonymiserDatasetUseCase(
            repository_source=repository_source,
            repository_sortie=repository_sortie,
            anonymiseur=anonymiseur,
            limite=arguments.limite,
        )
        nombre_traites = cas_usage.executer(
            envelopper_iterable=lambda a_traiter: tqdm(a_traiter, desc="Anonymisation Presidio", total=len(a_traiter)),
        )
    except Exception as erreur:
        log.LEVEL_4_ERROR("anonymiser_dataset", f"echec de l'anonymisation de {arguments.dataset} : {erreur}")
        raise

    # -------------------------------------------------------------------------
    # LOG FINAL INFO
    # -------------------------------------------------------------------------
    log.PARAMETER_VALUE("exemples anonymises", nombre_traites)
    log.FINISH_ACTION("anonymiser_dataset", "main", f"{nombre_traites} exemples anonymises (strategie={arguments.strategie})")

    print(f"{nombre_traites} exemples anonymises (strategie={arguments.strategie}, limite={arguments.limite or 'full'}).")
    print(f"Resultat ecrit dans {arguments.sortie} (le pivot original {arguments.dataset} n'a pas ete modifie).")

    # Rapport RGPD cumule (indicateurs generes automatiquement, pas
    # calcules a la main -- fusionne avec les executions precedentes).
    horodatage = datetime.now(timezone.utc).isoformat(timespec="seconds")
    limite_affichee = str(arguments.limite) if arguments.limite is not None else "full"

    log.STEP(2, "Rapport RGPD cumule", "comptage du total reel du dataset pivot source + fusion")
    total_dataset = repository_source.compter()

    execution = ExecutionAnonymisation(
        horodatage=horodatage,
        dataset=arguments.dataset,
        strategie=arguments.strategie,
        limite=limite_affichee,
        graine_aleatoire=cas_usage.graine_aleatoire,
        nombre_traites=nombre_traites,
        statistiques_par_source=cas_usage.statistiques,
    )

    chemin_rapport_json = Path(arguments.rapport_json)
    rapport_cumule = fusionner_execution(_charger_rapport_cumule(chemin_rapport_json), execution)
    _sauvegarder_rapport_cumule(rapport_cumule, chemin_rapport_json, Path(arguments.rapport_markdown), total_dataset)

    print()
    print(formater_resume_anonymisation_console(rapport_cumule, total_dataset))
    log.PARAMETER_VALUE("rapport RGPD JSON", arguments.rapport_json)
    log.PARAMETER_VALUE("rapport RGPD Markdown", arguments.rapport_markdown)


if __name__ == "__main__":
    main()
