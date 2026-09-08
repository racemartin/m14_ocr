"""
STEP 03.1
Point d'entree CLI — Etape 1, action "controler la qualite de
l'anonymisation".

Design fichier-a-fichier (08/09/2026, decision du capitaine --
remplace un enganche en direct dans la boucle d'anonymisation) :
compare le pivot ORIGINAL (--dataset, jamais modifie) au fichier
ANONYMISE (--anonymise, sortie separee de `anonymiser_dataset.py`),
croises par `identifiant`. Puisque le pivot original n'est jamais
mute, ce script peut se relancer a tout moment sur n'importe quelle
tranche deja anonymisee -- y compris retroactivement sur une vague
anonymisee il y a longtemps.

Detection de PII residuelle : regex sans modele (emails, telephones,
URLs, dates, bigrammes capitalises) sur le texte anonymise, puis
seconde opinion spaCy (memes modeles que PresidioAnonymiseur,
fr_core_news_md/en_core_web_sm) pour departager les bigrammes
ambigus -- jamais de LLM/IA generative. Les cas ou ni le regex ni
spaCy ne tranchent sont explicitement marques "pendant_revision_humaine",
jamais une confirmation inventee.

Stratum dedie "sans entite detectee" (--taille-echantillon-sans-entite,
08/09/2026, item explicite du capitaine) : independant du tirage
stratifie (type_exemple, source) ci-dessus, tire un echantillon a part
parmi les couples ou `texte_original == texte_anonymise` sur tous les
champs -- c'est-a-dire ceux ou Presidio n'a RIEN detecte. "Rien
detecte" peut vouloir dire "vraiment aucune PII" ou "Presidio a rate
une PII", d'ou la relecture dediee (30-50 cas par defaut) plutot que
de les laisser se noyer dans le tirage general. Compteurs et section
de rapport (§5) toujours SEPARES du reste.

Usage :
    uv run python interfaces/cli/controler_qualite_anonymisation.py \
        --dataset data/processed/dataset_pivot.jsonl \
        --anonymise data/processed/dataset_pivot_anonymise.jsonl \
        --taille-echantillon 200
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from chsa_triage.application.use_cases import (
    ControlerQualiteAnonymisationUseCase,
    controle_vers_dict,
    formater_rapport_controle_qualite_markdown,
)
from chsa_triage.application.use_cases.uc_03_01_rapport_anonymisation import rapport_depuis_dict
from chsa_triage.infrastructure.adapters import JsonlDatasetRepository
from tools.rafael.log_tool import LogTool

log = LogTool(origin="controler_qualite_anonymisation")

CHEMIN_ANONYMISE_DEFAUT = "data/processed/dataset_pivot_anonymise.jsonl"
CHEMIN_RAPPORT_RGPD_DEFAUT = "data/processed/rapport_anonymisation_rgpd.json"
CHEMIN_RAPPORT_QUALITE_JSON_DEFAUT = "data/processed/rapport_controle_qualite_anonymisation.json"
CHEMIN_RAPPORT_QUALITE_MARKDOWN_DEFAUT = "data/processed/rapport_controle_qualite_anonymisation.md"


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Chemin du fichier pivot JSONL ORIGINAL (jamais modifie)")
    parser.add_argument(
        "--anonymise",
        default=CHEMIN_ANONYMISE_DEFAUT,
        help=f"Chemin du fichier JSONL anonymise a controler (defaut {CHEMIN_ANONYMISE_DEFAUT})",
    )
    parser.add_argument(
        "--taille-echantillon",
        type=int,
        default=200,
        help="Taille de l'echantillon stratifie (type_exemple, source) compare parmi les exemples "
             "anonymises disponibles (defaut 200)",
    )
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument(
        "--taille-echantillon-sans-entite",
        type=int,
        default=40,
        help="Taille du stratum DEDIE aux exemples ou aucune entite n'a ete detectee "
             "(texte_original == texte_anonymise sur tous les champs) -- independant de "
             "--taille-echantillon ci-dessus. Le capitaine demande explicitement 30-50 cas "
             "relus a la main/seconde opinion spaCy (defaut 40)",
    )
    parser.add_argument("--graine-sans-entite", type=int, default=43)
    parser.add_argument(
        "--max-exemples-par-source",
        type=int,
        default=10,
        help="Nombre maximum d'exemples reels (original->anonymise) conserves PAR SOURCE dans le rapport (defaut 10)",
    )
    parser.add_argument(
        "--max-faux-positifs-par-source",
        type=int,
        default=10,
        help="Nombre maximum de candidats de faux positifs de masquage conserves PAR SOURCE (defaut 10)",
    )
    parser.add_argument(
        "--rapport-rgpd-json",
        default=CHEMIN_RAPPORT_RGPD_DEFAUT,
        help=f"Rapport RGPD cumule (Partie 1) a reutiliser pour les compteurs d'entites par type "
             f"(defaut {CHEMIN_RAPPORT_RGPD_DEFAUT})",
    )
    parser.add_argument("--rapport-json", default=CHEMIN_RAPPORT_QUALITE_JSON_DEFAUT)
    parser.add_argument("--rapport-markdown", default=CHEMIN_RAPPORT_QUALITE_MARKDOWN_DEFAUT)
    arguments = parser.parse_args()

    log.START_ACTION("controler_qualite_anonymisation", "main", "controle qualite par comparaison de fichiers")
    log.PARAMETER_VALUE("dataset (original)", arguments.dataset)
    log.PARAMETER_VALUE("anonymise", arguments.anonymise)
    log.PARAMETER_VALUE("taille-echantillon", arguments.taille_echantillon)
    log.PARAMETER_VALUE("taille-echantillon-sans-entite", arguments.taille_echantillon_sans_entite)

    # -------------------------------------------------------------------------
    # PREPARE ADAPTERS
    # -------------------------------------------------------------------------
    repository_original   = JsonlDatasetRepository(arguments.dataset)
    repository_anonymise  = JsonlDatasetRepository(arguments.anonymise)

    from chsa_triage.infrastructure.adapters import SpacyVerificateurEntitesNommees

    verificateur = SpacyVerificateurEntitesNommees()

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    log.STEP(1, "Comparaison original/anonymise", "regex + seconde opinion spaCy")
    try:
        cas_usage = ControlerQualiteAnonymisationUseCase(
            repository_original            = repository_original,
            repository_anonymise           = repository_anonymise,
            verificateur_entites           = verificateur,
            taille_echantillon             = arguments.taille_echantillon,
            graine_aleatoire               = arguments.graine,
            taille_echantillon_sans_entite = arguments.taille_echantillon_sans_entite,
            graine_aleatoire_sans_entite   = arguments.graine_sans_entite,
            max_exemples_par_source        = arguments.max_exemples_par_source,
            max_faux_positifs_par_source   = arguments.max_faux_positifs_par_source,
        )
        controle = cas_usage.executer()
    except Exception as erreur:
        log.LEVEL_4_ERROR(
            "controler_qualite_anonymisation",
            f"echec du controle qualite pour {arguments.dataset} / {arguments.anonymise} : {erreur}",
        )
        raise

    # -------------------------------------------------------------------------
    # LOG FINAL INFO
    # -------------------------------------------------------------------------
    if cas_usage.nombre_introuvables_dans_original:
        log.LEVEL_5_WARNING(
            "controler_qualite_anonymisation",
            f"{cas_usage.nombre_introuvables_dans_original} identifiant(s) present(s) dans "
            f"{arguments.anonymise} introuvable(s) dans {arguments.dataset} -- fichiers incoherents ?",
        )

    total_disponible = repository_anonymise.compter()

    # Charge le rapport RGPD cumule (Partie 1) pour reutiliser les
    # compteurs d'entites par type -- pas de recalcul ici.
    chemin_rgpd = Path(arguments.rapport_rgpd_json)
    statistiques_cumulees = {}
    if chemin_rgpd.exists() and chemin_rgpd.stat().st_size > 0:
        rapport_rgpd = rapport_depuis_dict(json.loads(chemin_rgpd.read_text(encoding="utf-8")))
        statistiques_cumulees = rapport_rgpd.statistiques_cumulees
    else:
        log.LEVEL_5_WARNING(
            "controler_qualite_anonymisation",
            f"rapport RGPD cumule introuvable ({arguments.rapport_rgpd_json}) -- section 1 du rapport "
            "de controle qualite restera vide, lancer anonymiser_dataset.py au moins une fois avant",
        )

    horodatage = datetime.now(timezone.utc).isoformat(timespec="seconds")

    chemin_json     = Path(arguments.rapport_json)
    chemin_markdown = Path(arguments.rapport_markdown)
    chemin_json.parent.mkdir(parents=True, exist_ok=True)
    chemin_markdown.parent.mkdir(parents=True, exist_ok=True)

    with chemin_json.open("w", encoding="utf-8") as f:
        json.dump(
            controle_vers_dict(controle, horodatage, arguments.dataset, arguments.anonymise),
            f,
            ensure_ascii=False,
            indent=2,
        )
    chemin_markdown.write_text(
        formater_rapport_controle_qualite_markdown(
            controle,
            horodatage,
            arguments.dataset,
            arguments.anonymise,
            arguments.taille_echantillon,
            total_disponible,
            statistiques_cumulees,
        ),
        encoding="utf-8",
    )

    log.PARAMETER_VALUE("exemples compares", controle.nombre_exemples_observes)
    log.PARAMETER_VALUE("candidats PII residuelle", len(controle.candidats_pii))
    log.PARAMETER_VALUE("candidats faux positifs", len(controle.candidats_faux_positifs))
    log.PARAMETER_VALUE("stratum sans entite : disponibles", controle.nombre_disponibles_sans_entite)
    log.PARAMETER_VALUE("stratum sans entite : relus", controle.nombre_exemples_sans_entite_observes)
    log.PARAMETER_VALUE("stratum sans entite : candidats PII", len(controle.candidats_pii_sans_entite))
    log.FINISH_ACTION(
        "controler_qualite_anonymisation", "main", f"rapport ecrit dans {chemin_json} / {chemin_markdown}"
    )

    print(f"{controle.nombre_exemples_observes} exemples compares (original vs anonymise).")
    print(f"Rapport de controle qualite ecrit : {chemin_json} / {chemin_markdown}")


if __name__ == "__main__":
    main()
