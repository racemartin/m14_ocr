"""
STEP 03.2
Point d'entree CLI — Etape 1, action "reviser humainement les
candidats de PII residuelle en attente" (exigence NF2 du cahier des
charges : anonymisation validee MANUELLEMENT, 0 PII residuelle sur
echantillon de controle).

Avant ce script, `CandidatPiiResiduelle.verdict == VERDICT_REVISION_HUMAINE`
(cf. `controler_qualite_anonymisation.py`) restait un cul-de-sac : ni
le regex ni la seconde opinion spaCy ne tranchent, et aucune decision
de personne n'etait jamais persistee. Ce script ferme cet ecart avec
deux modes :

- `verify` : recalcule (replay deterministe, cf.
  `uc_03_03_reviser_pii_residuelle.ReviserPiiResiduelleUseCase`) TOUS
  les candidats REVISION_HUMAINE sur TOUT ce qui a deja ete echantillonne
  par `controler_qualite_anonymisation.py` (les deux strates), exclut
  ceux ayant deja une decision, et pour chaque candidat restant,
  demande interactivement d'accepter/rejeter/sauter ; la decision est
  persistee IMMEDIATEMENT apres chaque reponse (pas en fin de lot), donc
  fermer le terminal a mi-parcours ne perd jamais le travail deja fait.
- `modify` : localise une decision deja prise par `--identifiant`
  (optionnellement `--champ`) et permet de la corriger sans repasser
  par toute la liste en attente.

Usage :
    uv run python interfaces/cli/reviser_pii_residuelle.py verify \
        --dataset data/processed/dataset_pivot.jsonl \
        --anonymise data/processed/dataset_pivot_anonymise.jsonl

    uv run python interfaces/cli/reviser_pii_residuelle.py modify --identifiant chsa-xxxxxxxx
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone

from chsa_triage.application.use_cases.uc_03_02_controler_qualite_anonymisation import JETON_MASQUE_DEFAUT
from chsa_triage.application.use_cases.uc_03_03_reviser_pii_residuelle import (
    CandidatARevoir,
    ReviserPiiResiduelleUseCase,
    texte_original_et_anonymise,
)
from chsa_triage.domain.model import DECISION_ACCEPTE, DECISION_REJETE
from chsa_triage.infrastructure.adapters import (
    JsonlDatasetRepository,
    JsonlDecisionsRevisionHumaine,
    JsonlRegistreEchantillonsControleQualite,
)
from tools.rafael.log_tool import LogTool

log = LogTool(origin="reviser_pii_residuelle")

CHEMIN_ANONYMISE_DEFAUT             = "data/processed/dataset_pivot_anonymise.jsonl"
CHEMIN_REGISTRE_ECHANTILLONS_DEFAUT = "data/processed/controle_qualite_identifiants_echantillonnes.jsonl"
CHEMIN_DECISIONS_DEFAUT             = "data/processed/decisions_revision_humaine.jsonl"


def _horodatage() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _afficher_candidat(candidat: CandidatARevoir, indice: int, total: int) -> None:
    print()
    print(f"[{indice}/{total}] identifiant={candidat.cle.identifiant}")
    print(
        f"  source={candidat.source_corpus}  champ={candidat.cle.champ}  "
        f"type_motif={candidat.cle.type_motif or '(faux positif de masquage, pas de regex typee)'}"
    )
    print(f"  passage : {candidat.passage!r}")


def _demander_action() -> str:
    while True:
        reponse = input("  [a]ccepter / [r]ejeter / [s]auter / [v]oir texte complet / [q]uitter : ").strip().lower()
        if reponse in ("a", "r", "s", "v", "q"):
            return reponse
        print("  reponse non reconnue, reessayer.")


def _afficher_texte_complet(cas_usage: ReviserPiiResiduelleUseCase, candidat: CandidatARevoir) -> None:
    original = cas_usage.repository_original.trouver_par_id(candidat.cle.identifiant)
    anonymise = cas_usage.repository_anonymise.trouver_par_id(candidat.cle.identifiant)
    paire = texte_original_et_anonymise(original, anonymise, candidat.cle.champ) if original and anonymise else None
    if paire is None:
        print("  (texte complet introuvable ; identifiant absent d'un des deux fichiers ?)")
        return
    texte_original, texte_anonymise = paire
    print(f"  --- {candidat.cle.champ} (original) ---\n  {texte_original}")
    print(f"  --- {candidat.cle.champ} (anonymise) ---\n  {texte_anonymise}")


# ##############################################################################
# _mode_verify
# ##############################################################################
def _mode_verify(arguments: argparse.Namespace) -> None:
    from chsa_triage.infrastructure.adapters import SpacyVerificateurEntitesNommees

    log.START_ACTION("reviser_pii_residuelle", "verify", "revision humaine interactive des candidats en attente")

    cas_usage = ReviserPiiResiduelleUseCase(
        repository_original=JsonlDatasetRepository(arguments.dataset),
        repository_anonymise=JsonlDatasetRepository(arguments.anonymise),
        verificateur_entites=SpacyVerificateurEntitesNommees(),
        registre_echantillons=JsonlRegistreEchantillonsControleQualite(arguments.registre_echantillons),
        decisions=JsonlDecisionsRevisionHumaine(arguments.decisions),
        jeton_masque=arguments.jeton_masque,
    )

    log.STEP(1, "Recalcul des candidats en attente", "replay sur tous les identifiants deja echantillonnes")
    candidats = cas_usage.candidats_en_attente()
    log.PARAMETER_VALUE("candidats en attente de decision humaine", len(candidats))

    if not candidats:
        print("Aucun candidat en attente de revision humaine.")
        log.FINISH_ACTION("reviser_pii_residuelle", "verify", "rien a reviser")
        return

    total = len(candidats)
    traites = 0
    log.STEP(2, "Revision interactive", f"{total} candidat(s), persistance immediate apres chaque decision")
    for indice, candidat in enumerate(candidats, start=1):
        _afficher_candidat(candidat, indice, total)
        while True:
            action = _demander_action()
            if action == "v":
                _afficher_texte_complet(cas_usage, candidat)
                continue
            if action == "s":
                break
            if action == "q":
                print(f"Arret demande : {traites}/{total} decisions prises cette session.")
                log.FINISH_ACTION(
                    "reviser_pii_residuelle", "verify", f"{traites}/{total} decisions prises, arret demande"
                )
                return
            decision = DECISION_ACCEPTE if action == "a" else DECISION_REJETE
            note = input("  note (optionnel) : ").strip()
            cas_usage.enregistrer_decision(candidat, decision, _horodatage(), note)
            traites += 1
            print(f"  -> decision '{decision}' enregistree dans {arguments.decisions}.")
            break

    print(f"{traites}/{total} decisions prises.")
    log.PARAMETER_VALUE("decisions prises", traites)
    log.FINISH_ACTION("reviser_pii_residuelle", "verify", f"{traites}/{total} decisions prises")


# ##############################################################################
# _mode_modify
# ##############################################################################
def _mode_modify(arguments: argparse.Namespace) -> None:
    log.START_ACTION("reviser_pii_residuelle", "modify", f"correction d'une decision pour {arguments.identifiant}")

    decisions = JsonlDecisionsRevisionHumaine(arguments.decisions)
    candidates = [d for d in decisions.toutes() if d.cle.identifiant == arguments.identifiant]
    if arguments.champ:
        candidates = [d for d in candidates if d.cle.champ == arguments.champ]

    if not candidates:
        portee = f"identifiant={arguments.identifiant!r}" + (f", champ={arguments.champ!r}" if arguments.champ else "")
        print(f"Aucune decision trouvee pour {portee}.")
        log.FINISH_ACTION("reviser_pii_residuelle", "modify", "aucune decision trouvee")
        return

    for indice, d in enumerate(candidates, start=1):
        print(
            f"[{indice}] champ={d.cle.champ}  type_motif={d.cle.type_motif or '(faux positif de masquage)'}  "
            f"decision actuelle={d.decision}  passage={d.passage!r}"
        )

    while True:
        choix = input(f"Quelle decision corriger ? [1-{len(candidates)}] : ").strip()
        if choix.isdigit() and 1 <= int(choix) <= len(candidates):
            break
        print("  choix non reconnu, reessayer.")
    a_corriger = candidates[int(choix) - 1]

    while True:
        nouvelle = input("Nouvelle decision [a=accepter / r=rejeter] : ").strip().lower()
        if nouvelle in ("a", "r"):
            break
        print("  reponse non reconnue, reessayer.")
    nouvelle_decision = DECISION_ACCEPTE if nouvelle == "a" else DECISION_REJETE

    nouvelle_note = input(f"Nouvelle note (vide = garder {a_corriger.note!r}) : ").strip()
    note_finale = nouvelle_note or a_corriger.note

    corrigee = replace(a_corriger, decision=nouvelle_decision, horodatage=_horodatage(), note=note_finale)
    decisions.enregistrer(corrigee)

    print(f"Decision mise a jour : {a_corriger.decision!r} -> {nouvelle_decision!r}.")
    log.FINISH_ACTION(
        "reviser_pii_residuelle", "modify", f"{a_corriger.cle} : {a_corriger.decision} -> {nouvelle_decision}"
    )


def main() -> None:
    # -------------------------------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sous_parseurs = parser.add_subparsers(dest="mode", required=True)

    parseur_verify = sous_parseurs.add_parser(
        "verify", help="Revue interactive des candidats en attente de decision humaine"
    )
    parseur_verify.add_argument("--dataset", required=True, help="Chemin du fichier pivot JSONL ORIGINAL")
    parseur_verify.add_argument("--anonymise", default=CHEMIN_ANONYMISE_DEFAUT)
    parseur_verify.add_argument("--registre-echantillons", default=CHEMIN_REGISTRE_ECHANTILLONS_DEFAUT)
    parseur_verify.add_argument("--decisions", default=CHEMIN_DECISIONS_DEFAUT)
    parseur_verify.add_argument("--jeton-masque", default=JETON_MASQUE_DEFAUT)

    parseur_modify = sous_parseurs.add_parser("modify", help="Corriger une decision humaine deja prise")
    parseur_modify.add_argument("--identifiant", required=True)
    parseur_modify.add_argument("--champ", default=None, help="Restreint la recherche a un champ precis (optionnel)")
    parseur_modify.add_argument("--decisions", default=CHEMIN_DECISIONS_DEFAUT)

    arguments = parser.parse_args()

    # -------------------------------------------------------------------------
    # USE CASE EXECUTE
    # -------------------------------------------------------------------------
    if arguments.mode == "verify":
        _mode_verify(arguments)
    else:
        _mode_modify(arguments)


if __name__ == "__main__":
    main()
