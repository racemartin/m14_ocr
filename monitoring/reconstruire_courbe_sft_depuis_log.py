"""
Reconstruit a posteriori la courbe de metriques (perte_train,
perte_validation, norme_gradient, etc.) d'un run SFT-LoRA reel deja
termine, a partir de son log brut sauvegarde, quand ce run n'a jamais
ecrit vers un backend `SuiviExperimentation` durable (ex.
`suivi.backend: mlflow` avec un SQLite local dans un conteneur HF Jobs
ephemere, deja detruit).

Cas d'usage concret documente : le run du 16/09/2026 (job HF Jobs
`6aaab9a95527934177eeaac8`, GPU L4, verdict "saine", poids publies dans
`mombasstic/chsa-triage-sft-lora`) avait `recipes/sft_qwen3_lora.yaml::
suivi.backend` a `mlflow` au lieu de `hf_dataset` : sa courbe n'a jamais
atteint un depot durable. Ce script est un outil d'ANALYSE DE LOG A
POSTERIORI, pas un composant du pipeline d'entrainement ; il ne touche
ni a la recette ni a `training/E2_04_sft_train.py`. Voir AGENTS.md pour
le contexte complet et la correction (dans une autre tache) du defaut
de configuration qui a cause la perte de cette courbe.

Produit un `metriques.jsonl` au format LONG attendu par
`HfDatasetSuiviExperimentation`/`monitoring/hf_dataset_runs.py` (une
ligne JSON par metrique-etape : `{"etape", "nom", "valeur",
"horodatage"}`), et un `parametres.json` avec les hyperparametres reels
de la recette au commit utilise par le job (lu via `git show`, jamais
en modifiant le fichier courant) et des metadonnees marquant
explicitement qu'il s'agit d'une reconstruction.

Horodatage (`horodatage`, epoch secondes) : RECONSTRUIT, pas mesure
ligne a ligne (le job n'imprime pas d'horodatage a chaque pas
d'entrainement). Methode : les barres de progression tqdm de
`transformers`/`trl` impriment un temps ecoule REEL depuis le debut de
la boucle a chaque pas (ex. "150/342 [08:23<10:12, 3.35s/it]") ; ce
temps ecoule est ancre sur un horodatage REEL extrait du nom du
repertoire de checkpoint que `TrlSftEntraineurAdapter` cree juste avant
`trainer.train()` (`run-YYYYmmddTHHMMSSZ`, cf.
`_horodatage_nom_run()` dans `trl_sft_entraineur.py`). Ce ne sont donc
pas des horodatages synthetiques/invente : ils sont deduits de deux
sources reelles imprimees par le job lui-meme. Coherence verifiee sur
ce run precis : ecart tqdm au dernier pas (~1182s) vs. `train_runtime`
rapporte par trl en fin de run (1182.9999s) vs. ecart entre l'horodatage
d'ancrage et le message de fin de run loggue par `E2_04_sft_train`
(~19min50s) concordent a quelques secondes pres.

Usage :
    uv run python monitoring/reconstruire_courbe_sft_depuis_log.py \\
        --log /chemin/vers/le/log.log \\
        --nom-run sft-lora-16092026-reconstruit \\
        --publier
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from calendar import timegm
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

JOB_ID_PAR_DEFAUT = "6aaab9a95527934177eeaac8"
REPO_METRIQUES_PAR_DEFAUT = "mombasstic/chsa-triage-sft-metrics"
NOM_RUN_PAR_DEFAUT = "sft-lora-16092026-reconstruit"

# Une barre tqdm de boucle d'entrainement ("X/Y [MM:SS<remaining, R.RRs/it]") ;
# `[^,\n]+` interdit explicitement le saut de ligne pour ne jamais
# capturer au-dela de la ligne courante.
RE_BARRE_ENTRAINEMENT = re.compile(
    r"(?P<etape>\d+)/(?P<total>\d+)\s*\[(?P<minutes>\d+):(?P<secondes>\d+)<[^,\n]+,\s*[\d.]+s/it\]"
)
# Horodatage d'ancrage : nom du repertoire de checkpoint cree juste
# avant `trainer.train()` (cf. `_horodatage_nom_run` dans
# trl_sft_entraineur.py).
RE_HORODATAGE_CHECKPOINT = re.compile(r"run-(?P<horodatage>\d{8}T\d{6})Z")
RE_COMMIT_GIT = re.compile(r"chsa-triage @ git\+https://github\.com/\S+@(?P<sha>[0-9a-f]{40})")

# Mappe les cles du dictionnaire imprime par `transformers`/`trl` vers
# les noms de metriques deja utilises par `E2_01_uc_entrainer_sft.py`
# (pour que le dashboard les reconnaisse comme un run reel), plus
# quelques metriques additionnelles (noms libres, le format LONG les
# accepte tous).
MAPPING_NOMS_METRIQUES = {
    "loss": "perte_train",
    "grad_norm": "norme_gradient",
    "learning_rate": "learning_rate",
    "mean_token_accuracy": "mean_token_accuracy",
    "entropy": "entropy",
    "eval_loss": "perte_validation",
    "eval_mean_token_accuracy": "eval_mean_token_accuracy",
    "eval_entropy": "eval_entropy",
}


@dataclass(frozen=True, slots=True)
class PointMetrique:
    """Une ligne du futur `metriques.jsonl` format LONG."""

    etape       : int
    nom           : str
    valeur          : float
    horodatage        : float


def extraire_barres_entrainement(texte_log: str) -> tuple[int, dict[int, int]]:
    """
    Determine le nombre total de pas d'entrainement (le plus grand
    denominateur observe parmi les barres au format "X/Y [...s/it]" :
    les sous-barres d'evaluation ont un denominateur bien plus petit,
    ex. "0/8") et associe chaque numero de pas a son temps ecoule reel
    (en secondes depuis le debut de la boucle, tel qu'imprime par tqdm).
    """
    correspondances = list(RE_BARRE_ENTRAINEMENT.finditer(texte_log))
    if not correspondances:
        raise ValueError("aucune barre de progression d'entrainement trouvee dans le log")
    total_pas = max(int(m.group("total")) for m in correspondances)
    elapsed_par_etape = {
        int(m.group("etape")): int(m.group("minutes")) * 60 + int(m.group("secondes"))
        for m in correspondances
        if int(m.group("total")) == total_pas
    }
    return total_pas, elapsed_par_etape


def extraire_horodatage_ancrage(texte_log: str) -> float:
    """Horodatage reel (epoch UTC) du debut de la boucle d'entrainement (pas 0)."""
    correspondance = RE_HORODATAGE_CHECKPOINT.search(texte_log)
    if correspondance is None:
        raise ValueError("aucun horodatage de checkpoint (run-YYYYmmddTHHMMSSZ) trouve dans le log")
    horodatage = datetime.strptime(correspondance.group("horodatage"), "%Y%m%dT%H%M%S")
    return float(timegm(horodatage.timetuple()))


def _parser_ligne_dictionnaire(ligne: str) -> dict | None:
    """
    Une ligne du log est retenue seulement si c'est un dictionnaire
    Python valide portant `loss` ou `eval_loss` (le format litteral
    imprime par `transformers`/`trl` a chaque pas de logging) ; toutes
    les autres lignes (barres tqdm, messages applicatifs, warnings,
    telechargements de paquets) sont silencieusement ignorees.
    """
    ligne = ligne.strip()
    if not (ligne.startswith("{'loss'") or ligne.startswith("{'eval_loss'")):
        return None
    try:
        objet = ast.literal_eval(ligne)
    except (ValueError, SyntaxError):
        return None
    return objet if isinstance(objet, dict) else None


def extraire_points_metriques(texte_log: str) -> list[PointMetrique]:
    """
    Parse le log complet et reconstruit la serie temporelle complete au
    format LONG. L'etape d'un point d'entrainement est le dernier
    numero de pas tqdm vu avant sa ligne ; l'etape d'un point
    d'evaluation (logue a la fin de chaque epoque, sans pas tqdm
    d'entrainement associe directement) est deduite de son `epoch`
    (`epoch * pas_par_epoque`, `pas_par_epoque` derive du nombre total
    de pas et du plus grand `epoch` observe dans le log).
    """
    total_pas, elapsed_par_etape = extraire_barres_entrainement(texte_log)
    horodatage_t0 = extraire_horodatage_ancrage(texte_log)

    objets_avec_dernier_pas: list[tuple[int, dict]] = []
    dernier_pas_entrainement = 0
    for ligne in texte_log.splitlines():
        correspondance_barre = RE_BARRE_ENTRAINEMENT.search(ligne)
        if correspondance_barre and int(correspondance_barre.group("total")) == total_pas:
            dernier_pas_entrainement = int(correspondance_barre.group("etape"))
            continue
        objet = _parser_ligne_dictionnaire(ligne)
        if objet is not None:
            objets_avec_dernier_pas.append((dernier_pas_entrainement, objet))

    if not objets_avec_dernier_pas:
        raise ValueError("aucune ligne de metrique ('loss'/'eval_loss') trouvee dans le log")

    nombre_epoques = round(max(float(objet.get("epoch", 0.0)) for _, objet in objets_avec_dernier_pas))
    if nombre_epoques <= 0 or total_pas % nombre_epoques != 0:
        raise ValueError(f"nombre de pas total ({total_pas}) non divisible par le nombre d'epoques deduit ({nombre_epoques})")
    pas_par_epoque = total_pas // nombre_epoques

    points: list[PointMetrique] = []
    for dernier_pas, objet in objets_avec_dernier_pas:
        est_evaluation = "eval_loss" in objet
        etape = round(float(objet["epoch"]) * pas_par_epoque) if est_evaluation else dernier_pas
        elapsed = elapsed_par_etape.get(etape)
        if elapsed is None:
            etape_proche = min(elapsed_par_etape, key=lambda e: abs(e - etape))
            elapsed = elapsed_par_etape[etape_proche]
        horodatage = horodatage_t0 + elapsed
        for cle_source, nom_cible in MAPPING_NOMS_METRIQUES.items():
            if cle_source in objet:
                points.append(PointMetrique(etape=etape, nom=nom_cible, valeur=float(objet[cle_source]), horodatage=horodatage))
    return points


def points_vers_jsonl(points: list[PointMetrique]) -> str:
    """Serialise les points au format LONG attendu par `HfDatasetSuiviExperimentation` (une ligne JSON par point)."""
    lignes = [
        json.dumps({"etape": p.etape, "nom": p.nom, "valeur": p.valeur, "horodatage": p.horodatage})
        for p in points
    ]
    return "\n".join(lignes) + ("\n" if lignes else "")


def extraire_metadonnees_log(texte_log: str) -> dict:
    """
    Metadonnees factuelles extraites directement du texte du log
    (comptages, verdict, chemins, duree reelle rapportee par trl) : pas
    de valeur inventee, seulement ce qui est effectivement imprime.
    """
    metadonnees: dict = {}

    def _chercher(motif: str) -> str | None:
        correspondance = re.search(motif, texte_log)
        return correspondance.group(1) if correspondance else None

    exemples_train = _chercher(r"exemples train formates\.+:\s*(\d+)")
    exemples_validation = _chercher(r"exemples validation formates\.*:\s*(\d+)")
    verdict = _chercher(r"verdict retenu\.+:\s*(\S+)")
    nombre_essais = _chercher(r"nombre d'essais\.+:\s*(\d+)")
    checkpoint_local = _chercher(r"checkpoint retenu\.+:\s*(\S+)")
    checkpoint_hf_repo = _chercher(r"Publication des poids .* sur HF Hub \((\S+)\)")

    if exemples_train is not None:
        metadonnees["nombre_exemples_train"] = int(exemples_train)
    if exemples_validation is not None:
        metadonnees["nombre_exemples_validation"] = int(exemples_validation)
    if verdict is not None:
        metadonnees["verdict_convergence"] = verdict
    if nombre_essais is not None:
        metadonnees["nombre_essais"] = int(nombre_essais)
    if checkpoint_local is not None:
        metadonnees["checkpoint_local"] = checkpoint_local
    if checkpoint_hf_repo is not None:
        metadonnees["checkpoint_hf_repo"] = checkpoint_hf_repo

    correspondance_resume = re.search(r"\{'train_runtime':[^}]*\}", texte_log)
    if correspondance_resume:
        resume = ast.literal_eval(correspondance_resume.group(0))
        metadonnees["train_runtime_s"] = resume.get("train_runtime")
        metadonnees["train_loss_final"] = resume.get("train_loss")

    total_pas, _ = extraire_barres_entrainement(texte_log)
    metadonnees["nombre_pas_total"] = total_pas

    return metadonnees


def extraire_recette_au_commit(texte_log: str, chemin_recette: str = "recipes/sft_qwen3_lora.yaml") -> tuple[dict, str | None]:
    """
    Recupere le contenu de la recette TEL QU'IL ETAIT au commit
    reellement utilise par le job (extrait du log, ex. "Building
    chsa-triage @ git+...@571abc1..."), via `git show`, jamais en
    lisant le fichier courant du worktree (qui peut avoir change depuis,
    notamment via une autre tache en parallele). Retourne `({}, None)`
    si le commit n'a pas pu etre identifie ou que `git show` echoue
    (ex. commit non present localement).
    """
    correspondance = RE_COMMIT_GIT.search(texte_log)
    if correspondance is None:
        return {}, None
    sha = correspondance.group("sha")
    try:
        import yaml

        resultat = subprocess.run(
            ["git", "show", f"{sha}:{chemin_recette}"],
            capture_output=True, text=True, check=True,
        )
        return yaml.safe_load(resultat.stdout), sha
    except (subprocess.CalledProcessError, ImportError):
        return {}, sha


def construire_parametres(texte_log: str, job_id: str) -> dict:
    """Assemble `parametres.json` : recette reelle au commit du job + metadonnees factuelles + marqueurs de reconstruction."""
    recette, commit_sha = extraire_recette_au_commit(texte_log)
    parametres: dict = {
        "reconstruit_depuis_log": True,
        "job_id": job_id,
        "horodatage_note": (
            "les horodatages de metriques.jsonl sont RECONSTRUITS a partir des ecarts "
            "de temps ecoule reels imprimes par les barres tqdm du job, ancres sur "
            "l'horodatage reel du nom de repertoire de checkpoint (run-YYYYmmddTHHMMSSZ) ; "
            "ce ne sont pas des horodatages synthetiques inventes, mais ils ne sont pas non "
            "plus des horodatages mesures ligne a ligne par le job lui-meme"
        ),
    }
    if commit_sha is not None:
        parametres["commit_git"] = commit_sha
    if recette:
        parametres["recette"] = recette
    parametres.update(extraire_metadonnees_log(texte_log))
    return parametres


def publier_run(repo_id: str, nom_run: str, repertoire_local: Path) -> None:
    """Publie `<nom_run>/metriques.jsonl` et `<nom_run>/parametres.json` vers un depot dataset HF (reseau reel)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    for nom_fichier in ("metriques.jsonl", "parametres.json"):
        api.upload_file(
            path_or_fileobj=str(repertoire_local / nom_fichier),
            path_in_repo=f"{nom_run}/{nom_fichier}",
            repo_id=repo_id,
            repo_type="dataset",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="Chemin du fichier log brut sauvegarde du job")
    parser.add_argument("--nom-run", default=NOM_RUN_PAR_DEFAUT, help=f"Nom du run publie (defaut {NOM_RUN_PAR_DEFAUT})")
    parser.add_argument("--job-id", default=JOB_ID_PAR_DEFAUT, help=f"Identifiant du job HF Jobs source (defaut {JOB_ID_PAR_DEFAUT})")
    parser.add_argument("--repo-metriques", default=REPO_METRIQUES_PAR_DEFAUT, help=f"Depot dataset HF cible (defaut {REPO_METRIQUES_PAR_DEFAUT})")
    parser.add_argument("--sortie-dir", default=None, help="Repertoire local de sortie (defaut : data/demos/<nom-run>/)")
    parser.add_argument("--publier", action="store_true", help="Publie reellement vers le depot HF (reseau reel) apres ecriture locale")
    arguments = parser.parse_args()

    texte_log = Path(arguments.log).read_text(encoding="utf-8")
    points = extraire_points_metriques(texte_log)
    parametres = construire_parametres(texte_log, arguments.job_id)

    repertoire_sortie = Path(arguments.sortie_dir) if arguments.sortie_dir else Path("data/demos") / arguments.nom_run
    repertoire_sortie.mkdir(parents=True, exist_ok=True)
    (repertoire_sortie / "metriques.jsonl").write_text(points_vers_jsonl(points), encoding="utf-8")
    (repertoire_sortie / "parametres.json").write_text(json.dumps(parametres, indent=2, default=str), encoding="utf-8")

    print(f"{len(points)} points de metrique ecrits dans {repertoire_sortie}/metriques.jsonl")
    print(f"parametres ecrits dans {repertoire_sortie}/parametres.json")

    if arguments.publier:
        publier_run(arguments.repo_metriques, arguments.nom_run, repertoire_sortie)
        print(f"publie : {arguments.repo_metriques}/{arguments.nom_run}/{{metriques.jsonl,parametres.json}}")
    else:
        print("Non publie (passer --publier pour envoyer reellement au Hub). Commande equivalente manuelle :")
        print(
            f"  hf upload {arguments.repo_metriques} {repertoire_sortie}/metriques.jsonl "
            f"{arguments.nom_run}/metriques.jsonl --repo-type dataset"
        )
        print(
            f"  hf upload {arguments.repo_metriques} {repertoire_sortie}/parametres.json "
            f"{arguments.nom_run}/parametres.json --repo-type dataset"
        )


if __name__ == "__main__":
    main()
