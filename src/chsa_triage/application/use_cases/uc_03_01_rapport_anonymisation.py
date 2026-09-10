"""
Rapport RGPD cumule genere automatiquement a chaque execution de
`anonymiser_dataset.py` ; remplace le calcul manuel ponctuel fait
pour `docs/02_etape1_donnees/01_rapport_rgpd.md`.

Le processus d'anonymisation est incremental (`--limite`, plusieurs
executions successives dans le temps, cf. `AnonymiserDatasetUseCase`).
Ce module est donc concu pour FUSIONNER, pas ecraser : chaque
execution ajoute sa propre contribution tracable (horodatage,
parametres) et ses compteurs sont additionnes aux compteurs cumules
deja persistes, pour que le rapport refleve TOUJOURS l'etat reel de
tout ce qui a ete anonymise jusqu'a present, pas seulement la derniere
tanche.

Aucune I/O ici (ni lecture/ecriture de fichier JSON, ni impression) :
uniquement des structures de donnees et des fonctions pures,
facilement testables. La lecture/ecriture du fichier JSON cumule et
l'affichage sont a la charge de l'appelant (le CLI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chsa_triage.application.use_cases.uc_03_00_anonymiser_dataset import StatistiquesSource


@dataclass(slots=True)
class ExecutionAnonymisation:
    """Trace d'UNE execution de `anonymiser_dataset.py` ayant contribue au rapport."""

    horodatage               : str  # ISO 8601
    dataset                  : str
    strategie                : str
    limite                   : str  # valeur de --limite telle qu'affichee ("5000", "full", ...)
    graine_aleatoire         : int
    nombre_traites            : int
    # Contribution de CETTE execution uniquement (pas cumulee) ; c'est
    # ce qui permet de reconstituer "qui a apporte quoi" a la lecture.
    statistiques_par_source     : dict[str, StatistiquesSource]


@dataclass(slots=True)
class RapportAnonymisationCumule:
    """Etat cumule de toutes les executions d'anonymisation ayant contribue au rapport."""

    executions             : list[ExecutionAnonymisation] = field(default_factory=list)
    statistiques_cumulees  : dict[str, StatistiquesSource] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Serialisation JSON (dict <-> dataclasses)
# ----------------------------------------------------------------------


def _statistiques_vers_dict(stats: StatistiquesSource) -> dict:
    return {
        "registres_traites"    : stats.registres_traites,
        "registres_avec_entite": stats.registres_avec_entite,
        "entites_par_type"     : dict(stats.entites_par_type),
    }


def _statistiques_depuis_dict(d: dict) -> StatistiquesSource:
    return StatistiquesSource(
        registres_traites     = d.get("registres_traites", 0),
        registres_avec_entite = d.get("registres_avec_entite", 0),
        entites_par_type      = dict(d.get("entites_par_type", {})),
    )


def rapport_vers_dict(rapport: RapportAnonymisationCumule) -> dict:
    """Serialise le rapport cumule en dict pret pour `json.dump`."""
    return {
        "executions": [
            {
                "horodatage"      : execution.horodatage,
                "dataset"         : execution.dataset,
                "strategie"       : execution.strategie,
                "limite"          : execution.limite,
                "graine_aleatoire": execution.graine_aleatoire,
                "nombre_traites"  : execution.nombre_traites,
                "statistiques_par_source": {
                    source: _statistiques_vers_dict(stats)
                    for source, stats in execution.statistiques_par_source.items()
                },
            }
            for execution in rapport.executions
        ],
        "statistiques_cumulees": {
            source: _statistiques_vers_dict(stats) for source, stats in rapport.statistiques_cumulees.items()
        },
    }


def rapport_depuis_dict(d: dict) -> RapportAnonymisationCumule:
    """Desserialise un rapport cumule depuis un dict issu de `json.load`."""
    executions = [
        ExecutionAnonymisation(
            horodatage       = execution["horodatage"],
            dataset          = execution["dataset"],
            strategie        = execution["strategie"],
            limite           = execution["limite"],
            graine_aleatoire = execution["graine_aleatoire"],
            nombre_traites   = execution["nombre_traites"],
            statistiques_par_source={
                source: _statistiques_depuis_dict(sd)
                for source, sd in execution.get("statistiques_par_source", {}).items()
            },
        )
        for execution in d.get("executions", [])
    ]
    statistiques_cumulees = {
        source: _statistiques_depuis_dict(sd) for source, sd in d.get("statistiques_cumulees", {}).items()
    }
    return RapportAnonymisationCumule(executions=executions, statistiques_cumulees=statistiques_cumulees)


# ----------------------------------------------------------------------
# Fusion (coeur du caractere "cumulatif" du rapport)
# ----------------------------------------------------------------------


def fusionner_execution(
    rapport: RapportAnonymisationCumule,
    execution: ExecutionAnonymisation,
) -> RapportAnonymisationCumule:
    """
    Retourne un NOUVEAU rapport cumule incluant `execution` (ne mute pas
    `rapport`). L'execution est ajoutee telle quelle a la liste
    tracable, et sa contribution par source est additionnee aux
    compteurs deja cumules.
    """
    nouvelles_executions = [*rapport.executions, execution]

    nouvelles_statistiques_cumulees: dict[str, StatistiquesSource] = {
        source: StatistiquesSource(
            registres_traites     = stats.registres_traites,
            registres_avec_entite = stats.registres_avec_entite,
            entites_par_type      = dict(stats.entites_par_type),
        )
        for source, stats in rapport.statistiques_cumulees.items()
    }
    for source, stats_execution in execution.statistiques_par_source.items():
        cumul = nouvelles_statistiques_cumulees.setdefault(source, StatistiquesSource())
        cumul.registres_traites     += stats_execution.registres_traites
        cumul.registres_avec_entite += stats_execution.registres_avec_entite
        for type_entite, compte in stats_execution.entites_par_type.items():
            cumul.entites_par_type[type_entite] = cumul.entites_par_type.get(type_entite, 0) + compte

    return RapportAnonymisationCumule(
        executions            = nouvelles_executions,
        statistiques_cumulees = nouvelles_statistiques_cumulees,
    )


# ----------------------------------------------------------------------
# Presentation (console + Markdown) ; purement derivee du rapport,
# jamais la source de verite (le JSON cumule l'est).
# ----------------------------------------------------------------------


def formater_resume_console(rapport: RapportAnonymisationCumule, total_dataset: int) -> str:
    """Resume lisible en console : chiffres CUMULES, avec le total du dataset pivot pour l'echelle."""
    total_traites      = sum(s.registres_traites              for s in rapport.statistiques_cumulees.values())
    total_avec_entite  = sum(s.registres_avec_entite          for s in rapport.statistiques_cumulees.values())
    total_entites      = sum(sum(s.entites_par_type.values()) for s in rapport.statistiques_cumulees.values())
    
    proportion_dataset = (total_traites / total_dataset * 100) if total_dataset else 0.0
    taux_entite        = (total_avec_entite / total_traites * 100) if total_traites else 0.0

    lignes = [
        "Rapport RGPD cumule (toutes executions d'anonymisation confondues) :",
        f"  Anonymise (cumule).......: {total_traites}/{total_dataset} exemples du dataset pivot ({proportion_dataset:.1f}%)",
        f"  Avec >=1 entite detectee.: {total_avec_entite}/{total_traites} exemples anonymises ({taux_entite:.1f}%)",
        f"  Entites detectees (total): {total_entites}",
        f"  Executions ayant contribue: {len(rapport.executions)}",
    ]
    for source in sorted(rapport.statistiques_cumulees):
        stats = rapport.statistiques_cumulees[source]
        taux = (stats.registres_avec_entite / stats.registres_traites * 100) if stats.registres_traites else 0.0
        lignes.append(
            f"    {source:<28}: {stats.registres_traites} traites (cumule), "
            f"{stats.registres_avec_entite} avec entite ({taux:.1f}%)"
        )
    return "\n".join(lignes)


def formater_rapport_markdown(rapport: RapportAnonymisationCumule, total_dataset: int) -> str:
    """
    Rapport Markdown complet, entierement REGENERE a partir du JSON
    cumule a chaque execution (pas d'edition incrementale du .md :
    c'est une vue derivee, le JSON cumule reste la seule source de
    verite persistee).
    """
    total_traites = sum(s.registres_traites for s in rapport.statistiques_cumulees.values())
    total_avec_entite = sum(s.registres_avec_entite for s in rapport.statistiques_cumulees.values())
    total_entites = sum(sum(s.entites_par_type.values()) for s in rapport.statistiques_cumulees.values())
    proportion_dataset = (total_traites / total_dataset * 100) if total_dataset else 0.0
    taux_entite = (total_avec_entite / total_traites * 100) if total_traites else 0.0

    lignes = [
        "# Rapport d'anonymisation RGPD : genere automatiquement",
        "",
        "> Genere par `interfaces/cli/anonymiser_dataset.py` a chaque execution ; "
        "ne pas editer a la main, ce fichier est entierement regenere depuis le "
        "rapport JSON cumule (`rapport_anonymisation_rgpd.json`) a chaque appel.",
        ">",
        "> **Portee de chaque chiffre** : les chiffres de ce document sont "
        "**cumules sur toutes les executions passees** d'`anonymiser_dataset.py` "
        "(voir la table des executions en bas de page pour le detail de qui a "
        "apporte quoi). Le total du dataset pivot est compte reellement dans le "
        "fichier a chaque execution, jamais code en dur.",
        "",
        "## Portee globale",
        "",
        f"- **Dataset pivot (compte reel a l'execution)** : {total_dataset} exemples.",
        f"- **Anonymise, cumule sur {len(rapport.executions)} execution(s)** : "
        f"{total_traites} exemples, soit {proportion_dataset:.2f}% du dataset pivot.",
        f"- **Avec au moins une entite detectee (cumule)** : {total_avec_entite}/{total_traites} "
        f"exemples anonymises ({taux_entite:.2f}%).",
        f"- **Entites detectees au total (cumule)** : {total_entites}.",
        "",
        "## Repartition cumulee par source",
        "",
        "| Source | Registres traites (cumule) | Avec >=1 entite | Taux | Entites (total) |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in sorted(rapport.statistiques_cumulees):
        stats = rapport.statistiques_cumulees[source]
        taux = (stats.registres_avec_entite / stats.registres_traites * 100) if stats.registres_traites else 0.0
        total_entites_source = sum(stats.entites_par_type.values())
        lignes.append(
            f"| {source} | {stats.registres_traites} | {stats.registres_avec_entite} | {taux:.1f}% | {total_entites_source} |"
        )

    lignes += ["", "## Entites detectees par type, par source (cumule)", ""]
    for source in sorted(rapport.statistiques_cumulees):
        stats = rapport.statistiques_cumulees[source]
        if not stats.entites_par_type:
            continue
        detail = ", ".join(
            f"{type_entite} {compte}"
            for type_entite, compte in sorted(stats.entites_par_type.items(), key=lambda kv: -kv[1])
        )
        lignes.append(f"- **{source}** : {detail}")

    lignes += [
        "",
        "## Executions ayant contribue (tracabilite)",
        "",
        "Chaque ligne correspond a UN appel de `anonymiser_dataset.py` ; "
        "`traites` est le nombre d'exemples apportes PAR CETTE EXECUTION "
        "uniquement (pas cumule), pour retracer qui a produit quels chiffres.",
        "",
        "| Horodatage | Dataset | Strategie | Limite | Graine | Traites (cette execution) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for execution in rapport.executions:
        lignes.append(
            f"| {execution.horodatage} | {execution.dataset} | {execution.strategie} | "
            f"{execution.limite} | {execution.graine_aleatoire} | {execution.nombre_traites} |"
        )
    lignes.append("")

    return "\n".join(lignes)
