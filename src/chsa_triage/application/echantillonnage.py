"""
Echantillonnage stratifie partage entre les cas d'usage qui doivent
prelever un sous-ensemble representatif d'ExemplePivot par
(type_exemple, source) : anonymisation incrementale (`--limite`,
`AnonymiserDatasetUseCase`) et sous-echantillonnage avant decoupage de
splits (`--n`, `DecouperSplitsUseCase`). Extrait de
`AnonymiserDatasetUseCase` pour eviter de dupliquer l'algorithme.
"""

from __future__ import annotations

import random

from chsa_triage.domain.model import ExemplePivot


def echantillon_stratifie(
    candidats: list[ExemplePivot],
    taille: int,
    graine_aleatoire: int = 42,
) -> list[ExemplePivot]:
    """
    Selectionne `taille` exemples parmi `candidats`, en respectant au
    mieux la proportion de chaque strate (type_exemple, source) dans
    l'echantillon (methode du plus grand reste, tirage aleatoire
    reproductible via `graine_aleatoire` au sein de chaque strate).
    """
    rng = random.Random(graine_aleatoire)

    groupes: dict[tuple[str, str], list[ExemplePivot]] = {}
    for exemple in candidats:
        cle = (exemple.type_exemple.value, exemple.source)
        groupes.setdefault(cle, []).append(exemple)

    total = len(candidats)
    quotas: dict[tuple[str, str], int] = {}
    restes: list[tuple[float, tuple[str, str]]] = []
    for cle, groupe in groupes.items():
        part_exacte = taille * (len(groupe) / total)
        quotas[cle] = min(int(part_exacte), len(groupe))
        restes.append((part_exacte - int(part_exacte), cle))

    # Methode du plus grand reste : distribue les unites manquantes
    # (arrondis vers le bas ci-dessus) aux strates dont le reste
    # fractionnaire est le plus grand, dans la limite de leur taille.
    deficit = taille - sum(quotas.values())
    for _, cle in sorted(restes, key=lambda r: r[0], reverse=True):
        if deficit <= 0:
            break
        if quotas[cle] < len(groupes[cle]):
            quotas[cle] += 1
            deficit -= 1

    selection: list[ExemplePivot] = []
    for cle in sorted(groupes):
        groupe = list(groupes[cle])
        rng.shuffle(groupe)
        selection.extend(groupe[: quotas[cle]])

    return selection
