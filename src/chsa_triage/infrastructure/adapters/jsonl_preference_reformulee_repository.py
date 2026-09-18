"""
Adaptateur secondaire : persistance JSONL des `ChosenReformule`
(chosen reformule vers `<think>`+JSON), produits par
`ReformulerPreferenceDpoUseCase` (Etape 3/DPO).

Implemente `RepositoryLectureEcriture[ChosenReformule]`. Adaptateur
DEDIE (meme discipline que `jsonl_exemple_formate_repository.py` /
`jsonl_checkpoint_repository.py`, pas une generalisation de
`JsonlDatasetRepository`) : quatrieme instance du port generique
apres `ExemplePivot`, `ExempleFormate` et `CheckpointEntraine`, cf.
docs/04_etape3_dpo/02_etapes_cas_usage.md §1.5. Ne mute jamais
`dataset_pivot_anonymise.jsonl` : fichier separe, meme principe deja
en place pour le pivot/anonymise (cf. AGENTS.md).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from chsa_triage.domain.model.exemple_pivot import Message
from chsa_triage.domain.model.preference_reformulee import ChosenReformule


def preference_reformulee_vers_dict(item: ChosenReformule) -> dict:
    return asdict(item)


def preference_reformulee_depuis_dict(d: dict) -> ChosenReformule:
    return ChosenReformule(
        identifiant=d["identifiant"],
        chosen_reformule=tuple(Message(**m) for m in d["chosen_reformule"]),
        horodatage=d["horodatage"],
    )


class JsonlPreferenceReformuleeRepository:
    """Adaptateur JSONL local implementant RepositoryLectureEcriture[ChosenReformule]."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self._chemin = Path(chemin_fichier)
        self._chemin.parent.mkdir(parents=True, exist_ok=True)
        if not self._chemin.exists():
            self._chemin.touch()

    def sauvegarder(self, item: ChosenReformule) -> None:
        self.sauvegarder_plusieurs([item])

    def sauvegarder_plusieurs(self, items: Iterable[ChosenReformule]) -> None:
        existants = {c.identifiant: c for c in self._lire_tous()}
        for item in items:
            existants[item.identifiant] = item
        self._ecrire_tous(existants.values())

    def trouver_par_id(self, identifiant: str) -> ChosenReformule | None:
        for item in self._lire_tous():
            if item.identifiant == identifiant:
                return item
        return None

    def lister(self, filtre: dict | None = None) -> Iterator[ChosenReformule]:
        for item in self._lire_tous():
            if filtre is None or all(getattr(item, cle, None) == valeur for cle, valeur in filtre.items()):
                yield item

    def compter(self, filtre: dict | None = None) -> int:
        return sum(1 for _ in self.lister(filtre))

    def identifiants_existants(self) -> set[str]:
        if self._chemin.stat().st_size == 0:
            return set()
        identifiants: set[str] = set()
        with self._chemin.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    identifiants.add(json.loads(ligne)["identifiant"])
        return identifiants

    def _lire_tous(self) -> list[ChosenReformule]:
        if self._chemin.stat().st_size == 0:
            return []
        items = []
        with self._chemin.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    items.append(preference_reformulee_depuis_dict(json.loads(ligne)))
        return items

    def _ecrire_tous(self, items: Iterable[ChosenReformule]) -> None:
        with self._chemin.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(preference_reformulee_vers_dict(item), ensure_ascii=False) + "\n")
