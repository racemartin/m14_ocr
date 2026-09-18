"""
Adaptateur secondaire : persistance JSONL des `ExempleFormatePreference`
(triplet prompt/chosen/rejected rendu texte), produits par
`FormaterDatasetChatMLPreferenceUseCase` (Etape 3/DPO).

Implemente `RepositoryLectureEcriture[ExempleFormatePreference]`.
Adaptateur DEDIE (meme discipline que
`jsonl_exemple_formate_repository.py`, pas une generalisation de
celui-ci : la forme de donnee differe, un triplet distinct plutot
qu'un texte unique), cinquieme instance du port generique apres
`ExemplePivot`, `ExempleFormate`, `CheckpointEntraine` et
`ChosenReformule`, cf. docs/04_etape3_dpo/02_etapes_cas_usage.md §3.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference


class JsonlExempleFormatePreferenceRepository:
    """Adaptateur JSONL local implementant RepositoryLectureEcriture[ExempleFormatePreference]."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self._chemin = Path(chemin_fichier)
        self._chemin.parent.mkdir(parents=True, exist_ok=True)
        if not self._chemin.exists():
            self._chemin.touch()

    def sauvegarder(self, item: ExempleFormatePreference) -> None:
        self.sauvegarder_plusieurs([item])

    def sauvegarder_plusieurs(self, items: Iterable[ExempleFormatePreference]) -> None:
        existants = {e.identifiant: e for e in self._lire_tous()}
        for item in items:
            existants[item.identifiant] = item
        self._ecrire_tous(existants.values())

    def trouver_par_id(self, identifiant: str) -> ExempleFormatePreference | None:
        for exemple in self._lire_tous():
            if exemple.identifiant == identifiant:
                return exemple
        return None

    def lister(self, filtre: dict | None = None) -> Iterator[ExempleFormatePreference]:
        for exemple in self._lire_tous():
            if filtre is None or all(getattr(exemple, cle, None) == valeur for cle, valeur in filtre.items()):
                yield exemple

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

    def _lire_tous(self) -> list[ExempleFormatePreference]:
        if self._chemin.stat().st_size == 0:
            return []
        exemples = []
        with self._chemin.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    d = json.loads(ligne)
                    exemples.append(
                        ExempleFormatePreference(
                            identifiant=d["identifiant"],
                            texte_prompt=d["texte_prompt"],
                            texte_chosen=d["texte_chosen"],
                            texte_rejected=d["texte_rejected"],
                        )
                    )
        return exemples

    def _ecrire_tous(self, exemples: Iterable[ExempleFormatePreference]) -> None:
        with self._chemin.open("w", encoding="utf-8") as f:
            for exemple in exemples:
                ligne = {
                    "identifiant": exemple.identifiant,
                    "texte_prompt": exemple.texte_prompt,
                    "texte_chosen": exemple.texte_chosen,
                    "texte_rejected": exemple.texte_rejected,
                }
                f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
