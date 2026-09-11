"""
Adaptateur secondaire : persistance JSONL des `ExempleFormate` (rendu
ChatML), produits par `FormaterDatasetChatMLUseCase`.

Implemente `RepositoryLectureEcriture[ExempleFormate]`. Adaptateur
DEDIE (pas une generalisation de `JsonlDatasetRepository`, qui reste
specifique a `ExemplePivot`) : meme discipline que
`jsonl_decisions_revision_humaine.py`, chaque entite du domaine a son
propre adaptateur JSONL plutot qu'une classe generique parametree.
Meme mecanisme de sauvegarde que `JsonlDatasetRepository` (relit puis
reecrit tout le fichier a chaque `sauvegarder()`/`sauvegarder_plusieurs()`,
dedoublonne par `identifiant`) : mêmes limites de performance, cf.
AGENTS.md (jamais `sauvegarder()` dans une boucle par item).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from chsa_triage.domain.model.exemple_formate import ExempleFormate


class JsonlExempleFormateRepository:
    """Adaptateur JSONL local implementant RepositoryLectureEcriture[ExempleFormate]."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self._chemin = Path(chemin_fichier)
        self._chemin.parent.mkdir(parents=True, exist_ok=True)
        if not self._chemin.exists():
            self._chemin.touch()

    def sauvegarder(self, item: ExempleFormate) -> None:
        self.sauvegarder_plusieurs([item])

    def sauvegarder_plusieurs(self, items: Iterable[ExempleFormate]) -> None:
        existants = {e.identifiant: e for e in self._lire_tous()}
        for item in items:
            existants[item.identifiant] = item
        self._ecrire_tous(existants.values())

    def trouver_par_id(self, identifiant: str) -> ExempleFormate | None:
        for exemple in self._lire_tous():
            if exemple.identifiant == identifiant:
                return exemple
        return None

    def lister(self, filtre: dict | None = None) -> Iterator[ExempleFormate]:
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

    def _lire_tous(self) -> list[ExempleFormate]:
        if self._chemin.stat().st_size == 0:
            return []
        exemples = []
        with self._chemin.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    d = json.loads(ligne)
                    exemples.append(ExempleFormate(identifiant=d["identifiant"], texte=d["texte"]))
        return exemples

    def _ecrire_tous(self, exemples: Iterable[ExempleFormate]) -> None:
        with self._chemin.open("w", encoding="utf-8") as f:
            for exemple in exemples:
                ligne = {"identifiant": exemple.identifiant, "texte": exemple.texte}
                f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
