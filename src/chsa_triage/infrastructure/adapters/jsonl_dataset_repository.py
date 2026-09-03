"""
Adaptateur secondaire : persistance des ExemplePivot dans un fichier
JSONL local.

Implemente le port `RepositoryLectureEcriture[ExemplePivot]`. C'est la
seule couche qui sait qu'un ExemplePivot est, concretement, une ligne
JSON dans un fichier -- ni le domaine ni l'application n'en ont
connaissance.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from chsa_triage.domain.model import (
    ConstantesVitales,
    ExemplePivot,
    Langue,
    Message,
    NiveauConfiance,
    TypeExemple,
    TypeSplit,
)


class JsonlDatasetRepository:
    """Adaptateur JSONL local implementant RepositoryLectureEcriture[ExemplePivot]."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self._chemin = Path(chemin_fichier)
        self._chemin.parent.mkdir(parents=True, exist_ok=True)
        if not self._chemin.exists():
            self._chemin.touch()

    # ------------------------------------------------------------------
    # Ecriture
    # ------------------------------------------------------------------

    def sauvegarder(self, item: ExemplePivot) -> None:
        """Remplace l'exemple s'il existe deja (meme identifiant), l'ajoute sinon."""
        existants = {e.identifiant: e for e in self._lire_tous()}
        existants[item.identifiant] = item
        self._ecrire_tous(existants.values())

    def sauvegarder_plusieurs(self, items: Iterable[ExemplePivot]) -> None:
        existants = {e.identifiant: e for e in self._lire_tous()}
        for item in items:
            existants[item.identifiant] = item
        self._ecrire_tous(existants.values())

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def trouver_par_id(self, identifiant: str) -> ExemplePivot | None:
        for exemple in self._lire_tous():
            if exemple.identifiant == identifiant:
                return exemple
        return None

    def lister(self, filtre: dict | None = None) -> Iterator[ExemplePivot]:
        for exemple in self._lire_tous():
            if filtre is None or self._correspond(exemple, filtre):
                yield exemple

    def compter(self, filtre: dict | None = None) -> int:
        return sum(1 for _ in self.lister(filtre))

    # ------------------------------------------------------------------
    # Details prives de serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _correspond(exemple: ExemplePivot, filtre: dict) -> bool:
        for cle, valeur in filtre.items():
            attribut = getattr(exemple, cle, None)
            if hasattr(attribut, "value"):
                attribut = attribut.value
            if attribut != valeur:
                return False
        return True

    def _lire_tous(self) -> list[ExemplePivot]:
        if self._chemin.stat().st_size == 0:
            return []
        exemples = []
        with self._chemin.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    exemples.append(self._depuis_dict(json.loads(ligne)))
        return exemples

    def _ecrire_tous(self, exemples: Iterable[ExemplePivot]) -> None:
        with self._chemin.open("w", encoding="utf-8") as f:
            for exemple in exemples:
                f.write(json.dumps(self._vers_dict(exemple), ensure_ascii=False) + "\n")

    @staticmethod
    def _vers_dict(exemple: ExemplePivot) -> dict:
        d = asdict(exemple)
        d["type_exemple"] = exemple.type_exemple.value
        d["langue"] = exemple.langue.value
        d["niveau_confiance"] = exemple.niveau_confiance.value
        d["split"] = exemple.split.value if exemple.split else None
        return d

    @staticmethod
    def _depuis_dict(d: dict) -> ExemplePivot:
        def messages(cle: str) -> tuple[Message, ...]:
            return tuple(Message(**m) for m in d.get(cle, []))

        constantes = d.get("constantes_vitales")
        return ExemplePivot(
            identifiant=d["identifiant"],
            source=d["source"],
            type_exemple=TypeExemple(d["type_exemple"]),
            langue=Langue(d["langue"]),
            symptomes=d.get("symptomes", ""),
            antecedents=d.get("antecedents"),
            constantes_vitales=ConstantesVitales(**constantes) if constantes else None,
            prompt=messages("prompt"),
            completion=messages("completion"),
            chosen=messages("chosen"),
            rejected=messages("rejected"),
            niveau_confiance=NiveauConfiance(d.get("niveau_confiance", "moyenne")),
            anonymise=d.get("anonymise", False),
            split=TypeSplit(d["split"]) if d.get("split") else None,
        )
