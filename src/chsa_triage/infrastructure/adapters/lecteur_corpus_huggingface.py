"""
Adaptateur secondaire : lecture d'un corpus brut directement depuis
le Hugging Face Hub (`datasets.load_dataset`).

Implemente le meme port `LecteurCorpus` que l'adaptateur fichier
local -- l'application ne voit aucune difference entre les deux.
"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset


class LecteurCorpusHuggingFace:
    """Adaptateur Hugging Face Hub implementant LecteurCorpus."""

    def __init__(self, identifiant_hub: str, configuration: str | None = None, split: str = "train") -> None:
        self._identifiant_hub = identifiant_hub
        self._configuration = configuration
        self._split = split
        self._dataset = None

    def lire_enregistrements(self) -> Iterator[dict]:
        for enregistrement in self._charger():
            yield dict(enregistrement)

    def compter_enregistrements(self) -> int:
        return len(self._charger())

    def _charger(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                self._identifiant_hub,
                self._configuration,
                split=self._split,
            )
        return self._dataset
