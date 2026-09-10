"""
Adaptateur secondaire : persistance JSONL du registre "quels
identifiants ont deja ete echantillonnes pour le controle qualite"
(par stratum). Implemente le port `RegistreEchantillonsControleQualite`.

Append-only : un identifiant, une fois marque vu pour un stratum, le
reste pour toujours -- pas de fusion/remplacement necessaire (a la
difference de `JsonlDatasetRepository.sauvegarder`, qui fusionne par
identifiant parce qu'un ExemplePivot peut changer de contenu). Meme
esprit que `ajouter_exemples_jsonl` (fichiers d'audit append-only),
mais pour un type d'enregistrement different (identifiant+stratum, pas
un ExemplePivot).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


class JsonlRegistreEchantillonsControleQualite:
    """Adaptateur JSONL local implementant RegistreEchantillonsControleQualite."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self._chemin = Path(chemin_fichier)
        self._chemin.parent.mkdir(parents=True, exist_ok=True)
        if not self._chemin.exists():
            self._chemin.touch()

    def identifiants_vus(self, stratum: str) -> set[str]:
        if self._chemin.stat().st_size == 0:
            return set()
        identifiants: set[str] = set()
        with self._chemin.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:
                    continue
                enregistrement = json.loads(ligne)
                if enregistrement["stratum"] == stratum:
                    identifiants.add(enregistrement["identifiant"])
        return identifiants

    def marquer_vus(self, stratum: str, identifiants: Iterable[str], horodatage: str) -> None:
        with self._chemin.open("a", encoding="utf-8") as f:
            for identifiant in identifiants:
                f.write(
                    json.dumps(
                        {"identifiant": identifiant, "stratum": stratum, "horodatage": horodatage},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
