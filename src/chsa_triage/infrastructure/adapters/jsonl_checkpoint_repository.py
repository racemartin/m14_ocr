"""
Adaptateur secondaire : persistance JSONL des metadonnees
`CheckpointEntraine`, produites par `SauvegarderCheckpointSftUseCase`.

Implemente `RepositoryLectureEcriture[CheckpointEntraine]`. Adaptateur
DEDIE (meme discipline que `jsonl_exemple_formate_repository.py` /
`jsonl_decisions_revision_humaine.py`, pas une generalisation de
`JsonlDatasetRepository`). Ne persiste jamais les poids LoRA
eux-memes (ecrits sur disque par `trl`/`peft`, cf.
`CheckpointEntraine.chemin`) : uniquement les metadonnees structurees.

GAP reel trouve+corrige le 19/09/2026 (Etape 3/DPO, etapes 12-13) :
`checkpoint_depuis_dict` desserialisait `hyperparametres` en
`HyperparametresEntrainement` (SFT) SANS CONDITION, alors que
`CheckpointEntraine.hyperparametres` accepte deja l'union SFT|DPO
depuis l'edition de `domain/model/checkpoint_entraine.py` (guide Etape
3 etape 5, deja fusionnee) : un checkpoint DPO persiste via ce depot
aurait leve un `TypeError` a la relecture (champs DPO comme `beta`
passes en kwarg inconnu a `HyperparametresEntrainement`). Corrige par
un discriminant sur la cle `beta` (presente UNIQUEMENT sur
`HyperparametresEntrainementDpo`, absente de `HyperparametresEntrainement`
qui porte `packing` a la place) : pas de champ `type` explicite ajoute
au schema JSONL, moins de rupture pour les checkpoints SFT deja
persistes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from chsa_triage.domain.model.checkpoint_entraine import (
    CheckpointEntraine,
    VerdictConvergence,
)
from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    HyperparametresEntrainement,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement


def checkpoint_vers_dict(checkpoint: CheckpointEntraine) -> dict:
    d = asdict(checkpoint)
    d["verdict_convergence"] = checkpoint.verdict_convergence.value
    return d


def _hyperparametres_depuis_dict(d: dict) -> HyperparametresEntrainement | HyperparametresEntrainementDpo:
    """`beta` n'existe que sur `HyperparametresEntrainementDpo` (cf. AVERTISSEMENT en tete de module)."""
    if "beta" in d:
        return HyperparametresEntrainementDpo(**d)
    return HyperparametresEntrainement(**d)


def checkpoint_depuis_dict(d: dict) -> CheckpointEntraine:
    configuration_lora = dict(d["configuration_lora"])
    configuration_lora["modules_cibles"] = tuple(configuration_lora["modules_cibles"])
    return CheckpointEntraine(
        identifiant=d["identifiant"],
        chemin=d["chemin"],
        modele_base=d["modele_base"],
        configuration_lora=ConfigurationLora(**configuration_lora),
        hyperparametres=_hyperparametres_depuis_dict(d["hyperparametres"]),
        metriques_finales=MetriquesEntrainement(**d["metriques_finales"]),
        verdict_convergence=VerdictConvergence(d["verdict_convergence"]),
        horodatage=d["horodatage"],
    )


class JsonlCheckpointRepository:
    """Adaptateur JSONL local implementant RepositoryLectureEcriture[CheckpointEntraine]."""

    def __init__(self, chemin_fichier: str | Path) -> None:
        self._chemin = Path(chemin_fichier)
        self._chemin.parent.mkdir(parents=True, exist_ok=True)
        if not self._chemin.exists():
            self._chemin.touch()

    def sauvegarder(self, item: CheckpointEntraine) -> None:
        self.sauvegarder_plusieurs([item])

    def sauvegarder_plusieurs(self, items: Iterable[CheckpointEntraine]) -> None:
        existants = {c.identifiant: c for c in self._lire_tous()}
        for item in items:
            existants[item.identifiant] = item
        self._ecrire_tous(existants.values())

    def trouver_par_id(self, identifiant: str) -> CheckpointEntraine | None:
        for checkpoint in self._lire_tous():
            if checkpoint.identifiant == identifiant:
                return checkpoint
        return None

    def lister(self, filtre: dict | None = None) -> Iterator[CheckpointEntraine]:
        for checkpoint in self._lire_tous():
            if filtre is None or all(getattr(checkpoint, cle, None) == valeur for cle, valeur in filtre.items()):
                yield checkpoint

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

    def _lire_tous(self) -> list[CheckpointEntraine]:
        if self._chemin.stat().st_size == 0:
            return []
        checkpoints = []
        with self._chemin.open("r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    checkpoints.append(checkpoint_depuis_dict(json.loads(ligne)))
        return checkpoints

    def _ecrire_tous(self, checkpoints: Iterable[CheckpointEntraine]) -> None:
        with self._chemin.open("w", encoding="utf-8") as f:
            for checkpoint in checkpoints:
                f.write(json.dumps(checkpoint_vers_dict(checkpoint), ensure_ascii=False) + "\n")
