"""
Test d'integration reel de TensorboardSuiviExperimentation ; pas un mock.

Ecrit dans un repertoire de logs temporaire (`tmp_path`) et confirme
que les metriques logguees sont bien relisibles ensuite via
`EventAccumulator`.

Se saute automatiquement si `tensorboard`/`torch` ne sont pas
installes, meme principe que
`tests/infrastructure/test_presidio_anonymiseur.py`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tensorboard")
pytest.importorskip("torch")

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from chsa_triage.infrastructure.adapters.tensorboard_suivi_experimentation import (
    TensorboardSuiviExperimentation,
)


def test_run_demarre_metriques_logguees_relisibles_apres_coup(tmp_path):
    suivi = TensorboardSuiviExperimentation(repertoire_logs=str(tmp_path))

    suivi.demarrer_run("run-test", {"rang_lora": 8})
    suivi.logger_metrique("perte", 1.2, 0)
    suivi.logger_metrique("perte", 0.8, 1)
    suivi.terminer_run()

    accumulateur = EventAccumulator(str(tmp_path / "run-test"))
    accumulateur.Reload()

    assert "perte" in accumulateur.Tags()["scalars"]
    valeurs = [(evenement.step, evenement.value) for evenement in accumulateur.Scalars("perte")]
    etapes = [step for step, _ in valeurs]
    metriques = [valeur for _, valeur in valeurs]
    assert etapes == [0, 1]
    # TensorBoard stocke les scalaires en float32 (protobuf) : comparer
    # avec une tolerance plutot que par egalite exacte.
    assert metriques == pytest.approx([1.2, 0.8], rel=1e-6)
