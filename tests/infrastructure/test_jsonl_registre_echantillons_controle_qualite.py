"""Test de l'adaptateur JsonlRegistreEchantillonsControleQualite ; round-trip sur disque."""

from __future__ import annotations

from pathlib import Path

from chsa_triage.infrastructure.adapters.jsonl_registre_echantillons_controle_qualite import (
    JsonlRegistreEchantillonsControleQualite,
)


def test_identifiants_vus_sur_fichier_vide(tmp_path: Path):
    registre = JsonlRegistreEchantillonsControleQualite(tmp_path / "vus.jsonl")
    assert registre.identifiants_vus("principal") == set()


def test_marquer_vus_persiste_et_se_relit(tmp_path: Path):
    chemin = tmp_path / "vus.jsonl"
    registre = JsonlRegistreEchantillonsControleQualite(chemin)

    registre.marquer_vus("principal", ["id-1", "id-2"], "2026-09-09T10:00:00+00:00")

    assert registre.identifiants_vus("principal") == {"id-1", "id-2"}


def test_strates_independantes(tmp_path: Path):
    chemin = tmp_path / "vus.jsonl"
    registre = JsonlRegistreEchantillonsControleQualite(chemin)

    registre.marquer_vus("principal", ["id-1"], "2026-09-09T10:00:00+00:00")
    registre.marquer_vus("sans_entite", ["id-2"], "2026-09-09T10:00:00+00:00")

    assert registre.identifiants_vus("principal") == {"id-1"}
    assert registre.identifiants_vus("sans_entite") == {"id-2"}


def test_marquer_vus_accumule_entre_appels(tmp_path: Path):
    chemin = tmp_path / "vus.jsonl"
    registre = JsonlRegistreEchantillonsControleQualite(chemin)

    registre.marquer_vus("principal", ["id-1"], "2026-09-09T10:00:00+00:00")
    registre.marquer_vus("principal", ["id-2"], "2026-09-09T11:00:00+00:00")

    assert registre.identifiants_vus("principal") == {"id-1", "id-2"}


def test_nouvelle_instance_relit_le_meme_fichier(tmp_path: Path):
    chemin = tmp_path / "vus.jsonl"
    JsonlRegistreEchantillonsControleQualite(chemin).marquer_vus("principal", ["id-1"], "2026-09-09T10:00:00+00:00")

    registre_relu = JsonlRegistreEchantillonsControleQualite(chemin)
    assert registre_relu.identifiants_vus("principal") == {"id-1"}
