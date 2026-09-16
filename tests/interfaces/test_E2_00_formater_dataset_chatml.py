"""
Tests de `interfaces/cli/E2_00_formater_dataset_chatml.py` : point
d'entree CLI autonome pour `FormaterDatasetChatMLUseCase` (jusqu'ici
invoque uniquement depuis l'interieur de `training/E2_04_sft_train.py`),
avec un mode didactique optionnel (`--exemples N`, cape a 2) qui logue
par `LogTool`, en lecture seule, l'`ExemplePivot` brut puis son rendu
ChatML.

`ChatMLFormateurAdapter` est monkeypatche par un faux adaptateur (pas
de tokenizer reel/reseau, meme raison que
`tests/application/test_E2_00_uc_formater_dataset_chatml.py`) ; le
reste (repositories, use case) est le code REEL sur des fichiers
temporaires, pour verifier la vraie ecriture sur disque, pas seulement
la logique en memoire.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

# interfaces/cli n'est pas un package installe ; ajouter la racine du
# depot au chemin de recherche, meme patron que
# tests/interfaces/test_E1_03_01_mappers_corpus.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chsa_triage.domain.model import ExempleFormate, ExemplePivot, Langue, Message, TypeExemple, TypeSplit
from chsa_triage.infrastructure.adapters import JsonlDatasetRepository, JsonlExempleFormateRepository
from interfaces.cli import E2_00_formater_dataset_chatml as cli
from tools.rafael.log_tool import LogTool


class FauxChatMLFormateurAdapter:
    """
    Faux `ChatMLFormateurAdapter` : concatene prompt/completion sans
    tokenizer reel, meme principe que `FauxFormateurConversation` dans
    `tests/application/test_E2_00_uc_formater_dataset_chatml.py`.
    Accepte `nom_modele=` en kwarg pour matcher l'appel reel du CLI.
    """

    def __init__(self, nom_modele: str = "") -> None:
        self.nom_modele = nom_modele
        self.appels = 0

    def formater(self, exemple: ExemplePivot) -> ExempleFormate:
        self.appels += 1
        texte_prompt = " ".join(m.contenu for m in exemple.prompt)
        texte_completion = " ".join(m.contenu for m in exemple.completion)
        return ExempleFormate(
            identifiant=exemple.identifiant,
            texte=f"<|im_start|>user\n{texte_prompt}<|im_end|>\n<|im_start|>assistant\n{texte_completion}<|im_end|>",
        )


def _exemple(source: str, split: TypeSplit | None = TypeSplit.TRAIN) -> ExemplePivot:
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant(source, cle),
        identifiant_source_brute=cle,
        source=source,
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        prompt=(Message(role="user", contenu=f"Question {cle[:6]} ?"),),
        completion=(Message(role="assistant", contenu=f"Reponse {cle[:6]}."),),
        split=split,
    )


@pytest.fixture(autouse=True)
def _faux_tokenizer(monkeypatch):
    """Empeche tout appel reel a `ChatMLFormateurAdapter`/transformers dans tout ce module."""
    monkeypatch.setattr(cli, "ChatMLFormateurAdapter", FauxChatMLFormateurAdapter)


def _executer_cli(monkeypatch, argv: list[str]) -> str:
    """Lance `cli.main()` avec `argv`, retourne la sortie stderr (LogTool) capturee."""
    monkeypatch.setattr(sys, "argv", ["E2_00_formater_dataset_chatml.py", *argv])
    cli.main()


class TestNombreExemplesDidactiquesCape:
    @pytest.mark.parametrize(
        ("valeur", "attendu"),
        [(0, 0), (1, 1), (2, 2), (3, 2), (5, 2), (-1, 0), (-100, 0)],
    )
    def test_cape_entre_0_et_2(self, valeur, attendu):
        assert cli.nombre_exemples_didactiques_cape(valeur) == attendu


class TestAfficherExempleDidactique:
    def test_logue_les_tours_prompt_completion_et_le_texte_chatml(self, capsys):
        log = LogTool(origin="test")
        exemple = _exemple("MediQAl")
        exemple_formate = ExempleFormate(
            identifiant=exemple.identifiant,
            texte="<|im_start|>user\nQuestion ?<|im_end|>\n<|im_start|>assistant\nReponse.<|im_end|>",
        )

        cli.afficher_exemple_didactique(log, 1, 2, exemple, exemple_formate)

        sortie = capsys.readouterr().err
        assert exemple.identifiant in sortie
        assert "role=user" in sortie
        assert "role=assistant" in sortie
        assert exemple.prompt[0].contenu in sortie
        assert exemple.completion[0].contenu in sortie
        assert "<|im_start|>assistant" in sortie

    def test_naffecte_aucun_fichier_ni_repository(self, tmp_path, capsys):
        """
        Purement lecture/console : ne prend en parametre aucun
        repository/chemin de fichier, ne peut donc rien ecrire nulle
        part (verification structurelle, pas seulement comportementale).
        """
        import inspect

        parametres = inspect.signature(cli.afficher_exemple_didactique).parameters
        assert set(parametres) == {"log_tool", "index", "total", "exemple", "exemple_formate"}


class TestMainSansModeDidactique:
    def test_ecrit_le_fichier_formate_normalement(self, tmp_path, monkeypatch):
        dataset = tmp_path / "pivot.jsonl"
        sortie = tmp_path / "formate.jsonl"
        exemples = [_exemple("MediQAl") for _ in range(3)]
        JsonlDatasetRepository(dataset).sauvegarder_plusieurs(exemples)

        _executer_cli(
            monkeypatch,
            ["--dataset", str(dataset), "--dataset-formate", str(sortie), "--split", "train"],
        )

        formates = list(JsonlExempleFormateRepository(sortie).lister())
        assert {e.identifiant for e in formates} == {e.identifiant for e in exemples}

    def test_sans_flag_exemples_naffiche_aucun_log_didactique(self, tmp_path, monkeypatch, capsys):
        dataset = tmp_path / "pivot.jsonl"
        sortie = tmp_path / "formate.jsonl"
        JsonlDatasetRepository(dataset).sauvegarder_plusieurs([_exemple("MediQAl")])

        _executer_cli(
            monkeypatch,
            ["--dataset", str(dataset), "--dataset-formate", str(sortie), "--split", "train"],
        )

        sortie_log = capsys.readouterr().err
        assert "Exemple didactique" not in sortie_log
        assert "Mode didactique" not in sortie_log


class TestMainAvecModeDidactique:
    def test_logue_au_plus_2_exemples_meme_si_plus_demandes(self, tmp_path, monkeypatch, capsys):
        dataset = tmp_path / "pivot.jsonl"
        sortie = tmp_path / "formate.jsonl"
        exemples = [_exemple("MediQAl") for _ in range(5)]
        JsonlDatasetRepository(dataset).sauvegarder_plusieurs(exemples)

        _executer_cli(
            monkeypatch,
            ["--dataset", str(dataset), "--dataset-formate", str(sortie), "--split", "train", "--exemples", "5"],
        )

        sortie_log = capsys.readouterr().err
        assert sortie_log.count("Exemple didactique") == 2
        assert "exemples didactiques (capes)" in sortie_log or "exemples didactiques (capes)".lower() in sortie_log.lower()

    def test_affiche_le_texte_chatml_final_avec_les_tours_de_role(self, tmp_path, monkeypatch, capsys):
        dataset = tmp_path / "pivot.jsonl"
        sortie = tmp_path / "formate.jsonl"
        JsonlDatasetRepository(dataset).sauvegarder_plusieurs([_exemple("MediQAl")])

        _executer_cli(
            monkeypatch,
            ["--dataset", str(dataset), "--dataset-formate", str(sortie), "--split", "train", "--exemples", "1"],
        )

        sortie_log = capsys.readouterr().err
        assert "<|im_start|>user" in sortie_log
        assert "<|im_start|>assistant" in sortie_log


class TestPasDeRegressionSurLeFichierEcrit:
    def test_ecriture_identique_avec_ou_sans_mode_didactique(self, tmp_path, monkeypatch):
        """
        Le mode didactique est purement additif : le fichier
        `--dataset-formate` produit doit etre BYTE A BYTE identique,
        que `--exemples` soit 0 (comportement d'aujourd'hui) ou 2
        (mode didactique active), a partir du meme pivot source.
        """
        dataset = tmp_path / "pivot.jsonl"
        exemples = [_exemple("MediQAl") for _ in range(3)] + [_exemple("FrenchMedMCQA") for _ in range(2)]
        JsonlDatasetRepository(dataset).sauvegarder_plusieurs(exemples)

        sortie_sans_didactique = tmp_path / "formate_sans.jsonl"
        sortie_avec_didactique = tmp_path / "formate_avec.jsonl"

        _executer_cli(
            monkeypatch,
            ["--dataset", str(dataset), "--dataset-formate", str(sortie_sans_didactique), "--split", "train"],
        )
        _executer_cli(
            monkeypatch,
            [
                "--dataset", str(dataset),
                "--dataset-formate", str(sortie_avec_didactique),
                "--split", "train",
                "--exemples", "2",
            ],
        )

        assert sortie_sans_didactique.read_text(encoding="utf-8") == sortie_avec_didactique.read_text(encoding="utf-8")
