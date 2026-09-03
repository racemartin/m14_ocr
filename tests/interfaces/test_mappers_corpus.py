"""
Tests des mappers specifiques par corpus (interfaces/cli/mappers_corpus.py).

Ces mappers n'avaient aucun test avant le smoke test d'integration du
02/09/2026 -- ajoutes a cette occasion. Utilisent des enregistrements
synthetiques representatifs des schemas documentes par les fiches
Hugging Face de chaque corpus (a confirmer/ajuster une fois les
corpus reels profiles, cf. docs/02_etape1_donnees).
"""

from __future__ import annotations

import sys
from pathlib import Path

# interfaces/cli n'est pas un package installe -- ajouter la racine
# du depot au chemin de recherche pour importer mappers_corpus.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chsa_triage.domain.model import Langue, TypeExemple
from interfaces.cli.mappers_corpus import (
    mapper_frenchmedmcqa,
    mapper_mediqal,
    mapper_medquad,
    mapper_ultramedical_preference,
)


def test_mapper_mediqal_valide():
    enregistrement = {"question": "Symptomes du rhume ?", "answer": "Nez qui coule, toux."}
    exemple = mapper_mediqal(enregistrement)

    assert exemple is not None
    assert exemple.source == "MediQAl"
    assert exemple.type_exemple == TypeExemple.SFT
    assert exemple.langue == Langue.FRANCAIS
    assert exemple.prompt[0].contenu == "Symptomes du rhume ?"
    assert exemple.completion[0].contenu == "Nez qui coule, toux."
    assert exemple.est_complet_pour_sft()


def test_mapper_mediqal_ignore_enregistrement_incomplet():
    assert mapper_mediqal({"question": "Sans reponse"}) is None
    assert mapper_mediqal({"answer": "Sans question"}) is None
    assert mapper_mediqal({}) is None


def test_mapper_frenchmedmcqa_valide():
    enregistrement = {
        "question": "Quel est le traitement de premiere intention ?",
        "options": {"a": "Paracetamol", "b": "Ibuprofene"},
        "correct_answers": "a",
    }
    exemple = mapper_frenchmedmcqa(enregistrement)

    assert exemple is not None
    assert exemple.source == "FrenchMedMCQA"
    assert "Paracetamol" in exemple.prompt[0].contenu
    assert exemple.completion[0].contenu == "a"


def test_mapper_frenchmedmcqa_ignore_sans_reponse_correcte():
    assert mapper_frenchmedmcqa({"question": "Q", "options": {}}) is None


def test_mapper_medquad_valide():
    enregistrement = {"Question": "What is diabetes?", "Answer": "A chronic condition."}
    exemple = mapper_medquad(enregistrement)

    assert exemple is not None
    assert exemple.source == "MedQuAD"
    assert exemple.langue == Langue.ANGLAIS
    assert exemple.est_complet_pour_sft()


def test_mapper_medquad_accepte_cles_minuscules():
    """Certaines exportations HF utilisent des cles en minuscules."""
    enregistrement = {"question": "What is diabetes?", "answer": "A chronic condition."}
    exemple = mapper_medquad(enregistrement)
    assert exemple is not None


def test_mapper_ultramedical_preference_valide():
    enregistrement = {
        "prompt": "Explique la fievre a un enfant.",
        "chosen": "La fievre est une reaction normale du corps...",
        "rejected": "Ce n'est pas grave, ignore-la.",
    }
    exemple = mapper_ultramedical_preference(enregistrement)

    assert exemple is not None
    assert exemple.type_exemple == TypeExemple.DPO
    assert exemple.est_complet_pour_dpo()
    assert exemple.chosen[0].contenu.startswith("La fievre")
    assert exemple.rejected[0].contenu.startswith("Ce n'est pas grave")


def test_mapper_ultramedical_preference_ignore_paire_incomplete():
    assert mapper_ultramedical_preference({"prompt": "Q", "chosen": "R"}) is None


def test_mapper_ultramedical_preference_format_chat_reel():
    """
    Format REEL du corpus (confirme le 03/09/2026 sur les 109353
    enregistrements) : chosen/rejected sont des listes de messages
    [{"content": ..., "role": "user"|"assistant"}], pas des chaines
    directes. Le mapper doit extraire le dernier message (assistant),
    pas serialiser toute la liste.
    """
    enregistrement = {
        "prompt": "Explique la fievre a un enfant.",
        "chosen": [
            {"content": "Explique la fievre a un enfant.", "role": "user"},
            {"content": "La fievre est une reaction normale du corps...", "role": "assistant"},
        ],
        "rejected": [
            {"content": "Explique la fievre a un enfant.", "role": "user"},
            {"content": "Ce n'est pas grave, ignore-la.", "role": "assistant"},
        ],
    }
    exemple = mapper_ultramedical_preference(enregistrement)

    assert exemple is not None
    assert exemple.chosen[0].contenu == "La fievre est une reaction normale du corps..."
    assert exemple.rejected[0].contenu == "Ce n'est pas grave, ignore-la."
    # Le prompt duplique en tete de la liste chosen/rejected ne doit pas
    # se retrouver dans le contenu du message pivot.
    assert "role" not in exemple.chosen[0].contenu
    assert "Explique la fievre a un enfant." not in exemple.chosen[0].contenu
