"""
Tests des mappers specifiques par corpus (interfaces/cli/mappers_corpus.py).

Ces mappers n'avaient aucun test avant le smoke test d'integration du
02/09/2026 ; ajoutes a cette occasion. Utilisent des enregistrements
synthetiques representatifs des schemas documentes par les fiches
Hugging Face de chaque corpus (a confirmer/ajuster une fois les
corpus reels profiles, cf. docs/02_etape1_donnees).
"""

from __future__ import annotations

import sys
from pathlib import Path

# interfaces/cli n'est pas un package installe ; ajouter la racine
# du depot au chemin de recherche pour importer mappers_corpus.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chsa_triage.domain.model import Langue, TypeExemple
from interfaces.cli.mappers_corpus import (
    mapper_frenchmedmcqa,
    mapper_mediqal,
    mapper_mediqal_qcm,
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


def test_mapper_mediqal_identifiant_est_deterministe_et_trace_le_champ_id():
    enregistrement = {"id": "42", "question": "Symptomes du rhume ?", "answer": "Nez qui coule, toux."}
    exemple_1 = mapper_mediqal(enregistrement)
    exemple_2 = mapper_mediqal(dict(enregistrement))

    assert exemple_1.identifiant == exemple_2.identifiant  # meme registre -> meme identifiant
    assert exemple_1.identifiant_source_brute == "42"


def test_mapper_mediqal_qcm_mcqu_valide():
    """Schema reel MediQAl "mcqu" (07/09/2026) : 1 seule reponse correcte."""
    enregistrement = {
        "clinical_case": "Monsieur R. part au Gabon pendant 2 ans.",
        "question": "Au sujet des vaccinations :",
        "answer_a": "Le vaccin contre la fievre jaune est obligatoire",
        "answer_b": "Autre reponse",
        "answer_c": "Autre reponse",
        "answer_d": "Autre reponse",
        "answer_e": "Autre reponse",
        "correct_answers": "A",
        "task": "QCU",
    }
    exemple = mapper_mediqal_qcm(enregistrement)

    assert exemple is not None
    assert exemple.source == "MediQAl"
    assert "Monsieur R. part au Gabon" in exemple.prompt[0].contenu
    assert "Au sujet des vaccinations" in exemple.prompt[0].contenu
    assert exemple.completion[0].contenu == "Le vaccin contre la fievre jaune est obligatoire"
    assert exemple.est_complet_pour_sft()


def test_mapper_mediqal_qcm_sans_cas_clinique():
    """`clinical_case` est `null` pour certains enregistrements ; le prompt doit alors etre juste la question."""
    enregistrement = {
        "clinical_case": None,
        "question": "Quel diagnostic evoquez-vous ?",
        "answer_a": "Autre",
        "answer_b": "Autre",
        "answer_c": "Kyste du tractus thyreoglosse",
        "answer_d": "Autre",
        "answer_e": "Autre",
        "correct_answers": "C",
        "task": "QCU",
    }
    exemple = mapper_mediqal_qcm(enregistrement)

    assert exemple is not None
    assert exemple.prompt[0].contenu == "Quel diagnostic evoquez-vous ?"
    assert exemple.completion[0].contenu == "Kyste du tractus thyreoglosse"


def test_mapper_mediqal_qcm_mcqm_concatene_les_reponses_multiples():
    """Schema reel MediQAl "mcqm" (07/09/2026) : plusieurs lettres separees par une virgule."""
    enregistrement = {
        "clinical_case": "Une fillette de 6 ans developpe une parotidite.",
        "question": "Quelle(s) proposition(s) peut (peuvent) s'appliquer a l'epidemiologie de la maladie ?",
        "answer_a": "Transmission manuportee",
        "answer_b": "La phase de contagiosite dure 8 jours",
        "answer_c": "Maladie strictement humaine",
        "answer_d": "Confere une immunite",
        "answer_e": "Transmission indirecte possible",
        "correct_answers": "C,D",
        "task": "QCM",
    }
    exemple = mapper_mediqal_qcm(enregistrement)

    assert exemple is not None
    assert exemple.completion[0].contenu == "Maladie strictement humaine Confere une immunite"


def test_mapper_mediqal_qcm_ignore_enregistrement_incomplet():
    assert mapper_mediqal_qcm({"question": "Sans reponse correcte", "answer_a": "X"}) is None
    assert mapper_mediqal_qcm({"clinical_case": None, "correct_answers": "A"}) is None
    assert mapper_mediqal_qcm({}) is None


def test_mapper_mediqal_qcm_meme_id_mais_task_different_ne_collisionne_pas():
    """
    Cas reel verifie sur les fichiers bruts : mediqal_mcqu.jsonl et
    mediqal_mcqm.jsonl partagent des valeurs de `id` identiques (les
    deux fichiers ont leur propre numerotation) ; l'identifiant
    deterministe doit rester distinct grace au `task` (QCU/QCM).
    """
    base = {
        "clinical_case": None,
        "question": "Q",
        "answer_a": "reponse",
        "answer_b": "autre",
        "answer_c": "autre",
        "answer_d": "autre",
        "answer_e": "autre",
        "correct_answers": "A",
        "id": "999",
    }
    exemple_qcu = mapper_mediqal_qcm({**base, "task": "QCU"})
    exemple_qcm = mapper_mediqal_qcm({**base, "task": "QCM"})

    assert exemple_qcu.identifiant != exemple_qcm.identifiant
    assert exemple_qcu.identifiant_source_brute == "999"
    assert exemple_qcm.identifiant_source_brute == "999"


def test_mapper_mediqal_oeq_meme_id_que_mcqu_ne_collisionne_pas():
    """
    Cas reel verifie sur les fichiers bruts : mediqal_oeq.jsonl et
    mediqal_mcqu.jsonl partagent 1492 valeurs de `id` identiques.
    """
    exemple_oeq = mapper_mediqal({"id": "1", "question": "Q1", "answer": "R1"})
    exemple_mcqu = mapper_mediqal_qcm(
        {
            "id": "1",
            "task": "QCU",
            "clinical_case": None,
            "question": "Q2",
            "answer_a": "reponse",
            "answer_b": "autre",
            "answer_c": "autre",
            "answer_d": "autre",
            "answer_e": "autre",
            "correct_answers": "A",
        }
    )
    assert exemple_oeq.identifiant != exemple_mcqu.identifiant


def test_mapper_frenchmedmcqa_valide():
    """
    Schema reel (confirme sur nthngdy/frenchmedmcqa, 07/09/2026) :
    champs plats `answer_a`..`answer_e`, PAS de champ `options`.
    `correct_answers` est un entier, index 0-based dans a..e (confirme
    via `datasets` features + verification manuelle sur des
    enregistrements reels).
    """
    enregistrement = {
        "question": "Quel est le traitement de premiere intention ?",
        "answer_a": "Paracetamol",
        "answer_b": "Ibuprofene",
        "correct_answers": 0,
        "number_correct_answers": 0,
    }
    exemple = mapper_frenchmedmcqa(enregistrement)

    assert exemple is not None
    assert exemple.source == "FrenchMedMCQA"
    assert exemple.prompt[0].contenu == "Quel est le traitement de premiere intention ?"
    assert exemple.completion[0].contenu == "Paracetamol"


def test_mapper_frenchmedmcqa_resout_index_non_nul():
    enregistrement = {
        "question": "Laquelle est fausse ?",
        "answer_a": "A",
        "answer_b": "B",
        "answer_c": "C",
        "answer_d": "D",
        "answer_e": "Peu ionisantes",
        "correct_answers": 4,
        "number_correct_answers": 0,
    }
    exemple = mapper_frenchmedmcqa(enregistrement)

    assert exemple is not None
    assert exemple.completion[0].contenu == "Peu ionisantes"


def test_mapper_frenchmedmcqa_ignore_sans_reponse_correcte():
    assert mapper_frenchmedmcqa({"question": "Q"}) is None
    assert mapper_frenchmedmcqa({"question": "Q", "correct_answers": None}) is None
    assert mapper_frenchmedmcqa({"question": "Q", "correct_answers": 0}) is None  # answer_a absent


def test_mapper_frenchmedmcqa_identifiant_trace_le_champ_id():
    enregistrement = {"id": "abc123", "question": "Q", "answer_a": "R", "correct_answers": 0}
    exemple = mapper_frenchmedmcqa(enregistrement)
    assert exemple.identifiant_source_brute == "abc123"


def test_mapper_frenchmedmcqa_deux_registres_identiques_produisent_le_meme_identifiant():
    """
    Cas reel verifie sur le fichier brut : un doublon EXACT existe
    (meme `id`, meme contenu) ; l'identifiant deterministe doit
    permettre de le detecter (meme id -> meme identifiant), pour que
    `ConstruireDatasetPivotUseCase` puisse le dedoublonner.
    """
    enregistrement = {"id": "dup-1", "question": "Q", "answer_a": "R", "correct_answers": 0}
    exemple_1 = mapper_frenchmedmcqa(dict(enregistrement))
    exemple_2 = mapper_frenchmedmcqa(dict(enregistrement))
    assert exemple_1.identifiant == exemple_2.identifiant


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


def test_mapper_medquad_identifiant_deterministe_via_hash_question_reponse():
    """Pas de champ `id` brut pour MedQuAD ; la cle naturelle est un hash de Question+Answer."""
    enregistrement = {"Question": "What is diabetes?", "Answer": "A chronic condition."}
    exemple_1 = mapper_medquad(dict(enregistrement))
    exemple_2 = mapper_medquad(dict(enregistrement))

    assert exemple_1.identifiant == exemple_2.identifiant
    assert exemple_1.identifiant_source_brute  # non vide
    assert exemple_1.identifiant_source_brute == exemple_1.identifiant_source_brute  # stable


def test_mapper_medquad_meme_question_reponse_differente_donne_un_identifiant_different():
    exemple_1 = mapper_medquad({"Question": "What is diabetes?", "Answer": "Answer A"})
    exemple_2 = mapper_medquad({"Question": "What is diabetes?", "Answer": "Answer B"})
    assert exemple_1.identifiant != exemple_2.identifiant


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


def test_mapper_ultramedical_preference_identifiant_source_brute_est_le_prompt_id():
    enregistrement = {
        "prompt_id": "MedMCQA,11404",
        "label_type": "easy",
        "prompt": "Q",
        "chosen": "C",
        "rejected": "R",
    }
    exemple = mapper_ultramedical_preference(enregistrement)
    assert exemple.identifiant_source_brute == "MedMCQA,11404"


def test_mapper_ultramedical_preference_meme_prompt_id_mais_label_type_different_ne_collisionne_pas():
    """
    Cas reel verifie sur le fichier brut : `prompt_id` seul n'est PAS
    unique (annotations multiples par prompt, ex. label_type
    "easy"/"length") ; des reponses differentes doivent produire des
    identifiants differents.
    """
    base = {"prompt_id": "MedMCQA,11404", "prompt": "Q"}
    exemple_easy = mapper_ultramedical_preference(
        {**base, "label_type": "easy", "chosen": "Reponse facile", "rejected": "Mauvaise reponse"}
    )
    exemple_length = mapper_ultramedical_preference(
        {**base, "label_type": "length", "chosen": "Reponse longue detaillee", "rejected": "Reponse courte"}
    )
    assert exemple_easy.identifiant != exemple_length.identifiant
    assert exemple_easy.identifiant_source_brute == exemple_length.identifiant_source_brute == "MedMCQA,11404"


def test_mapper_ultramedical_preference_registre_strictement_identique_donne_le_meme_identifiant():
    """Deux registres bruts identiques (vrai doublon, cas reel verifie) -> meme identifiant."""
    enregistrement = {
        "prompt_id": "MedMCQA,11404",
        "label_type": "length",
        "prompt": "Q",
        "chosen": "C",
        "rejected": "R",
    }
    exemple_1 = mapper_ultramedical_preference(dict(enregistrement))
    exemple_2 = mapper_ultramedical_preference(dict(enregistrement))
    assert exemple_1.identifiant == exemple_2.identifiant


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
