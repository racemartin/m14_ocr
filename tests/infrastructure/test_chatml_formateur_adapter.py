"""
Test d'integration reel de ChatMLFormateurAdapter ; pas un mock.

Verifie, sur un ExemplePivot construit a la main, que le texte rendu
par le vrai tokenizer Qwen3-1.7B-Base contient bien
`<|im_start|>assistant` et que les tokens de controle ChatML restent
atomiques : cette seconde verification reutilise directement
`verifier_chat_template` de `scripts/check_env_gpu.py` (pas de
duplication de la logique de fragmentation), deja ecrite et testee
manuellement dans le cadre de la checklist Environnement B.

Se saute automatiquement si `transformers` n'est pas installe, meme
principe que `tests/infrastructure/test_presidio_anonymiseur.py`, et
egalement si le tokenizer ne peut pas etre telecharge depuis Hugging
Face (pas de reseau/acces dans l'environnement d'execution) : cas
distingue explicitement pour ne pas confondre "code casse" et
"environnement isole du reseau".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple
from chsa_triage.infrastructure.adapters.chatml_formateur_adapter import (
    ChatMLFormateurAdapter,
)
from scripts.check_env_gpu import verifier_chat_template

NOM_MODELE = "Qwen/Qwen3-1.7B-Base"


def _tokenizer_disponible() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained(NOM_MODELE, trust_remote_code=True)
        return True
    except Exception:  # noqa: BLE001 - reseau/HF indisponible, pas un bug de code
        return False


pytestmark = pytest.mark.skipif(
    not _tokenizer_disponible(),
    reason=f"Tokenizer {NOM_MODELE} non accessible (pas de reseau/acces Hugging Face "
    "dans cet environnement) : a rejouer dans un environnement avec acces reseau.",
)


def _exemple_pivot() -> ExemplePivot:
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test_e2", "cas-01"),
        source="Test",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        prompt=(Message(role="user", contenu="Le patient presente une fievre a 39."),),
        completion=(Message(role="assistant", contenu="Suspicion d'infection, a surveiller."),),
    )


def test_formater_produit_un_texte_chatml_avec_tour_assistant():
    adaptateur = ChatMLFormateurAdapter(nom_modele=NOM_MODELE)
    exemple = _exemple_pivot()

    resultat = adaptateur.formater(exemple)

    assert resultat.identifiant == exemple.identifiant
    assert "<|im_start|>assistant" in resultat.texte
    assert "Suspicion d'infection" in resultat.texte


def test_tokens_controle_chatml_restent_atomiques():
    assert verifier_chat_template(NOM_MODELE) is True


def _exemple_pivot_dpo() -> ExemplePivot:
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test_e3", "cas-01"),
        source="Test",
        type_exemple=TypeExemple.DPO,
        langue=Langue.FRANCAIS,
        prompt=(Message(role="user", contenu="Le patient presente une douleur thoracique."),),
        chosen=(
            Message(
                role="assistant",
                contenu='<think>Douleur thoracique, risque cardiaque.</think>'
                '{"niveau": 2, "categorie": "cardio-vasculaire", "ressources_estimees": "ECG"}',
            ),
        ),
        rejected=(Message(role="assistant", contenu="Ce n'est probablement rien de grave."),),
    )


def test_formater_preference_rend_un_triplet_texte_distinct():
    """
    Bug reel CORRIGE le 19/09/2026 (cf. AGENTS.md, docstring de
    `ChatMLFormateurAdapter.formater_preference()`) : l'ancienne
    implementation passait par `apply_chat_template()` sur le tour
    assistant isole, ce qui faisait retirer le bloc
    `<think>...</think>` par le chat template natif de Qwen3-1.7B-Base
    (traite comme un tour "non final" en l'absence de tour `user`
    precedent dans la liste passee). La correction construit
    `texte_chosen`/`texte_rejected` directement avec les tokens de
    controle reels du template plutot que via `apply_chat_template()`
    pour ce champ precis (`_rendre_tour_assistant_seul`) : `<think>`
    survit desormais, verifie ici par assertion inversee (documentait
    le bug avant, documente la correction maintenant).
    """
    adaptateur = ChatMLFormateurAdapter(nom_modele=NOM_MODELE)
    exemple = _exemple_pivot_dpo()

    resultat = adaptateur.formater_preference(exemple)

    assert resultat.identifiant == exemple.identifiant
    assert "Le patient presente une douleur thoracique" in resultat.texte_prompt
    assert '"niveau": 2' in resultat.texte_chosen
    assert "<think>Douleur thoracique, risque cardiaque.</think>" in resultat.texte_chosen
    assert "Ce n'est probablement rien de grave" in resultat.texte_rejected
    assert '"niveau": 2' not in resultat.texte_rejected
    assert "Ce n'est probablement rien de grave" not in resultat.texte_prompt


def test_formater_preference_texte_prompt_ne_contient_jamais_le_tour_assistant():
    adaptateur = ChatMLFormateurAdapter(nom_modele=NOM_MODELE)
    exemple = _exemple_pivot_dpo()

    resultat = adaptateur.formater_preference(exemple)

    assert "<|im_start|>assistant" not in resultat.texte_prompt or resultat.texte_prompt.rstrip().endswith(
        "<|im_start|>assistant"
    )
    assert '"niveau"' not in resultat.texte_prompt
    assert resultat.texte_prompt.rstrip().endswith("<|im_start|>assistant")


def test_formater_invite_zero_shot_ne_contient_pas_la_completion():
    """
    Oppose de `formater()` : utilise par l'evaluation baseline zero-shot
    (Etape 1bis, `E1_06_00_evaluer_baseline_zero_shot.py`). Doit rendre
    UNIQUEMENT le prompt (jamais la completion), et se terminer par le
    tour assistant vide (`add_generation_prompt=True`), pret pour la
    generation.
    """
    adaptateur = ChatMLFormateurAdapter(nom_modele=NOM_MODELE)
    exemple = _exemple_pivot()

    texte = adaptateur.formater_invite_zero_shot(exemple)

    assert "Le patient presente une fievre" in texte
    assert "Suspicion d'infection" not in texte
    assert texte.rstrip().endswith("<|im_start|>assistant")
