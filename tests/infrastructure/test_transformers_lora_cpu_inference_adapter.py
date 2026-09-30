"""
Tests de TransformersLoraCpuInferenceAdapter.

Unitaires uniquement (toujours executes, sans reseau ni torch/transformers/
peft reels) : meme patron que `test_transformers_lora_inference_adapter.py`
(faux modele/tokenizer injectes via `_modele`/`_tokenizer`). Ne reteste PAS
`_parametres_generation_transformers` (deja teste dans
`test_transformers_inference_adapter.py`, reutilisee ici telle quelle).

Contrairement a `test_transformers_lora_inference_adapter.py`, il n'existe
pas de test "sans GPU leve RuntimeError" ici : c'est precisement le
comportement inverse qui est attendu (jamais de RuntimeError pour absence
de GPU, cet adaptateur est le chemin CPU assume). Le veritable chargement
(bf16 puis repli float32) touche torch/transformers/peft reels et n'est
donc pas exerce par ces tests, memes limites que
`test_transformers_lora_inference_adapter.py` (aucun run GPU/CPU reel ici,
cf. AGENTS.md).
"""

from __future__ import annotations

import pytest

from chsa_triage.infrastructure.adapters.transformers_lora_cpu_inference_adapter import (
    TransformersLoraCpuInferenceAdapter,
)

DEPOT_LORA_TEST = "mombasstic/chsa-triage-dpo-lora"


class FauxEntrees(dict):
    """Faux retour de `tokenizer(texte, return_tensors="pt")` : dict-like, avec `.to(device)`."""

    def to(self, device):
        return self


class FauxTokenizer:
    def __init__(self, ids_entree: list[int], texte_decode: str = "reponse generee") -> None:
        self._ids_entree = ids_entree
        self._texte_decode = texte_decode
        self.dernier_appel_chat_template: dict | None = None

    def __call__(self, texte: str, return_tensors: str) -> FauxEntrees:
        return FauxEntrees({"input_ids": [list(self._ids_entree)]})

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.dernier_appel_chat_template = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n"

    def decode(self, tokens, skip_special_tokens):
        return self._texte_decode


class FauxModele:
    def __init__(self, tokens_generes: list[int]) -> None:
        self.device = "cpu"
        self._tokens_generes = tokens_generes
        self.dernier_appel_generate: dict | None = None

    def generate(self, **kwargs):
        self.dernier_appel_generate = kwargs
        ids_entree = kwargs["input_ids"][0]
        return [list(ids_entree) + list(self._tokens_generes)]


def _construire_adaptateur(ids_entree=(1, 2, 3), tokens_generes=(4, 5)):
    modele = FauxModele(list(tokens_generes))
    tokenizer = FauxTokenizer(list(ids_entree))
    adaptateur = TransformersLoraCpuInferenceAdapter(
        depot_lora=DEPOT_LORA_TEST, _modele=modele, _tokenizer=tokenizer
    )
    return adaptateur, modele, tokenizer


def test_generer_mode_invite_deja_rendue_ne_reapplique_pas_le_chat_template():
    adaptateur, modele, tokenizer = _construire_adaptateur(ids_entree=(1, 2, 3), tokens_generes=(4, 5, 6))
    invite = "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n"

    reponse = adaptateur.generer([{"role": "user", "content": invite}], {"invite_deja_rendue": True, "n_predict": 5})

    assert tokenizer.dernier_appel_chat_template is None
    assert reponse.texte == "reponse generee"
    assert reponse.nombre_tokens_entree == 3
    assert reponse.nombre_tokens_sortie == 3
    assert reponse.latence_ms >= 0.0


def test_generer_mode_par_defaut_applique_le_chat_template_avec_generation_prompt():
    adaptateur, modele, tokenizer = _construire_adaptateur()
    messages = [{"role": "user", "content": "Bonjour"}]

    adaptateur.generer(messages, {"temperature": 0.0})

    assert tokenizer.dernier_appel_chat_template == {
        "messages": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_generer_mode_invite_deja_rendue_leve_si_aucun_message():
    adaptateur, _, _ = _construire_adaptateur()
    with pytest.raises(ValueError):
        adaptateur.generer([], {"invite_deja_rendue": True})


def test_parametre_invite_deja_rendue_jamais_transmis_a_generate():
    adaptateur, modele, _ = _construire_adaptateur()

    adaptateur.generer([{"role": "user", "content": "x"}], {"invite_deja_rendue": True, "temperature": 0.5})

    assert "invite_deja_rendue" not in modele.dernier_appel_generate


def test_reponse_reference_le_modele_de_base_et_le_depot_lora():
    adaptateur, _, _ = _construire_adaptateur()

    reponse = adaptateur.generer([{"role": "user", "content": "x"}], {"invite_deja_rendue": True})

    assert reponse.metadonnees["modele_base"] == adaptateur.nom_modele_base
    assert reponse.metadonnees["depot_lora"] == DEPOT_LORA_TEST


def test_generer_utilise_le_modele_et_tokenizer_injectes_sans_charger_de_vrais():
    """Aucun import torch/transformers/peft reel declenche : _modele/_tokenizer deja fournis."""
    adaptateur, modele, tokenizer = _construire_adaptateur()

    adaptateur.generer([{"role": "user", "content": "x"}], {"invite_deja_rendue": True})

    assert adaptateur._modele is modele
    assert adaptateur._tokenizer is tokenizer
