"""
Tests de TransformersInferenceAdapter.

Unitaires uniquement (toujours executes, sans reseau ni GPU reel) : un
faux modele/tokenizer injectes via `_modele`/`_tokenizer` (meme patron
que `_client` sur `LlamaCppInferenceAdapter`), verifient le mapping
invite -> `ReponseModele` pour les deux modes (`invite_deja_rendue`
True/False) et la traduction des parametres de generation, sans
importer torch/transformers pour de vrai : ces doubles ne modelisent
PAS de vrais tensors, seulement l'interface (dict-like + listes) dont
`generer()` a besoin.

Il n'existe PAS de suite d'integration reelle opt-in ici (contrairement
a `test_llamacpp_inference_adapter.py`) : aucun GPU CUDA n'est
disponible dans cet environnement de developpement, cf. AGENTS.md.
"""

from __future__ import annotations

import pytest

from chsa_triage.infrastructure.adapters.transformers_inference_adapter import (
    TransformersInferenceAdapter,
    _parametres_generation_transformers,
)


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
        self.device = "cuda:0"
        self._tokens_generes = tokens_generes
        self.dernier_appel_generate: dict | None = None

    def generate(self, **kwargs):
        self.dernier_appel_generate = kwargs
        ids_entree = kwargs["input_ids"][0]
        return [list(ids_entree) + list(self._tokens_generes)]


def _construire_adaptateur(ids_entree=(1, 2, 3), tokens_generes=(4, 5)):
    modele = FauxModele(list(tokens_generes))
    tokenizer = FauxTokenizer(list(ids_entree))
    return TransformersInferenceAdapter(_modele=modele, _tokenizer=tokenizer), modele, tokenizer


def test_generer_mode_invite_deja_rendue_ne_reapplique_pas_le_chat_template():
    adaptateur, modele, tokenizer = _construire_adaptateur(ids_entree=(1, 2, 3), tokens_generes=(4, 5, 6))
    invite = "<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n"

    reponse = adaptateur.generer([{"role": "user", "content": invite}], {"invite_deja_rendue": True, "n_predict": 5})

    assert tokenizer.dernier_appel_chat_template is None
    assert reponse.texte == "reponse generee"
    assert reponse.nombre_tokens_entree == 3
    assert reponse.nombre_tokens_sortie == 3
    assert reponse.latence_ms >= 0.0
    assert reponse.metadonnees["modele"] == adaptateur.nom_modele


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


def test_parametres_generation_transformers_traduit_n_predict_et_temperature_nulle():
    resultat = _parametres_generation_transformers({"n_predict": 5, "temperature": 0.0})

    assert resultat == {"max_new_tokens": 5, "do_sample": False}


def test_parametres_generation_transformers_temperature_positive_active_do_sample():
    resultat = _parametres_generation_transformers({"n_predict": 10, "temperature": 0.7})

    assert resultat == {"max_new_tokens": 10, "do_sample": True, "temperature": 0.7}


def test_parametres_generation_transformers_defaut_sans_n_predict():
    resultat = _parametres_generation_transformers({})

    assert resultat["max_new_tokens"] == 256


def test_parametres_generation_transformers_passe_les_cles_inconnues_telles_quelles():
    resultat = _parametres_generation_transformers({"n_predict": 5, "top_p": 0.9})

    assert resultat["top_p"] == 0.9


def test_generer_utilise_le_modele_et_tokenizer_injectes_sans_charger_de_vrais():
    """Aucun import torch/transformers reel declenche : _modele/_tokenizer deja fournis."""
    adaptateur, modele, tokenizer = _construire_adaptateur()

    adaptateur.generer([{"role": "user", "content": "x"}], {"invite_deja_rendue": True})

    assert adaptateur._modele is modele
    assert adaptateur._tokenizer is tokenizer


def test_generer_sans_gpu_leve_runtime_error_explicite():
    """
    Sans _modele/_tokenizer injectes, le vrai chemin de chargement est
    emprunte : torch est une dependance de base de ce projet (toujours
    installee, cf. pyproject.toml), donc ce test s'execute reellement
    ici, sans double. Verifie qu'AUCUN repli silencieux vers le CPU
    n'existe : sur cet environnement de developpement (confirme sans
    GPU CUDA, cf. AGENTS.md), l'appel doit echouer explicitement plutot
    que de charger le modele en pleine precision sur CPU.
    """
    import torch

    if torch.cuda.is_available():
        pytest.skip("GPU CUDA reellement disponible dans cet environnement : rien a verifier ici")

    adaptateur = TransformersInferenceAdapter()

    with pytest.raises(RuntimeError, match="GPU CUDA"):
        adaptateur.generer([{"role": "user", "content": "x"}], {"invite_deja_rendue": True})
