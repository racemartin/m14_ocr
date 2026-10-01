"""
Adaptateur secondaire : inference locale via `transformers` (pleine
precision bf16, SANS quantification), pour evaluer la baseline
zero-shot sur GPU reel sans melanger l'effet de la quantification GGUF
avec celui de l'entrainement. Meme contrat `invite_deja_rendue` que
`LlamaCppInferenceAdapter`, ce qui permet a `EvaluerBaselineZeroShotUseCase`
d'etre reutilise sans modification entre les deux.

Sans `invite_deja_rendue`, le chat template est applique ici avec le
MEME tokenizer que celui charge pour la generation (contrairement au
mode HTTP de llama.cpp, qui delegue au chat template embarque dans le
GGUF, potentiellement different).

Echoue explicitement (`RuntimeError`) sans GPU CUDA : cet adaptateur
existe pour eviter la quantification, jamais un repli silencieux vers
un mode plus lent que le baseline CPU deja existant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from chsa_triage.domain.ports.moteur_inference import ReponseModele

MODELE_DEFAUT = "Qwen/Qwen3-1.7B-Base"
NOMBRE_TOKENS_GENERES_DEFAUT = 256


def _parametres_generation_transformers(parametres: dict) -> dict:
    """
    Traduit les cles de `parametres_generation` deja utilisees par
    `EvaluerBaselineZeroShotUseCase`/`E1_06_00_evaluer_baseline.py`
    (vocabulaire llama.cpp : `n_predict`, `temperature`) vers les
    kwargs reels de `GenerationMixin.generate` (`max_new_tokens`,
    `do_sample`/`temperature`). `temperature<=0.0` -> generation
    deterministe (`do_sample=False`), meme convention que le CLI local
    (`--temperature 0.0` par defaut). Toute autre cle (deja au
    vocabulaire transformers, ex. `top_p`) est transmise telle quelle,
    SAUF `stop`/`guided_regex` : hints de decodage contraint propres au
    serveur vLLM (cf. E4_00/E4_01_uc_*.py), sans equivalent natif dans
    `GenerationMixin.generate` -- jamais testes contre un `generate()`
    reel, donc explicitement ecartes ici plutot que transmis en
    aveugle.
    """
    parametres = dict(parametres)
    resultat: dict[str, Any] = {
        "max_new_tokens": parametres.pop(
            "n_predict", NOMBRE_TOKENS_GENERES_DEFAUT
        )
    }
    temperature = parametres.pop("temperature", None)
    if temperature is not None:
        if temperature <= 0.0:
            resultat["do_sample"] = False
        else:
            resultat["do_sample"] = True
            resultat["temperature"] = temperature
    parametres.pop("stop", None)
    parametres.pop("guided_regex", None)
    resultat.update(parametres)
    return resultat


@dataclass(slots=True)
class TransformersInferenceAdapter:
    """
    Adaptateur GPU implementant `MoteurInference` via
    `transformers.AutoModelForCausalLM`/`AutoTokenizer` charges EN
    PROCESSUS (pas de serveur HTTP externe, contrairement a
    `LlamaCppInferenceAdapter`).

    `_modele`/`_tokenizer` sont injectables (utilises par les tests avec
    des doubles en memoire, sans torch/transformers/GPU reels) ; laisses
    a `None` en usage normal, le vrai modele/tokenizer est charge
    paresseusement (une seule fois) au premier `generer()`.
    """

    nom_modele: str = MODELE_DEFAUT
    _modele: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    def _obtenir_modele_et_tokenizer(self) -> tuple[Any, Any]:
        if self._modele is None or self._tokenizer is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TransformersInferenceAdapter necessite un GPU CUDA : son but "
                    "est d'evaluer en pleine precision bf16, SANS quantification "
                    "(contrairement a LlamaCppInferenceAdapter/GGUF). Tourner sans "
                    "GPU serait encore plus lent que le baseline local llama.cpp "
                    "deja existant (cf. AGENTS.md) ; utilisez ce dernier en local, "
                    "ou lancez ce script sur un job HF Jobs avec GPU "
                    "(interfaces/cli/E1_06_01_evaluer_baseline_gpu.py)."
                )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.nom_modele, trust_remote_code=True
            )
            self._modele = AutoModelForCausalLM.from_pretrained(
                self.nom_modele,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
        return self._modele, self._tokenizer

    def generer(
        self, messages: list[dict], parametres: dict | None = None
    ) -> ReponseModele:
        """
        `latence_ms` mesure exactement le temps de `model.generate`
        (`time.perf_counter()` autour de l'appel), pas le chargement du
        modele ni la tokenisation. `parametres["invite_deja_rendue"]`
        (defaut `False`) bascule entre les deux modes decrits dans le
        docstring du module ; retiree de `parametres` avant transmission
        a `generate` (ce n'est pas un parametre de generation reel).
        """
        parametres = dict(parametres or {})
        invite_deja_rendue = parametres.pop("invite_deja_rendue", False)
        modele, tokenizer = self._obtenir_modele_et_tokenizer()

        if invite_deja_rendue:
            if not messages:
                raise ValueError(
                    "invite_deja_rendue=True necessite au moins un message (l'invite deja rendue)"
                )
            texte = messages[-1]["content"]
        else:
            texte = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        entrees = tokenizer(texte, return_tensors="pt").to(modele.device)
        ids_entree = list(entrees["input_ids"][0])
        nombre_tokens_entree = len(ids_entree)

        kwargs_generation = _parametres_generation_transformers(parametres)

        debut = time.perf_counter()
        sortie = modele.generate(**entrees, **kwargs_generation)
        latence_ms = (time.perf_counter() - debut) * 1000

        tokens_generes = list(sortie[0])[nombre_tokens_entree:]
        texte_genere = tokenizer.decode(
            tokens_generes, skip_special_tokens=True
        )

        return ReponseModele(
            texte=texte_genere,
            nombre_tokens_entree=nombre_tokens_entree,
            nombre_tokens_sortie=len(tokens_generes),
            latence_ms=latence_ms,
            metadonnees={"modele": self.nom_modele},
        )
