"""
Adaptateur secondaire : inference locale via `transformers`+`peft`
(modele de base en pleine precision bf16 PLUS les poids LoRA
entraines par-dessus), pour l'evaluation "post-SFT" (README §2.4) du
modele REELLEMENT entraine (`mombasstic/chsa-triage-sft-lora`, verdict
"saine", cf. AGENTS.md/README §2.3) sur le MEME sous-ensemble de 278
exemples deja evalue par les deux baselines zero-shot (§2.2), avec les
memes metriques, pour une comparaison numero-contre-numero.

Implemente le meme port `MoteurInference`, et respecte EXACTEMENT le
meme contrat `parametres["invite_deja_rendue"]` que
`TransformersInferenceAdapter` (popee avant generation) : c'est ce qui
permet a `EvaluerBaselineZeroShotUseCase` d'etre REUTILISE SANS
MODIFICATION une troisieme fois (apres CPU/GGUF puis GPU/bf16
zero-shot), seul l'adaptateur d'inference change. Le nom du cas
d'usage garde son vocabulaire "Baseline"/"ZeroShot" d'origine (Etape
1bis) : ce n'est plus litteralement une evaluation "zero-shot" ici
(le modele EST entraine), mais l'algorithme (generer une reponse par
exemple du split test, comparer a la reference, agreger) est
identique et deliberement non duplique ; voir
`interfaces/cli/E2_05_evaluer_post_sft.py` et README §2.4 pour la
justification complete de ce choix de reemploi plutot que de
renommage.

Duplique volontairement `_parametres_generation_transformers`
(import direct depuis `transformers_inference_adapter`, jamais
reecrite ici) : seule la construction du modele change entre les deux
adaptateurs (base seul vs base+LoRA), la traduction des parametres de
generation est strictement identique.

Echoue explicitement (`RuntimeError`) si aucun GPU CUDA n'est
disponible, meme raison et meme message que `TransformersInferenceAdapter` :
jamais de repli silencieux vers le CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from chsa_triage.domain.ports.moteur_inference import ReponseModele
from chsa_triage.infrastructure.adapters.transformers_inference_adapter import (
    _parametres_generation_transformers,
)

MODELE_BASE_DEFAUT = "Qwen/Qwen3-1.7B-Base"


@dataclass(slots=True)
class TransformersLoraInferenceAdapter:
    """
    Adaptateur GPU implementant `MoteurInference` via
    `transformers.AutoModelForCausalLM` (modele de base, bf16) PLUS
    `peft.PeftModel.from_pretrained` (poids LoRA charges par-dessus,
    depuis un depot HF de type modele, ex.
    `mombasstic/chsa-triage-sft-lora`).

    `_modele`/`_tokenizer` sont injectables (tests avec doubles en
    memoire, sans torch/transformers/peft/GPU reels), meme patron que
    `TransformersInferenceAdapter`. Laisses a `None` en usage normal,
    le vrai modele de base + adaptateur LoRA est charge paresseusement
    (une seule fois) au premier `generer()`.
    """

    depot_lora    : str
    nom_modele_base : str = MODELE_BASE_DEFAUT
    _modele       : Any = field(default=None, repr=False)
    _tokenizer    : Any = field(default=None, repr=False)

    def _obtenir_modele_et_tokenizer(self) -> tuple[Any, Any]:
        if self._modele is None or self._tokenizer is None:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TransformersLoraInferenceAdapter necessite un GPU CUDA : son but "
                    "est d'evaluer le modele SFT (base + LoRA) en pleine precision bf16 "
                    "(cf. TransformersInferenceAdapter pour le meme choix sur la baseline "
                    "zero-shot). Tourner sans GPU serait impraticable ; lancez ce script "
                    "sur un job HF Jobs avec GPU "
                    "(interfaces/cli/E2_05_evaluer_post_sft.py)."
                )

            # `peft`/`transformers` importes seulement ici (pas plus haut) : ce
            # sont des dependances de l'extra `remote` uniquement, jamais
            # installees en CI (qui n'installe que dev+local+api, sans GPU) --
            # importer avant la verification GPU ci-dessus faisait echouer ce
            # test en CI sur un ModuleNotFoundError au lieu du RuntimeError
            # explicite attendu.
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.nom_modele_base, trust_remote_code=True
            )
            modele_base = AutoModelForCausalLM.from_pretrained(
                self.nom_modele_base,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
            self._modele = PeftModel.from_pretrained(modele_base, self.depot_lora)
        return self._modele, self._tokenizer

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        """
        Identique a `TransformersInferenceAdapter.generer()` (memes
        deux modes selon `parametres["invite_deja_rendue"]`, meme
        mesure de `latence_ms` autour de `model.generate` uniquement) :
        seule la construction du modele (base+LoRA au lieu de base
        seul) differe, cf. `_obtenir_modele_et_tokenizer` ci-dessus.
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
            texte = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        entrees = tokenizer(texte, return_tensors="pt").to(modele.device)
        ids_entree = list(entrees["input_ids"][0])
        nombre_tokens_entree = len(ids_entree)

        kwargs_generation = _parametres_generation_transformers(parametres)

        debut = time.perf_counter()
        sortie = modele.generate(**entrees, **kwargs_generation)
        latence_ms = (time.perf_counter() - debut) * 1000

        tokens_generes = list(sortie[0])[nombre_tokens_entree:]
        texte_genere = tokenizer.decode(tokens_generes, skip_special_tokens=True)

        return ReponseModele(
            texte=texte_genere,
            nombre_tokens_entree=nombre_tokens_entree,
            nombre_tokens_sortie=len(tokens_generes),
            latence_ms=latence_ms,
            metadonnees={"modele_base": self.nom_modele_base, "depot_lora": self.depot_lora},
        )
