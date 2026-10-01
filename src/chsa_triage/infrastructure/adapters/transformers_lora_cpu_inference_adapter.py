"""
Adaptateur tertiaire : inference locale via `transformers`+`peft`
(modele de base PLUS les poids LoRA entraines par-dessus), sur CPU,
SANS aucun serveur separe a lancer (le modele est charge en processus,
meme famille que `TransformersLoraInferenceAdapter`).

Existe UNIQUEMENT comme outil de comparaison de precision hors-ligne
(README, section "Comparaison de precision hors-ligne") : le but est
de comparer une reponse en pleine precision a la reponse deja obtenue
via le chemin quantifie CPU existant (llama.cpp/GGUF Q4_K_M,
`LlamaCppInferenceAdapter`), pour voir si la quantification change le
comportement du modele. Ce n'est PAS un troisieme chemin de service :
il n'est jamais branche dans le frontend Streamlit
(`interfaces/web/app_test_inference.py`), et son mode
`CHSA_MOTEUR_INFERENCE` dedie (cf. `interfaces/api/main.py`) est
documente comme un outil de comparaison, jamais comme une troisieme
option de deploiement au meme titre que `local`/`distant`. Volontairement
lent (CPU, pleine precision, aucune optimisation) : ne jamais l'utiliser
pour du chat en direct.

Implemente le meme port `MoteurInference` et respecte EXACTEMENT le
meme contrat `parametres["invite_deja_rendue"]` que
`TransformersLoraInferenceAdapter` (popee avant generation). Reutilise
`_parametres_generation_transformers` (import direct depuis
`transformers_inference_adapter`, jamais reecrite ici), meme
discipline que `TransformersLoraInferenceAdapter`.

Ne leve JAMAIS de `RuntimeError` pour absence de GPU (a la difference
de `TransformersLoraInferenceAdapter`, qui refuse explicitement tout
repli CPU) : ce chemin EST le repli CPU assume, deliberement separe
plutot que d'affaiblir la garde GPU de l'autre adaptateur.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from tools.rafael.log_tool import LogTool

from chsa_triage.domain.ports.moteur_inference import ReponseModele
from chsa_triage.infrastructure.adapters.transformers_inference_adapter import (
    _parametres_generation_transformers,
)

MODELE_BASE_DEFAUT = "Qwen/Qwen3-1.7B-Base"

log = LogTool(origin="comparaison_precision_cpu")


@dataclass(slots=True)
class TransformersLoraCpuInferenceAdapter:
    """
    Adaptateur CPU implementant `MoteurInference` via
    `transformers.AutoModelForCausalLM` (modele de base) PLUS
    `peft.PeftModel.from_pretrained` (poids LoRA charges par-dessus,
    depuis un depot HF de type modele, ex.
    `mombasstic/chsa-triage-dpo-lora`), le tout charge sur CPU
    (`device_map="cpu"`).

    `_modele`/`_tokenizer` sont injectables (tests avec doubles en
    memoire, sans torch/transformers/peft reels), meme patron que
    `TransformersLoraInferenceAdapter`. Laisses a `None` en usage
    normal, le vrai modele de base + adaptateur LoRA est charge
    paresseusement (une seule fois) au premier `generer()`.

    Bug reel observe en usage (30/09/2026) : `generer()` n'etait pas
    serialise, et deux requetes arrivees proches l'une de l'autre (ex.
    une requete dont la connexion client avait ete perdue, mais dont le
    traitement serveur continuait, suivie d'une relance manuelle)
    entraient TOUTES LES DEUX dans le chargement paresseux avant que la
    premiere ait fini d'assigner `self._modele` - chargement du modele
    en double (explique la memoire proche de 100% et le swap observes),
    et generation concurrente sur la meme instance de modele (jamais
    garantie thread-safe par `transformers`), signature coherente avec
    la sortie degeneree observee (boucle du meme caractere). Corrige en
    serialisant tout `generer()` (pas seulement le chargement paresseux)
    derriere `_verrou` : cet adaptateur est un outil de comparaison
    ponctuelle, jamais pense pour plusieurs requetes concurrentes, donc
    mettre la seconde requete en attente de la premiere est le
    comportement correct, pas une limitation a lever plus tard.
    """

    depot_lora      : str
    nom_modele_base : str = MODELE_BASE_DEFAUT
    _modele         : Any = field(default=None, repr=False)
    _tokenizer      : Any = field(default=None, repr=False)
    _verrou         : threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def _obtenir_modele_et_tokenizer(self) -> tuple[Any, Any]:
        if self._modele is None or self._tokenizer is None:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.START_ACTION(
                "TransformersLoraCpuInferenceAdapter",
                "_obtenir_modele_et_tokenizer",
                "telechargement/chargement du modele (une seule fois par processus)",
            )
            log.PARAMETER_VALUE("modele_base", self.nom_modele_base)
            log.PARAMETER_VALUE("depot_lora", self.depot_lora)

            log.STEP(1, "STEP 1 Tokenizer", self.nom_modele_base)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.nom_modele_base, trust_remote_code=True
            )

            log.STEP(
                1,
                "STEP 2 Modele de base",
                "float32",
            )
            # float32 directement, PLUS de tentative bf16 (changement du
            # 01/10/2026, mesure reelle a l'appui) : bf16 chargeait sans
            # erreur sur ce CPU (donc le repli try/except ci-dessous
            # n'etait jamais declenche), mais la generation elle-meme etait
            # catastrophiquement lente - un run reel a mesure ~145s/token
            # (28 tokens en 4079s), coherent avec une emulation logicielle
            # de bf16 sur un CPU x86 sans acceleration materielle dediee.
            # float32 est nativement supporte par tout CPU (aucune
            # emulation), donc attendu nettement plus rapide malgre le
            # double de memoire occupee par les poids.
            modele_base = AutoModelForCausalLM.from_pretrained(
                self.nom_modele_base,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
            dtype_utilise = "float32"
            log.PARAMETER_VALUE("dtype retenu", dtype_utilise)

            log.STEP(1, "STEP 3 Adaptateur LoRA", self.depot_lora)
            self._modele = PeftModel.from_pretrained(modele_base, self.depot_lora)

            log.FINISH_ACTION(
                "TransformersLoraCpuInferenceAdapter",
                "_obtenir_modele_et_tokenizer",
                f"pret (dtype={dtype_utilise})",
            )
        return self._modele, self._tokenizer

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        """
        Identique a `TransformersLoraInferenceAdapter.generer()` (memes
        deux modes selon `parametres["invite_deja_rendue"]`, meme
        mesure de `latence_ms` autour de `model.generate` uniquement) :
        seule la construction du modele (CPU au lieu de GPU) differe,
        cf. `_obtenir_modele_et_tokenizer` ci-dessus. Latence CPU
        attendue nettement plus elevee que le chemin GPU, sans gravite
        ici puisque cet adaptateur n'est jamais utilise pour du chat en
        direct.

        Serialise via `_verrou` (toute la methode, pas seulement le
        chargement paresseux) : deux requetes concurrentes partageant
        la meme instance de modele ont deja produit, en usage reel, un
        double chargement et une sortie degeneree (cf. docstring de
        classe). La seconde requete attend simplement la premiere.
        """
        with self._verrou:
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

            log.START_ACTION(
                "TransformersLoraCpuInferenceAdapter",
                "generer",
                "generation (CPU, pleine precision, aucune optimisation)",
            )
            log.PARAMETER_VALUE("tokens d'entree", nombre_tokens_entree)
            debut = time.perf_counter()
            sortie = modele.generate(**entrees, **kwargs_generation)
            latence_ms = (time.perf_counter() - debut) * 1000

            tokens_generes = list(sortie[0])[nombre_tokens_entree:]
            texte_genere = tokenizer.decode(tokens_generes, skip_special_tokens=True)
            log.PARAMETER_VALUE("tokens generes", len(tokens_generes))
            log.FINISH_ACTION(
                "TransformersLoraCpuInferenceAdapter",
                "generer",
                f"{len(tokens_generes)} tokens en {latence_ms / 1000:.1f}s",
            )

            return ReponseModele(
                texte=texte_genere,
                nombre_tokens_entree=nombre_tokens_entree,
                nombre_tokens_sortie=len(tokens_generes),
                latence_ms=latence_ms,
                metadonnees={"modele_base": self.nom_modele_base, "depot_lora": self.depot_lora},
            )
