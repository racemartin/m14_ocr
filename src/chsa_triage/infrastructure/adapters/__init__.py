"""
Point d'entree du sous-paquet infrastructure.adapters.

IMPORTANT : les imports sont PARESSEUX (via __getattr__, PEP 562).
Chaque adaptateur a ses propres dependances externes (pandas, HF
`datasets`, Presidio, ydata-profiling...) qui ne sont pas toutes
installees dans tous les environnements (ex. Environnement A local
n'a pas besoin des dependances GPU). Charger `infrastructure.adapters`
ne doit donc JAMAIS forcer l'import de tous les adaptateurs : seul
l'adaptateur reellement utilise doit etre importe, et donc seule sa
dependance externe doit etre presente.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - uniquement pour les outils de typage
    from chsa_triage.infrastructure.adapters.chatml_formateur_adapter import (
        ChatMLFormateurAdapter,
    )
    from chsa_triage.infrastructure.adapters.hf_dataset_suivi_experimentation import (
        HfDatasetSuiviExperimentation,
    )
    from chsa_triage.infrastructure.adapters.jsonl_checkpoint_repository import (
        JsonlCheckpointRepository,
    )
    from chsa_triage.infrastructure.adapters.jsonl_dataset_repository import (
        JsonlDatasetRepository,
    )
    from chsa_triage.infrastructure.adapters.jsonl_decisions_revision_humaine import (
        JsonlDecisionsRevisionHumaine,
    )
    from chsa_triage.infrastructure.adapters.jsonl_exemple_formate_preference_repository import (
        JsonlExempleFormatePreferenceRepository,
    )
    from chsa_triage.infrastructure.adapters.jsonl_exemple_formate_repository import (
        JsonlExempleFormateRepository,
    )
    from chsa_triage.infrastructure.adapters.jsonl_preference_reformulee_repository import (
        JsonlPreferenceReformuleeRepository,
    )
    from chsa_triage.infrastructure.adapters.jsonl_registre_echantillons_controle_qualite import (
        JsonlRegistreEchantillonsControleQualite,
    )
    from chsa_triage.infrastructure.adapters.lecteur_corpus_fichier_local import (
        LecteurCorpusFichierLocal,
    )
    from chsa_triage.infrastructure.adapters.lecteur_corpus_huggingface import (
        LecteurCorpusHuggingFace,
    )
    from chsa_triage.infrastructure.adapters.llamacpp_inference_adapter import (
        LlamaCppInferenceAdapter,
    )
    from chsa_triage.infrastructure.adapters.mlflow_suivi_experimentation import (
        MlflowSuiviExperimentation,
    )
    from chsa_triage.infrastructure.adapters.presidio_anonymiseur import (
        PresidioAnonymiseur,
    )
    from chsa_triage.infrastructure.adapters.spacy_verificateur_entites import (
        SpacyVerificateurEntitesNommees,
    )
    from chsa_triage.infrastructure.adapters.tensorboard_suivi_experimentation import (
        TensorboardSuiviExperimentation,
    )
    from chsa_triage.infrastructure.adapters.transformers_inference_adapter import (
        TransformersInferenceAdapter,
    )
    from chsa_triage.infrastructure.adapters.transformers_lora_inference_adapter import (
        TransformersLoraInferenceAdapter,
    )
    from chsa_triage.infrastructure.adapters.trl_dpo_entraineur import (
        TrlDpoEntraineurAdapter,
    )
    from chsa_triage.infrastructure.adapters.trl_sft_entraineur import (
        TrlSftEntraineurAdapter,
    )
    from chsa_triage.infrastructure.adapters.vllm_endpoint_inference_adapter import (
        VllmEndpointInferenceAdapter,
    )
    from chsa_triage.infrastructure.adapters.ydata_profileur import (
        YdataProfileur,
    )

__all__ = [
    "ChatMLFormateurAdapter",
    "HfDatasetSuiviExperimentation",
    "JsonlCheckpointRepository",
    "JsonlDatasetRepository",
    "JsonlDecisionsRevisionHumaine",
    "JsonlExempleFormatePreferenceRepository",
    "JsonlExempleFormateRepository",
    "JsonlPreferenceReformuleeRepository",
    "JsonlRegistreEchantillonsControleQualite",
    "LecteurCorpusFichierLocal",
    "LecteurCorpusHuggingFace",
    "LlamaCppInferenceAdapter",
    "MlflowSuiviExperimentation",
    "PresidioAnonymiseur",
    "SpacyVerificateurEntitesNommees",
    "TensorboardSuiviExperimentation",
    "TransformersInferenceAdapter",
    "TransformersLoraInferenceAdapter",
    "TrlDpoEntraineurAdapter",
    "TrlSftEntraineurAdapter",
    "VllmEndpointInferenceAdapter",
    "YdataProfileur",
]

# Correspondance nom expose -> (module, attribut) pour le chargement paresseux.
_CARTE_IMPORTS_PARESSEUX = {
    "ChatMLFormateurAdapter": (
        "chsa_triage.infrastructure.adapters.chatml_formateur_adapter",
        "ChatMLFormateurAdapter",
    ),
    "HfDatasetSuiviExperimentation": (
        "chsa_triage.infrastructure.adapters.hf_dataset_suivi_experimentation",
        "HfDatasetSuiviExperimentation",
    ),
    "JsonlCheckpointRepository": (
        "chsa_triage.infrastructure.adapters.jsonl_checkpoint_repository",
        "JsonlCheckpointRepository",
    ),
    "JsonlDatasetRepository": (
        "chsa_triage.infrastructure.adapters.jsonl_dataset_repository",
        "JsonlDatasetRepository",
    ),
    "JsonlDecisionsRevisionHumaine": (
        "chsa_triage.infrastructure.adapters.jsonl_decisions_revision_humaine",
        "JsonlDecisionsRevisionHumaine",
    ),
    "JsonlExempleFormatePreferenceRepository": (
        "chsa_triage.infrastructure.adapters.jsonl_exemple_formate_preference_repository",
        "JsonlExempleFormatePreferenceRepository",
    ),
    "JsonlExempleFormateRepository": (
        "chsa_triage.infrastructure.adapters.jsonl_exemple_formate_repository",
        "JsonlExempleFormateRepository",
    ),
    "JsonlPreferenceReformuleeRepository": (
        "chsa_triage.infrastructure.adapters.jsonl_preference_reformulee_repository",
        "JsonlPreferenceReformuleeRepository",
    ),
    "JsonlRegistreEchantillonsControleQualite": (
        "chsa_triage.infrastructure.adapters.jsonl_registre_echantillons_controle_qualite",
        "JsonlRegistreEchantillonsControleQualite",
    ),
    "LecteurCorpusFichierLocal": (
        "chsa_triage.infrastructure.adapters.lecteur_corpus_fichier_local",
        "LecteurCorpusFichierLocal",
    ),
    "LecteurCorpusHuggingFace": (
        "chsa_triage.infrastructure.adapters.lecteur_corpus_huggingface",
        "LecteurCorpusHuggingFace",
    ),
    "LlamaCppInferenceAdapter": (
        "chsa_triage.infrastructure.adapters.llamacpp_inference_adapter",
        "LlamaCppInferenceAdapter",
    ),
    "MlflowSuiviExperimentation": (
        "chsa_triage.infrastructure.adapters.mlflow_suivi_experimentation",
        "MlflowSuiviExperimentation",
    ),
    "PresidioAnonymiseur": (
        "chsa_triage.infrastructure.adapters.presidio_anonymiseur",
        "PresidioAnonymiseur",
    ),
    "SpacyVerificateurEntitesNommees": (
        "chsa_triage.infrastructure.adapters.spacy_verificateur_entites",
        "SpacyVerificateurEntitesNommees",
    ),
    "TensorboardSuiviExperimentation": (
        "chsa_triage.infrastructure.adapters.tensorboard_suivi_experimentation",
        "TensorboardSuiviExperimentation",
    ),
    "TransformersInferenceAdapter": (
        "chsa_triage.infrastructure.adapters.transformers_inference_adapter",
        "TransformersInferenceAdapter",
    ),
    "TransformersLoraInferenceAdapter": (
        "chsa_triage.infrastructure.adapters.transformers_lora_inference_adapter",
        "TransformersLoraInferenceAdapter",
    ),
    "TrlDpoEntraineurAdapter": (
        "chsa_triage.infrastructure.adapters.trl_dpo_entraineur",
        "TrlDpoEntraineurAdapter",
    ),
    "TrlSftEntraineurAdapter": (
        "chsa_triage.infrastructure.adapters.trl_sft_entraineur",
        "TrlSftEntraineurAdapter",
    ),
    "VllmEndpointInferenceAdapter": (
        "chsa_triage.infrastructure.adapters.vllm_endpoint_inference_adapter",
        "VllmEndpointInferenceAdapter",
    ),
    "YdataProfileur": (
        "chsa_triage.infrastructure.adapters.ydata_profileur",
        "YdataProfileur",
    ),
}


def __getattr__(nom: str):
    """Charge l'adaptateur demande a la volee (PEP 562)."""
    import importlib

    if nom not in _CARTE_IMPORTS_PARESSEUX:
        raise AttributeError(f"module {__name__!r} n'a pas d'attribut {nom!r}")

    nom_module, nom_attribut = _CARTE_IMPORTS_PARESSEUX[nom]
    module = importlib.import_module(nom_module)
    return getattr(module, nom_attribut)
