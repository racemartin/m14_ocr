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
    from chsa_triage.infrastructure.adapters.jsonl_dataset_repository import (
        JsonlDatasetRepository,
    )
    from chsa_triage.infrastructure.adapters.jsonl_decisions_revision_humaine import (
        JsonlDecisionsRevisionHumaine,
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
    from chsa_triage.infrastructure.adapters.vllm_endpoint_inference_adapter import (
        VllmEndpointInferenceAdapter,
    )
    from chsa_triage.infrastructure.adapters.ydata_profileur import (
        YdataProfileur,
    )

__all__ = [
    "ChatMLFormateurAdapter",
    "JsonlDatasetRepository",
    "JsonlDecisionsRevisionHumaine",
    "JsonlRegistreEchantillonsControleQualite",
    "LecteurCorpusFichierLocal",
    "LecteurCorpusHuggingFace",
    "LlamaCppInferenceAdapter",
    "MlflowSuiviExperimentation",
    "PresidioAnonymiseur",
    "SpacyVerificateurEntitesNommees",
    "TensorboardSuiviExperimentation",
    "VllmEndpointInferenceAdapter",
    "YdataProfileur",
]

# Correspondance nom expose -> (module, attribut) pour le chargement paresseux.
_CARTE_IMPORTS_PARESSEUX = {
    "ChatMLFormateurAdapter": (
        "chsa_triage.infrastructure.adapters.chatml_formateur_adapter",
        "ChatMLFormateurAdapter",
    ),
    "JsonlDatasetRepository": (
        "chsa_triage.infrastructure.adapters.jsonl_dataset_repository",
        "JsonlDatasetRepository",
    ),
    "JsonlDecisionsRevisionHumaine": (
        "chsa_triage.infrastructure.adapters.jsonl_decisions_revision_humaine",
        "JsonlDecisionsRevisionHumaine",
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
