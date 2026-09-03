"""
Fonctions de mapping "enregistrement brut -> ExemplePivot", une par
corpus source.

Ces fonctions vivent volontairement HORS du domaine et des ports :
elles portent la connaissance specifique de la structure de chaque
corpus (MediQAl, FrenchMedMCQA, MedQuAD, UltraMedical-Preference),
qui est un detail d'integration, pas une regle metier generale.
Elles sont injectees dans `ConstruireDatasetPivotUseCase.executer()`.

NOTE : le contenu exact des mappers ci-dessous sera affine une fois
le profilage (Etape 1, ydata-profiling) execute sur chaque corpus
reel -- les noms de colonnes ci-dessous sont ceux documentes par les
fiches Hugging Face des datasets et pourront necessiter un ajustement
mineur.
"""

from __future__ import annotations

from chsa_triage.domain.model import (
    ExemplePivot,
    Langue,
    Message,
    NiveauConfiance,
    TypeExemple,
)


def mapper_mediqal(enregistrement: dict) -> ExemplePivot | None:
    """Mapping MediQAl (FR) -> ExemplePivot de type SFT."""
    question = enregistrement.get("question") or enregistrement.get("query")
    reponse  = enregistrement.get("answer") or enregistrement.get("reponse")

    if not question or not reponse:
        return None

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("mediqal"),
        source="MediQAl",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        symptomes=str(question),
        prompt=(Message(role="user", contenu=str(question)),),
        completion=(Message(role="assistant", contenu=str(reponse)),),
        niveau_confiance=NiveauConfiance.MOYENNE,
    )


def mapper_frenchmedmcqa(enregistrement: dict) -> ExemplePivot | None:
    """Mapping FrenchMedMCQA (FR, QCM) -> ExemplePivot de type SFT."""
    question = enregistrement.get("question")
    options  = enregistrement.get("options") or {}
    reponse_correcte = enregistrement.get("correct_answers") or enregistrement.get("answer")

    if not question or not reponse_correcte:
        return None

    texte_options = "\n".join(f"{cle}: {valeur}" for cle, valeur in options.items())
    prompt_complet = f"{question}\n\nOptions :\n{texte_options}"

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("frenchmedmcqa"),
        source="FrenchMedMCQA",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        symptomes="",
        prompt=(Message(role="user", contenu=prompt_complet),),
        completion=(Message(role="assistant", contenu=str(reponse_correcte)),),
        niveau_confiance=NiveauConfiance.HAUTE,  # QCM valide, reponse certaine
    )


def mapper_medquad(enregistrement: dict) -> ExemplePivot | None:
    """Mapping MedQuAD (EN) -> ExemplePivot de type SFT."""
    question = enregistrement.get("Question") or enregistrement.get("question")
    reponse  = enregistrement.get("Answer") or enregistrement.get("answer")

    if not question or not reponse:
        return None

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("medquad"),
        source="MedQuAD",
        type_exemple=TypeExemple.SFT,
        langue=Langue.ANGLAIS,
        symptomes=str(question),
        prompt=(Message(role="user", contenu=str(question)),),
        completion=(Message(role="assistant", contenu=str(reponse)),),
        niveau_confiance=NiveauConfiance.MOYENNE,
    )


def mapper_ultramedical_preference(enregistrement: dict) -> ExemplePivot | None:
    """Mapping UltraMedical-Preference (EN) -> ExemplePivot de type DPO."""
    prompt   = enregistrement.get("prompt") or enregistrement.get("instruction")
    chosen   = enregistrement.get("chosen")
    rejected = enregistrement.get("rejected")

    if not prompt or not chosen or not rejected:
        return None

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("ultramedical"),
        source="UltraMedical-Preference",
        type_exemple=TypeExemple.DPO,
        langue=Langue.ANGLAIS,
        symptomes="",
        prompt=(Message(role="user", contenu=str(prompt)),),
        chosen=(Message(role="assistant", contenu=str(chosen)),),
        rejected=(Message(role="assistant", contenu=str(rejected)),),
        niveau_confiance=NiveauConfiance.HAUTE,
    )


MAPPERS_PAR_CORPUS = {
    "mediqal": mapper_mediqal,
    "frenchmedmcqa": mapper_frenchmedmcqa,
    "medquad": mapper_medquad,
    "ultramedical_preference": mapper_ultramedical_preference,
}
