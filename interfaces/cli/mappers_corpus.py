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

NOTE (07/09/2026, confirme sur les fichiers reels telecharges) :
MediQAl (ANR-MALADES/MediQAl) a 3 configurations sur le Hub, avec
DEUX schemas differents, desormais toutes deux couvertes :
  - "oeq" (question ouverte)  : champs `question` + `answer`.
    -> `mapper_mediqal` (cle `mediqal` / `mediqal_oeq`).
  - "mcqu" (QCM, 1 reponse) et "mcqm" (QCM, reponses multiples,
    ex. correct_answers="C,D") : champs `question` + `clinical_case`
    (peut etre `null`) + `answer_a` a `answer_e` + `correct_answers`
    + `task` ("QCU"/"QCM") -- PAS de champ `answer`.
    -> `mapper_mediqal_qcm` (cles `mediqal_mcqu` / `mediqal_mcqm`).
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
    """
    Mapping MediQAl (FR) -> ExemplePivot de type SFT.

    Ne couvre que la configuration "oeq" (question ouverte, champs
    `question`/`answer`). Les configurations "mcqu"/"mcqm" (QCM, champs
    `answer_a`..`answer_e` + `correct_answers`) n'ont PAS ce champ
    `answer` -- utiliser `mapper_mediqal_qcm` pour celles-ci. Voir la
    note de module ci-dessus.
    """
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


_LETTRES_OPTIONS = ("a", "b", "c", "d", "e")


def mapper_mediqal_qcm(enregistrement: dict) -> ExemplePivot | None:
    """
    Mapping MediQAl "mcqu"/"mcqm" (FR, QCM 1 ou plusieurs reponses)
    -> ExemplePivot de type SFT.

    Traitement simple decide par le capitaine (identique a
    `mapper_medquad`) : pas de liste d'options dans le prompt.
    - prompt = `clinical_case` (quand non vide/non nul) concatene avec
      `question` ; le cas clinique est necessaire car de nombreuses
      questions ("Au sujet des vaccinations :") n'ont pas de sens sans
      lui.
    - completion = texte(s) de la (des) reponse(s) correcte(s),
      resolu(s) en cherchant chaque lettre de `correct_answers`
      (ex. "C" ou "C,D") dans `answer_a`..`answer_e`, puis concatenes
      quand il y en a plusieurs (mcqm).
    """
    question = enregistrement.get("question")
    if not question:
        return None

    cas_clinique = enregistrement.get("clinical_case")
    if cas_clinique:
        prompt_texte = f"{cas_clinique}\n\n{question}"
    else:
        prompt_texte = str(question)

    correct_answers = enregistrement.get("correct_answers")
    if not correct_answers:
        return None

    lettres = [lettre.strip().lower() for lettre in str(correct_answers).split(",") if lettre.strip()]
    textes_reponses = [
        str(enregistrement[f"answer_{lettre}"])
        for lettre in lettres
        if enregistrement.get(f"answer_{lettre}")
    ]

    if not textes_reponses:
        return None

    completion_texte = " ".join(textes_reponses)

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("mediqal"),
        source="MediQAl",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        symptomes=prompt_texte,
        prompt=(Message(role="user", contenu=prompt_texte),),
        completion=(Message(role="assistant", contenu=completion_texte),),
        niveau_confiance=NiveauConfiance.HAUTE,  # QCM valide, reponse certaine
    )


def mapper_frenchmedmcqa(enregistrement: dict) -> ExemplePivot | None:
    """
    Mapping FrenchMedMCQA (FR, QCM) -> ExemplePivot de type SFT.

    Schema reel (confirme sur nthngdy/frenchmedmcqa, les 3 splits,
    1080 enregistrements, 07/09/2026) : champs plats `answer_a` a
    `answer_e` (PAS de champ `options`), `correct_answers` (entier,
    index 0-based dans a..e -- confirme a la fois par
    `datasets.load_dataset(...).features["correct_answers"]`, qui est
    un simple `Value("int64")`, et manuellement sur plusieurs
    enregistrements reels, ex. correct_answers=4 -> "e" pour une
    question sur les particules alpha, correct_answers=0 -> "a" pour
    une question sur la progesterone), et `number_correct_answers`
    (`ClassLabel(names=["1","2","3","4","5"])` -- index 0 signifie
    "1 reponse correcte"). Sur les 1080 enregistrements reels
    (train+validation+test), `number_correct_answers` vaut toujours 0
    (= 1 seule reponse) : le champ `correct_answers` est un entier
    unique, il ne peut de toute facon pas encoder plusieurs index a la
    fois, donc le cas multi-reponse n'est ni observe ni representable
    par ce schema sur ce miroir HF -- pas de gestion speciale requise.
    """
    question = enregistrement.get("question")
    index_reponse = enregistrement.get("correct_answers")

    if not question or index_reponse is None:
        return None

    try:
        index_reponse = int(index_reponse)
    except (TypeError, ValueError):
        return None

    if not 0 <= index_reponse < len(_LETTRES_OPTIONS):
        return None

    lettre = _LETTRES_OPTIONS[index_reponse]
    reponse_correcte = enregistrement.get(f"answer_{lettre}")

    if not reponse_correcte:
        return None

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("frenchmedmcqa"),
        source="FrenchMedMCQA",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        symptomes="",
        prompt=(Message(role="user", contenu=str(question)),),
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


def _extraire_reponse_assistant(messages) -> str | None:
    """
    UltraMedical-Preference stocke chosen/rejected au format chat :
    liste de messages [{"content": ..., "role": "user"|"assistant"},
    ...], PAS une chaine directe -- confirme sur les 109353
    enregistrements reels (03/09/2026) : toujours une liste de 2
    messages, le dernier toujours role="assistant". Ce mapper extrait
    ce dernier message ; sans cette extraction, `str(messages)`
    serialise toute la liste (prompt duplique + syntaxe de dict
    Python) au lieu du texte de la reponse.
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list) and messages:
        dernier = messages[-1]
        if isinstance(dernier, dict):
            return dernier.get("content")
    return None


def mapper_ultramedical_preference(enregistrement: dict) -> ExemplePivot | None:
    """Mapping UltraMedical-Preference (EN) -> ExemplePivot de type DPO."""
    prompt   = enregistrement.get("prompt") or enregistrement.get("instruction")
    chosen   = _extraire_reponse_assistant(enregistrement.get("chosen"))
    rejected = _extraire_reponse_assistant(enregistrement.get("rejected"))

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
    "mediqal_oeq": mapper_mediqal,
    "mediqal_mcqu": mapper_mediqal_qcm,
    "mediqal_mcqm": mapper_mediqal_qcm,
    "frenchmedmcqa": mapper_frenchmedmcqa,
    "medquad": mapper_medquad,
    "ultramedical_preference": mapper_ultramedical_preference,
}
