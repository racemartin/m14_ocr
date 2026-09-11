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
reel ; les noms de colonnes ci-dessous sont ceux documentes par les
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
    + `task` ("QCU"/"QCM"), PAS de champ `answer`.
    -> `mapper_mediqal_qcm` (cles `mediqal_mcqu` / `mediqal_mcqm`).

NOTE (08/09/2026, identifiant deterministe, cf.
`ExemplePivot.nouvel_identifiant`) : chaque mapper derive desormais un
identifiant STABLE (pas aleatoire) a partir d'une cle naturelle propre
a chaque registre brut, verifiee sur les fichiers reels de
`data/raw/` (pas supposee depuis la fiche Hugging Face) :
  - mediqal_oeq/mcqu/mcqm, frenchmedmcqa : champ `id` du registre brut.
    ATTENTION, verifie sur les fichiers reels : `mediqal_oeq.jsonl`
    et `mediqal_mcqu.jsonl` partagent 1492 valeurs de `id` identiques
    bien qu'ils decrivent des registres differents (et 1280 avec
    mediqal_mcqm) ; l'espace de noms passe a `nouvel_identifiant` doit
    donc etre plus fin que le simple `source="MediQAl"` commun aux
    trois (`mediqal_oeq`/`mediqal_mcqu`/`mediqal_mcqm`, distingues via
    le champ `task` pour le mapper QCM partage par les deux derniers).
  - medquad : aucun champ `id` dans le registre brut ; cle naturelle
    = `Question` + `Answer` concatenes (verifie : 16 359 valeurs
    uniques sur 16 407 registres, 48 doublons EXACTS Question+Answer ;
    `Question` seule n'aurait donne que 14 979 valeurs uniques,
    beaucoup moins fiable).
  - ultramedical_preference : `prompt_id` seul n'est PAS unique
    (verifie : 77 046 valeurs uniques sur 109 353 registres) ;
    cle naturelle = `prompt_id` + `label_type` + reponse `chosen` +
    reponse `rejected` (verifie : 97 081 valeurs uniques, donc 12 272
    registres strictement identiques sur ces 4 champs, de vrais
    doublons, pas une collision de cle insuffisante).

Les registres dont la cle naturelle produit un identifiant deja vu
sont de VRAIS doublons (contenu strictement identique sur les champs
qui alimentent le pivot) ; `ConstruireDatasetPivotUseCase` les
detecte et les ecarte (dedoublonnage reel,
08/09/2026), voir `doublons_supprimes.jsonl` et
`docs/02_etape1_donnees/00_couverture_exigences_officielles.md` pour
le detail par source.
"""

from __future__ import annotations

import hashlib

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

    mediqal_oeq.jsonl
    • id
    • clinical_case
    • cc_question_number
    • question
    • answer
    • medical_subject
    • question_type

    Ne couvre que la configuration "oeq" (question ouverte, champs
    `question`/`answer`). Les configurations "mcqu"/"mcqm" (QCM, champs
    `answer_a`..`answer_e` + `correct_answers`) n'ont PAS ce champ
    `answer` ; utiliser `mapper_mediqal_qcm` pour celles-ci. Voir la
    note de module ci-dessus.
    """
    question = enregistrement.get("question") or enregistrement.get("query")
    reponse  = enregistrement.get("answer")   or enregistrement.get("reponse")

    if not question or not reponse:
        return None

    cle_naturelle = str(enregistrement.get("id"))

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("mediqal_oeq", cle_naturelle),
        identifiant_source_brute=cle_naturelle,
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

    mediqal_mcqu.jsonl
    • id
    • clinical_case
    • question
    • answer_a
    • answer_b
    • answer_c
    • answer_d
    • answer_e
    • correct_answers
    • task
    • medical_subject
    • question_type

    mediqal_mcqm.jsonl
    • id
    • clinical_case
    • question
    • answer_a
    • answer_b
    • answer_c
    • answer_d
    • answer_e
    • correct_answers
    • task
    • medical_subject
    • question_type

    Traitement simple, identique a
    `mapper_medquad` : pas de liste d'options dans le prompt.
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

    # Distingue mcqu ("QCU") de mcqm ("QCM") ; verifie sur les fichiers
    # reels : `task` vaut exclusivement "QCU" dans mediqal_mcqu.jsonl et
    # "QCM" dans mediqal_mcqm.jsonl, aucune valeur mixte. Necessaire car
    # les deux fichiers partagent des valeurs de `id` avec mediqal_oeq
    # (et donc, sans cette distinction, l'identifiant deterministe
    # collisionnerait entre registres differents).
    espace_noms = "mediqal_mcqu" if enregistrement.get("task") == "QCU" else "mediqal_mcqm"
    cle_naturelle = str(enregistrement.get("id"))

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant(espace_noms, cle_naturelle),
        identifiant_source_brute=cle_naturelle,
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

    frenchmedmcqa.jsonl
    • id
    • question
    • answer_a
    • answer_b
    • answer_c
    • answer_d
    • answer_e
    • correct_answers
    • number_correct_answers

    Schema reel (confirme sur nthngdy/frenchmedmcqa, les 3 splits,
    1080 enregistrements, 07/09/2026) : champs plats `answer_a` a
    `answer_e` (PAS de champ `options`), `correct_answers` (entier,
    index 0-based dans a..e, confirme a la fois par
    `datasets.load_dataset(...).features["correct_answers"]`, qui est
    un simple `Value("int64")`, et manuellement sur plusieurs
    enregistrements reels, ex. correct_answers=4 -> "e" pour une
    question sur les particules alpha, correct_answers=0 -> "a" pour
    une question sur la progesterone), et `number_correct_answers`
    (`ClassLabel(names=["1","2","3","4","5"])`, index 0 signifie
    "1 reponse correcte"). Sur les 1080 enregistrements reels
    (train+validation+test), `number_correct_answers` vaut toujours 0
    (= 1 seule reponse) : le champ `correct_answers` est un entier
    unique, il ne peut de toute facon pas encoder plusieurs index a la
    fois, donc le cas multi-reponse n'est ni observe ni representable
    par ce schema sur ce miroir HF ; pas de gestion speciale requise.
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

    cle_naturelle = str(enregistrement.get("id"))

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("frenchmedmcqa", cle_naturelle),
        identifiant_source_brute=cle_naturelle,
        source="FrenchMedMCQA",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        symptomes="",
        prompt=(Message(role="user", contenu=str(question)),),
        completion=(Message(role="assistant", contenu=str(reponse_correcte)),),
        niveau_confiance=NiveauConfiance.HAUTE,  # QCM valide, reponse certaine
    )


def mapper_medquad(enregistrement: dict) -> ExemplePivot | None:
    """Mapping MedQuAD (EN) -> ExemplePivot de type SFT.
    * medquad.jsonl
    • focus
    • question
    • answer
    """
    question = enregistrement.get("Question") or enregistrement.get("question")
    reponse  = enregistrement.get("Answer") or enregistrement.get("answer")

    if not question or not reponse:
        return None

    # Aucun champ `id` dans le registre brut MedQuAD (verifie sur le
    # fichier reel) ; cle naturelle = hash de Question+Answer, verifie
    # unique a 16 359/16 407 (48 doublons EXACTS Question+Answer ;
    # Question seule n'aurait donne que 14 979 valeurs uniques).
    cle_naturelle = hashlib.sha256(f"{question}||{reponse}".encode("utf-8")).hexdigest()

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("medquad", cle_naturelle),
        identifiant_source_brute=cle_naturelle,
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
    ...], PAS une chaine directe ; confirme sur les 109353
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
    """Mapping UltraMedical-Preference (EN) -> ExemplePivot de type DPO.
    
    ultramedical_preference.jsonl
    • prompt_id
    • label_type
    • prompt
    • chosen
        • role
        • content
    • rejected
        • role
        • content
    • messages
        • role
        • content
    • metadata
        • golden_answer
        • chosen_model
        • score
        • evaluation
        • rejected_model
        • score
        • evaluation
        • feedback
    
    """
    prompt   = enregistrement.get("prompt") or enregistrement.get("instruction")
    chosen   = _extraire_reponse_assistant(enregistrement.get("chosen"))
    rejected = _extraire_reponse_assistant(enregistrement.get("rejected"))

    if not prompt or not chosen or not rejected:
        return None

    prompt_id = enregistrement.get("prompt_id", "")
    label_type = enregistrement.get("label_type", "")

    # `prompt_id` seul n'est PAS unique (verifie sur le fichier reel :
    # 77 046 valeurs uniques sur 109 353 registres ; le meme prompt
    # est annote plusieurs fois avec des criteres de preference
    # differents, ex. label_type "easy"/"hard"/"length"). Cle naturelle
    # = prompt_id + label_type + reponses chosen/rejected, verifiee
    # unique a 97 081/109 353 (12 272 registres strictement identiques
    # sur ces 4 champs, de vrais doublons, pas une collision de cle
    # insuffisante).
    cle_naturelle = f"{prompt_id}|{label_type}|{chosen}|{rejected}"

    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("ultramedical_preference", cle_naturelle),
        identifiant_source_brute=str(prompt_id),
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
