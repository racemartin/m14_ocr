"""
Adaptateur secondaire : anonymisation via Microsoft Presidio
(AnalyzerEngine + AnonymizerEngine). Implemente le port `Anonymiseur`.

Bug reel decouvert au smoke test : `AnalyzerEngine()` sans config
explicite ne supporte que l'anglais et peut declencher le
telechargement d'un modele spaCy volumineux non desire. Ce projet
exige du bilingue FR/EN : `NlpEngineProvider` est donc configure
explicitement vers les modeles deja installes localement.

Deux risques RGPD reels traites ici (detail dans
docs/02_etape1_donnees/01_rapport_rgpd.md §7) :
1. Faux negatifs des recognizers par defaut pour des identifiants
   francais (cf. `RecognizeurNirFrance`, NIR reel investigue).
2. Sur-anonymisation de l'age clinique : DATE_TIME ne distingue pas
   une date de naissance (identifiante) d'un age/une duree relative
   (signal clinique, pas identifiant) — cf. `_normaliser_ages` et
   `_construire_operateur_date_time`.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from chsa_triage.domain.ports.anonymiseur import (
    EntiteDetectee,
    ResultatAnonymisation,
)

# Correspondance langue du domaine -> code langue attendu par Presidio.
_CODES_LANGUE_PRESIDIO = {"fr": "fr", "en": "en"}

# Configuration explicite du moteur NLP multi-langue : reutilise les
# modeles spaCy deja telecharges localement (voir guide d'installation
# §1.3), evite tout telechargement automatique surprise.
_CONFIGURATION_NLP_MULTILANGUE = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "fr", "model_name": "fr_core_news_md"},
        {"lang_code": "en", "model_name": "en_core_web_sm"},
    ],
}


# ----------------------------------------------------------------------
# Recognizer personnalise : NIR francais (numero de securite sociale)
# Absent des recognizers par defaut de Presidio (verifie par inspection
# du code source, pas suppose).
#
# Structure reelle du NIR (decret n°82-103) : 15 chiffres = sexe(1) +
# annee de naissance(2) + mois(2) + departement(2, ou 2A/2B Corse) +
# commune(3) + numero d'ordre(3) + cle de controle(2). La cle se
# calcule par modulo 97 sur les 13 premiers chiffres, avec pour la
# Corse la substitution standard A->0/-1 000 000, B->0/-2 000 000.
# ----------------------------------------------------------------------
_MOTIF_NIR = (
    r"\b([12])[ .-]?"  # sexe : 1 = homme, 2 = femme
    r"(\d{2})[ .-]?"  # 2 derniers chiffres de l'annee de naissance
    r"(0[1-9]|1[0-2])[ .-]?"  # mois de naissance (01-12)
    r"(\d{2}|2[ABab])[ .-]?"  # departement de naissance (2A/2B = Corse)
    r"(\d{3})[ .-]?"  # code commune de naissance
    r"(\d{3})[ .-]?"  # numero d'ordre dans le mois/lieu
    r"(\d{2})\b"  # cle de controle (modulo 97)
)


class RecognizeurNirFrance(PatternRecognizer):
    """Reconnaisseur du NIR francais (securite sociale/carte Vitale).
    Le regex seul sur "15 chiffres" produirait trop de faux positifs ;
    `validate_result` verifie la cle de controle reelle (modulo 97) et
    rejette tout match dont la cle ne correspond pas.

    Limite connue : ne couvre pas les codes mois speciaux (naissance a
    l'etranger) ni les departements d'outre-mer (3 chiffres) ; non
    rencontres dans les donnees reelles du projet, ajoute par prudence.
    """

    PATTERNS = [
        Pattern("NIR francais (15 chiffres + cle modulo 97)", _MOTIF_NIR, 0.4)
    ]
    CONTEXT = [
        "sécurité sociale",
        "numéro de sécurité sociale",
        "nir",
        "carte vitale",
        "assuré social",
        "sécu",
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="FR_NIR",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr",
        )

    def validate_result(self, pattern_text: str) -> bool:
        compact = re.sub(r"[ .-]", "", pattern_text).upper()
        if len(compact) != 15:
            return False
        corps, cle = compact[:13], compact[13:]
        if not cle.isdigit():
            return False

        ajustement = 0
        if "A" in corps:
            corps = corps.replace("A", "0")
            ajustement = 1_000_000
        elif "B" in corps:
            corps = corps.replace("B", "0")
            ajustement = 2_000_000
        if not corps.isdigit():
            return False

        valeur = int(corps) - ajustement
        cle_calculee = 97 - (valeur % 97)
        return cle_calculee == int(cle)


# ----------------------------------------------------------------------
# Normalisation de l'age AVANT l'analyse Presidio : minimisation
# proportionnee (generaliser ce qui est cliniquement necessaire), pas
# une suppression indiscriminee. Bug reel confirme cote anglais :
# `en_core_web_sm` etiquette "7-year-old"/"aged 3 years" comme
# DATE_TIME ; `fr_core_news_md` n'a declenche aucune detection sur les
# equivalents francais dans les cas testes, mais la normalisation est
# appliquee aussi cote francais par prudence (le modele pourrait
# changer de comportement plus tard).
# ----------------------------------------------------------------------

_TRANCHE_PEDIATRIQUE = "pediatrique"
_TRANCHE_ADOLESCENT = "adolescent"
_TRANCHE_ADULTE = "adulte"
_TRANCHE_PERSONNE_AGEE = "personne_agee"

# Tranches retenues (0-12 / 13-17 / 18-64 / 65+) ;
# aucune autre coupure d'age n'est definie ailleurs dans le projet
# (cahier des charges, ESI) qui primerait sur celle-ci.
_JETONS_TRANCHE_AGE = {
    "fr": {
        _TRANCHE_PEDIATRIQUE: "<AGE_PEDIATRIQUE>",
        _TRANCHE_ADOLESCENT: "<AGE_ADOLESCENT>",
        _TRANCHE_ADULTE: "<AGE_ADULTE>",
        _TRANCHE_PERSONNE_AGEE: "<AGE_PERSONNE_AGEE>",
    },
    "en": {
        _TRANCHE_PEDIATRIQUE: "<AGE_PEDIATRIC>",
        _TRANCHE_ADOLESCENT: "<AGE_ADOLESCENT>",
        _TRANCHE_ADULTE: "<AGE_ADULT>",
        _TRANCHE_PERSONNE_AGEE: "<AGE_ELDERLY>",
    },
}

_MOTIFS_AGE_FR = [
    # "âgé de 60 ans", "âgée de 70 ans", "âgé (60 ans)"
    re.compile(r"âgée?\s*\(?\s*(?:de\s+)?(\d{1,3})\s*ans\)?", re.IGNORECASE),
    # "un enfant de 2 ans", "patiente de 70 ans", "homme de 20 ans"...
    re.compile(
        r"\b(?:enfant|nourrisson|adolescente?|nouveau-n[ée]|homme|femme|patiente?|garçon|fille)"
        r"\s+de\s+(\d{1,3})\s*ans\b",
        re.IGNORECASE,
    ),
]
_MOTIFS_AGE_EN = [
    # "7-year-old", "70 year old", "45 years old"
    re.compile(r"\b(\d{1,3})[- ]?years?[- ]?old\b", re.IGNORECASE),
    # "aged 20", "aged 3 years"
    re.compile(r"\baged\s+(\d{1,3})\b", re.IGNORECASE),
]
_MOTIFS_AGE_PAR_LANGUE = {"fr": _MOTIFS_AGE_FR, "en": _MOTIFS_AGE_EN}


def _tranche_pour_age(age: int) -> str:
    if age <= 12:
        return _TRANCHE_PEDIATRIQUE
    if age <= 17:
        return _TRANCHE_ADOLESCENT
    if age <= 64:
        return _TRANCHE_ADULTE
    return _TRANCHE_PERSONNE_AGEE


def _normaliser_ages(texte: str, code_langue: str) -> str:
    """Remplace toute mention explicite d'age par un jeton de tranche
    clinique AVANT l'analyse Presidio, pour que DATE_TIME ne voie
    jamais le nombre exact et ne l'elimine pas : l'age est un signal
    clinique reel, pas une PII a supprimer sans nuance."""
    motifs = _MOTIFS_AGE_PAR_LANGUE.get(code_langue, _MOTIFS_AGE_EN)
    jetons = _JETONS_TRANCHE_AGE.get(code_langue, _JETONS_TRANCHE_AGE["en"])

    def _remplacer(correspondance: re.Match[str]) -> str:
        age = int(correspondance.group(1))
        return jetons[_tranche_pour_age(age)]

    for motif in motifs:
        texte = motif.sub(_remplacer, texte)
    return texte


# ----------------------------------------------------------------------
# Operateur DATE_TIME personnalise : distingue une date calendaire
# absolue (identifiante -> masquee) d'une duree relative ("il y a 3
# semaines", clinique, pas identifiante -> laissee intacte). Complete
# `_normaliser_ages` : celui-ci traite l'age, celui-la les durees.
# ----------------------------------------------------------------------

_MOTIF_DATE_ABSOLUE = re.compile(
    r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"
    r"|\d{4}[/.-]\d{1,2}[/.-]\d{1,2}"
    r"|\b\d{1,2}\s+(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|"
    r"septembre|octobre|novembre|d[ée]cembre|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+\d{4}\b",
    re.IGNORECASE,
)

_UNITE_DUREE = (
    r"jours?|semaines?|mois|ans?|ann[ée]es?|days?|weeks?|months?|years?"
)
_MOTIF_DUREE_RELATIVE = re.compile(
    rf"\b(?:il y a|depuis|pendant|dans|since|for)\b[^.;\n]{{0,20}}?\d{{1,3}}\s*(?:{_UNITE_DUREE})\b"
    rf"|\b\d{{1,3}}\s*(?:{_UNITE_DUREE})\s+ago\b",
    re.IGNORECASE,
)


def _est_date_absolue(texte_capture: str) -> bool:
    return bool(_MOTIF_DATE_ABSOLUE.search(texte_capture))


def _est_duree_relative(texte_capture: str) -> bool:
    return bool(_MOTIF_DUREE_RELATIVE.search(texte_capture))


def _construire_operateur_date_time(
    jeton_pour_date: Callable[[str], str],
) -> OperatorConfig:
    """
    `jeton_pour_date` applique le MEME comportement de masquage que la
    strategie courante (replace/mask/redact) pour rester coherent avec
    l'operateur DEFAULT ; seule la decision "masquer ou conserver"
    change pour DATE_TIME.
    """

    def _operateur(texte_capture: str) -> str:
        if _est_date_absolue(texte_capture):
            return jeton_pour_date(texte_capture)
        if _est_duree_relative(texte_capture):
            return texte_capture
        return jeton_pour_date(texte_capture)

    return OperatorConfig("custom", {"lambda": _operateur})


def _construire_analyzer_multilangue() -> AnalyzerEngine:
    """Construit un AnalyzerEngine supportant explicitement FR et EN."""
    fournisseur = NlpEngineProvider(
        nlp_configuration=_CONFIGURATION_NLP_MULTILANGUE
    )
    moteur_nlp = fournisseur.create_engine()
    moteur = AnalyzerEngine(
        nlp_engine=moteur_nlp,
        supported_languages=["fr", "en"],
    )
    moteur.registry.add_recognizer(RecognizeurNirFrance())
    return moteur


class PresidioAnonymiseur:
    """Adaptateur Presidio implementant le port Anonymiseur."""

    def __init__(self, strategie: str = "replace") -> None:
        """
        strategie : "replace" | "mask" | "redact" ; cf. recommandation
        de la mission de tester plusieurs strategies de masquage.
        """
        self._analyzer = _construire_analyzer_multilangue()
        self._anonymizer = AnonymizerEngine()
        self._strategie = strategie

    def anonymiser(self, texte: str, langue: str) -> ResultatAnonymisation:
        if not texte:
            return ResultatAnonymisation(
                texte_original=texte,
                texte_anonymise=texte,
                entites_detectees=(),
            )

        code_langue = _CODES_LANGUE_PRESIDIO.get(langue, "en")
        texte_normalise = _normaliser_ages(texte, code_langue)
        resultats_analyse = self._analyzer.analyze(
            text=texte_normalise, language=code_langue
        )

        operateurs = self._construire_operateurs()
        resultat = self._anonymizer.anonymize(
            text=texte_normalise,
            analyzer_results=resultats_analyse,
            operators=operateurs,
        )

        entites = tuple(
            EntiteDetectee(
                type_entite=r.entity_type,
                debut=r.start,
                fin=r.end,
                score=r.score,
            )
            for r in resultats_analyse
        )

        return ResultatAnonymisation(
            texte_original=texte,
            texte_anonymise=resultat.text,
            entites_detectees=entites,
        )

    def _construire_operateurs(self) -> dict[str, OperatorConfig]:
        if self._strategie == "mask":
            defaut = OperatorConfig(
                "mask",
                {"masking_char": "*", "chars_to_mask": 100, "from_end": False},
            )
            jeton_pour_date: Callable[[str], str] = lambda t: "*" * len(t)
        elif self._strategie == "redact":
            defaut = OperatorConfig("redact", {})
            jeton_pour_date = lambda t: ""
        else:
            # "replace" par defaut
            defaut = OperatorConfig("replace", {"new_value": "<INFO_MASQUEE>"})
            jeton_pour_date = lambda t: "<INFO_MASQUEE>"

        return {
            "DEFAULT": defaut,
            "DATE_TIME": _construire_operateur_date_time(jeton_pour_date),
        }
