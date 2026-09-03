"""
Adaptateur secondaire : anonymisation via Microsoft Presidio
(AnalyzerEngine + AnonymizerEngine), conformement a la recommandation
de la mission.

Implemente le port `Anonymiseur`.

NOTE IMPORTANTE (bug reel decouvert lors du smoke test d'integration) :
`AnalyzerEngine()` construit SANS configuration explicite ne supporte
que l'anglais par defaut et peut declencher le telechargement
automatique d'un modele spaCy volumineux (`en_core_web_lg`, ~400 Mo)
non desire. Ce projet exige un dataset BILINGUE (FR/EN) -- il faut
donc configurer explicitement un `NlpEngineProvider` multi-langue,
pointant vers les modeles deja installes localement
(`fr_core_news_md`, `en_core_web_sm`), cf.
docs/01_environnement/00_guide_installation_environnement.md.
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
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


def _construire_analyzer_multilangue() -> AnalyzerEngine:
    """Construit un AnalyzerEngine supportant explicitement FR et EN."""
    fournisseur = NlpEngineProvider(nlp_configuration=_CONFIGURATION_NLP_MULTILANGUE)
    moteur_nlp = fournisseur.create_engine()
    return AnalyzerEngine(
        nlp_engine=moteur_nlp,
        supported_languages=["fr", "en"],
    )


class PresidioAnonymiseur:
    """Adaptateur Presidio implementant le port Anonymiseur."""

    def __init__(self, strategie: str = "replace") -> None:
        """
        strategie : "replace" | "mask" | "redact" -- cf. recommandation
        de la mission de tester plusieurs strategies de masquage.
        """
        self._analyzer = _construire_analyzer_multilangue()
        self._anonymizer = AnonymizerEngine()
        self._strategie = strategie

    def anonymiser(self, texte: str, langue: str) -> ResultatAnonymisation:
        if not texte:
            return ResultatAnonymisation(texte_original=texte, texte_anonymise=texte, entites_detectees=())

        code_langue = _CODES_LANGUE_PRESIDIO.get(langue, "en")
        resultats_analyse = self._analyzer.analyze(text=texte, language=code_langue)

        operateurs = self._construire_operateurs()
        resultat = self._anonymizer.anonymize(
            text=texte,
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
            return {"DEFAULT": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 100, "from_end": False})}
        if self._strategie == "redact":
            return {"DEFAULT": OperatorConfig("redact", {})}
        # "replace" par defaut
        return {"DEFAULT": OperatorConfig("replace", {"new_value": "<INFO_MASQUEE>"})}
