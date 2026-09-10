"""
Adaptateur secondaire : verification d'entite nommee via spaCy pur
(pas Presidio), utilise comme "seconde opinion" par le controle
qualite d'anonymisation.

Reutilise EXACTEMENT les memes modeles deja installes localement que
`PresidioAnonymiseur` (`fr_core_news_md`, `en_core_web_sm`) ; ce
n'est pas un nouveau modele externe, seulement un second usage du
meme moteur NLP deja present dans le projet, applique directement au
texte (sans passer par Presidio) pour trancher les candidats de PII
residuelle trouves par regex sur le texte deja anonymise.

Implemente le port `VerificateurEntitesNommees`.
"""

from __future__ import annotations

import spacy

from chsa_triage.domain.ports.verificateur_entites import VerdictEntiteNommee

# Memes modeles que PresidioAnonymiseur (cf. presidio_anonymiseur.py) --
# deja installes localement, aucun telechargement supplementaire.
_MODELES_SPACY = {"fr": "fr_core_news_md", "en": "en_core_web_sm"}

# Labels spaCy consideres "pertinents PII" par langue ; les schemas
# d'etiquettes different entre le modele francais (PER/ORG/LOC/MISC)
# et le modele anglais (PERSON/ORG/GPE/LOC/NORP).
_LABELS_PERTINENTS = {
    "fr": {"PER", "ORG", "LOC"},
    "en": {"PERSON", "ORG", "GPE", "LOC", "NORP"},
}


class SpacyVerificateurEntitesNommees:
    """Adaptateur spaCy implementant le port VerificateurEntitesNommees."""

    def __init__(self) -> None:
        self._modeles: dict[str, spacy.language.Language] = {}

    def verifier(self, texte: str, langue: str, debut: int, fin: int) -> VerdictEntiteNommee:
        code_langue = langue if langue in _MODELES_SPACY else "en"
        doc = self._modele(code_langue)(texte)
        labels_pertinents = _LABELS_PERTINENTS[code_langue]

        chevauchements = [ent for ent in doc.ents if ent.start_char < fin and ent.end_char > debut]
        if not chevauchements:
            return VerdictEntiteNommee.AUCUNE_ENTITE
        if any(ent.label_ in labels_pertinents for ent in chevauchements):
            return VerdictEntiteNommee.ENTITE_PERTINENTE
        return VerdictEntiteNommee.ENTITE_NON_PERTINENTE

    def _modele(self, code_langue: str) -> spacy.language.Language:
        if code_langue not in self._modeles:
            self._modeles[code_langue] = spacy.load(_MODELES_SPACY[code_langue])
        return self._modeles[code_langue]
