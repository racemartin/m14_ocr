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

from collections import OrderedDict

import spacy

from chsa_triage.domain.ports.verificateur_entites import VerdictEntiteNommee

# Memes modeles que PresidioAnonymiseur (cf. presidio_anonymiseur.py) --
# deja installes localement, aucun telechargement supplementaire.
_MODELES_SPACY = {"fr": "fr_core_news_md", "en": "en_core_web_sm"}

# Nombre d'entrees (texte, langue) -> Doc conservees dans le cache. Le
# controle qualite appelle verifier() plusieurs fois de suite avec le
# meme texte (un par candidat detecte dans ce champ) avant de passer
# au champ/exemple suivant : une petite fenetre LRU suffit a eliminer
# la redondance reelle sans faire grossir la memoire sur toute la
# duree d'une execution complete.
_TAILLE_CACHE_DOC = 8

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
        self._cache_docs: OrderedDict[tuple[str, str], spacy.tokens.Doc] = OrderedDict()

    def verifier(self, texte: str, langue: str, debut: int, fin: int) -> VerdictEntiteNommee:
        code_langue = langue if langue in _MODELES_SPACY else "en"
        doc = self._doc_analyse(texte, code_langue)
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

    def _doc_analyse(self, texte: str, code_langue: str) -> spacy.tokens.Doc:
        cle = (texte, code_langue)
        if cle in self._cache_docs:
            self._cache_docs.move_to_end(cle)
            return self._cache_docs[cle]
        doc = self._modele(code_langue)(texte)
        self._cache_docs[cle] = doc
        if len(self._cache_docs) > _TAILLE_CACHE_DOC:
            self._cache_docs.popitem(last=False)
        return doc
