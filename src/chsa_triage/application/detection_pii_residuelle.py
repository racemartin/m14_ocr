"""
Detection heuristique par expressions regulieres de candidats de PII
residuelle sur un texte DEJA anonymise -- meme methode que celle
utilisee pour la relecture manuelle assistee documentee dans
`docs/02_etape1_donnees/01_rapport_rgpd.md` §4 (emails, telephones,
URLs, dates completes, motifs "deux mots capitalises consecutifs").

Aucun modele n'est necessaire ici : les regex suffisent pour reperer
les candidats. Le tri entre "vraie entite nommee probable" et "faux
positif du regex" (termes medicaux capitalises) est fait ailleurs,
par une seconde opinion spaCy (cf. `verificateur_entites.py`) --
volontairement PAS dans ce module, pour que la detection regex reste
pure et testable sans dependance NLP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Categories DETERMINISTES : un match suffit, pas d'ambiguite
# "terme medical vs PII" a lever (contrairement au motif bigramme
# capitalise ci-dessous).
CATEGORIES_DETERMINISTES = frozenset({"email", "telephone", "url", "date"})

# Nombre de caracteres de contexte conserves de part et d'autre du
# passage detecte, pour que le passage reste inspectable dans le
# rapport sans reproduire le texte entier.
CONTEXTE_CARACTERES = 40

_MOTIFS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}"),
    "url": re.compile(r"https?://\S+|www\.\S+"),
    "telephone": re.compile(r"(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]){2,5}\d{2,4}"),
    "date": re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
        r"|\b\d{1,2}\s+(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|"
        r"septembre|octobre|novembre|d[ée]cembre|"
        r"January|February|March|April|May|June|July|August|September|October|"
        r"November|December)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    # Deux mots (ou plus) commencant par une majuscule, consecutifs --
    # motif le plus bruite (confond frequemment vocabulaire medical
    # capitalise et noms propres), d'ou la seconde opinion spaCy en aval.
    "bigramme_capitalise": re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+\b"),
}


@dataclass(frozen=True, slots=True)
class CandidatRegex:
    """Un passage signale par une regex de detection de PII residuelle."""

    type_motif : str
    debut       : int
    fin          : int
    passage      : str  # texte trouve + contexte autour, pour inspection


def detecter_candidats(texte: str) -> list[CandidatRegex]:
    """Applique tous les motifs de detection sur `texte`, retourne les candidats trouves."""
    if not texte:
        return []

    candidats: list[CandidatRegex] = []
    for type_motif, motif in _MOTIFS.items():
        for correspondance in motif.finditer(texte):
            debut, fin = correspondance.start(), correspondance.end()
            debut_contexte = max(0, debut - CONTEXTE_CARACTERES)
            fin_contexte = min(len(texte), fin + CONTEXTE_CARACTERES)
            candidats.append(
                CandidatRegex(
                    type_motif=type_motif,
                    debut=debut,
                    fin=fin,
                    passage=texte[debut_contexte:fin_contexte],
                )
            )
    return candidats
