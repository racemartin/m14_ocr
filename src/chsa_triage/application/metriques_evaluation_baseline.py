"""
Metriques pures d'evaluation d'une reponse generee contre une reponse
de reference, sans port ni dependance externe : meme principe que
`application/verdict_convergence.py`/`application/echantillonnage.py`,
testeables sans reseau ni GPU avec des donnees synthetiques.

Utilisees par `E1_06_00_evaluer_baseline_zero_shot.py` (Etape 1bis,
baseline zero-shot de `Qwen/Qwen3-1.7B-Base`, sans entrainement) pour
comparer chaque generation du moteur d'inference (`MoteurInference`)
au `completion` reel d'un `ExemplePivot` de type SFT.

Point de vigilance honnete (voir
`docs/03_etape2_sft/00_introduction_concepts.md`, "le format de sortie
cible n'existe pas encore dans les donnees") : le cahier des charges
(F3, §9) attend une sortie JSON strict `{niveau, categorie,
ressources_estimees}` pour mesurer l'accuracy de classification ESI,
mais les `completion` REELS du pivot actuel (MediQAl, FrenchMedMCQA,
MedQuAD) sont des reponses en langage naturel, pas ce format JSON.
`extraire_niveau_triage`/`exactitude_classification_niveau` sont donc
ecrites et testees pour ce format cible (donnees synthetiques), mais
n'auront quasiment aucune paire comparable sur le dataset REEL
d'aujourd'hui : `nombre_comparables` le rend visible plutot que de
masquer un taux d'accuracy calcule sur une poignee de coincidences.
L'exact match / F1 textuel, eux, s'appliquent tels quels au dataset
reel (comparaison texte contre texte, pas de schema suppose).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

_MOTIF_BLOC_JSON = re.compile(r"\{.*\}", re.DOTALL)
_MOTIF_PONCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_MOTIF_ESPACES = re.compile(r"\s+")


def extraire_json(texte: str) -> dict | None:
    """
    Tente d'extraire un objet JSON de `texte`. Essaie d'abord
    `texte` tel quel (cas ou le modele ne produit QUE le JSON), puis le
    plus grand bloc `{...}` trouve (cas ou le JSON est precede d'un
    bloc `<think>...</think>` ou de texte libre, cf. F4 du cahier des
    charges). Retourne `None` si aucun objet JSON valide n'est
    trouvable, jamais une exception : le texte genere par un modele
    NON entraine (baseline zero-shot) peut ne contenir aucun JSON du
    tout.
    """
    for candidat in (texte, _premier_bloc_json(texte)):
        if candidat is None:
            continue
        try:
            objet = json.loads(candidat)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(objet, dict):
            return objet
    return None


def _premier_bloc_json(texte: str) -> str | None:
    trouve = _MOTIF_BLOC_JSON.search(texte)
    return trouve.group(0) if trouve else None


def extraire_niveau_triage(texte: str) -> str | None:
    """
    Extrait le champ `niveau` (echelle ESI, F2/F3 du cahier des
    charges) d'un texte cense contenir le JSON cible
    `{niveau, categorie, ressources_estimees}`. Retourne `None` si le
    texte ne contient pas de JSON exploitable ou pas de champ `niveau`.
    La valeur est retournee normalisee en texte (`str(...).strip()`)
    pour tolerer aussi bien `"niveau": 3` que `"niveau": "3"` cote
    modele.
    """
    objet = extraire_json(texte)
    if objet is None or "niveau" not in objet:
        return None
    return str(objet["niveau"]).strip()


@dataclass(frozen=True, slots=True)
class ResultatExactitudeNiveau:
    """Resultat de `exactitude_classification_niveau` : distingue explicitement la couverture du score."""

    nombre_paires        : int
    nombre_comparables   : int
    nombre_corrects       : int

    @property
    def exactitude(self) -> float | None:
        """`None` (pas `0.0`) quand aucune paire n'est comparable : distingue "aucune donnee" de "0% correct"."""
        if self.nombre_comparables == 0:
            return None
        return self.nombre_corrects / self.nombre_comparables


def exactitude_classification_niveau(paires: list[tuple[str, str]]) -> ResultatExactitudeNiveau:
    """
    Calcule l'exactitude de classification du niveau de triage ESI sur
    `paires` de `(texte_genere, texte_reference)`. Une paire n'est
    "comparable" que si un `niveau` est extractible des DEUX textes
    (cf. `extraire_niveau_triage`) ; les paires non comparables sont
    comptees (`nombre_paires - nombre_comparables`) mais n'entrent pas
    dans `exactitude`.
    """
    nombre_comparables = 0
    nombre_corrects = 0
    for genere, reference in paires:
        niveau_genere = extraire_niveau_triage(genere)
        niveau_reference = extraire_niveau_triage(reference)
        if niveau_genere is None or niveau_reference is None:
            continue
        nombre_comparables += 1
        if niveau_genere == niveau_reference:
            nombre_corrects += 1
    return ResultatExactitudeNiveau(
        nombre_paires=len(paires), nombre_comparables=nombre_comparables, nombre_corrects=nombre_corrects
    )


def normaliser_texte(texte: str) -> str:
    """Minuscules, ponctuation retiree, espaces multiples reduits : normalisation commune a EM et F1."""
    texte = texte.lower()
    texte = _MOTIF_PONCTUATION.sub(" ", texte)
    texte = _MOTIF_ESPACES.sub(" ", texte).strip()
    return texte


def correspondance_exacte(genere: str, reference: str) -> bool:
    """Exact match (EM) : egalite des deux textes apres `normaliser_texte`."""
    return normaliser_texte(genere) == normaliser_texte(reference)


def score_f1_tokens(genere: str, reference: str) -> float:
    """
    F1 au niveau token (meme principe que le F1 SQuAD) sur les textes
    normalises : harmonique de precision/rappel calcules sur le
    multi-ensemble de tokens communs. `1.0` si les deux textes sont
    vides apres normalisation (rien a comparer, correspondance
    triviale) ; `0.0` si un seul des deux est vide.
    """
    tokens_generes = normaliser_texte(genere).split()
    tokens_reference = normaliser_texte(reference).split()

    if not tokens_generes and not tokens_reference:
        return 1.0
    if not tokens_generes or not tokens_reference:
        return 0.0

    communs = Counter(tokens_generes) & Counter(tokens_reference)
    nombre_communs = sum(communs.values())
    if nombre_communs == 0:
        return 0.0

    precision = nombre_communs / len(tokens_generes)
    rappel = nombre_communs / len(tokens_reference)
    return 2 * precision * rappel / (precision + rappel)


def taux_exact_match(paires: list[tuple[str, str]]) -> float:
    """Proportion de paires `(genere, reference)` en exact match. `0.0` (pas d'exception) si `paires` est vide."""
    if not paires:
        return 0.0
    return sum(correspondance_exacte(genere, reference) for genere, reference in paires) / len(paires)


def f1_moyen(paires: list[tuple[str, str]]) -> float:
    """F1 token moyen sur `paires`. `0.0` (pas d'exception) si `paires` est vide."""
    if not paires:
        return 0.0
    return sum(score_f1_tokens(genere, reference) for genere, reference in paires) / len(paires)
