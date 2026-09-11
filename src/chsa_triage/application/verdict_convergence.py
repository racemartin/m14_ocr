"""
Diagnostic pur d'une courbe d'entrainement SFT : aucun port, aucune
dependance externe, meme principe que `application/echantillonnage.py`
et `application/detection_pii_residuelle.py`. Consomme directement la
`courbe_metriques` deja produite par `EntraineurSupervise.entrainer()`
(cf. `domain.ports.entraineur_supervise.ResultatEntrainementSFT`), sans
appel de port supplementaire.

Les seuils numeriques ci-dessous sont des PARAMETRES A CALIBRER, pas
des valeurs mesurees : aucune vraie courbe d'entrainement n'a encore
ete observee au moment de l'ecriture (l'adaptateur GPU,
`TrlSftEntraineurAdapter`, n'existe pas encore, cf.
`docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md` etape 9).
Ils devront etre ajustes une fois un premier run reel disponible.
"""

from __future__ import annotations

import math

from chsa_triage.domain.model.checkpoint_entraine import VerdictConvergence
from chsa_triage.domain.ports.entraineur_supervise import MetriquesEntrainement

# Baisse relative minimale de la perte d'entrainement entre le premier
# et le dernier pas de la courbe, en dessous de laquelle le modele est
# considere comme n'ayant pas appris (SOUS_APPRENTISSAGE). Provisoire.
SEUIL_BAISSE_TRAIN_RELATIVE_MINIMALE = 0.05

# Hausse absolue de la perte de validation, mesuree entre le debut et
# la fin de la fenetre `FENETRE_PAS_VALIDATION` la plus recente, au
# dela de laquelle le run est considere en surapprentissage. Provisoire.
SEUIL_HAUSSE_VALIDATION_SURAPPRENTISSAGE = 0.05

# Nombre de points de mesure de validation, en fin de courbe, examines
# pour detecter une remontee de la perte de validation.
FENETRE_PAS_VALIDATION = 3

# Facteur de croissance de la norme de gradient (dernier pas / premier
# pas) au-dela duquel, combine a une perte d'entrainement qui remonte,
# la courbe est consideree comme instable (divergence).
SEUIL_RATIO_DIVERGENCE_GRADIENT = 10.0


def evaluer_convergence(courbe: tuple[MetriquesEntrainement, ...]) -> VerdictConvergence:
    """
    Pose un diagnostic sur une courbe d'entrainement complete
    (`ResultatEntrainementSFT.courbe_metriques`), dans cet ordre de
    priorite :

    1. `INSTABLE` : une perte ou une norme de gradient NaN/infinie
       apparait n'importe ou dans la courbe, ou la norme de gradient
       diverge (ratio dernier/premier pas au-dela de
       `SEUIL_RATIO_DIVERGENCE_GRADIENT`) pendant que la perte
       d'entrainement remonte.
    2. `SOUS_APPRENTISSAGE` : la perte d'entrainement stagne (baisse
       relative entre premier et dernier pas sous
       `SEUIL_BAISSE_TRAIN_RELATIVE_MINIMALE`).
    3. `SURAPPRENTISSAGE` : la perte de validation remonte sur les
       `FENETRE_PAS_VALIDATION` derniers points mesures (hausse
       au-dela de `SEUIL_HAUSSE_VALIDATION_SURAPPRENTISSAGE`) alors que
       la perte d'entrainement continue de baisser.
    4. `SAINE` : dans tous les autres cas.

    Leve `ValueError` si `courbe` est vide (aucun diagnostic possible).
    """
    if not courbe:
        raise ValueError("evaluer_convergence necessite une courbe non vide")

    if _est_instable(courbe):
        return VerdictConvergence.INSTABLE

    pertes_train = [point.perte_train for point in courbe]
    baisse_relative = _baisse_relative(pertes_train[0], pertes_train[-1])
    if baisse_relative < SEUIL_BAISSE_TRAIN_RELATIVE_MINIMALE:
        return VerdictConvergence.SOUS_APPRENTISSAGE

    if _surapprentissage(courbe):
        return VerdictConvergence.SURAPPRENTISSAGE

    return VerdictConvergence.SAINE


def _est_instable(courbe: tuple[MetriquesEntrainement, ...]) -> bool:
    for point in courbe:
        valeurs = [point.perte_train, point.norme_gradient]
        if point.perte_validation is not None:
            valeurs.append(point.perte_validation)
        if any(math.isnan(v) or math.isinf(v) for v in valeurs):
            return True

    premier_gradient = courbe[0].norme_gradient
    dernier_gradient = courbe[-1].norme_gradient
    if premier_gradient > 0:
        ratio_gradient = dernier_gradient / premier_gradient
        perte_train_remonte = courbe[-1].perte_train > courbe[0].perte_train
        if ratio_gradient >= SEUIL_RATIO_DIVERGENCE_GRADIENT and perte_train_remonte:
            return True

    return False


def _baisse_relative(perte_initiale: float, perte_finale: float) -> float:
    if perte_initiale == 0:
        return 0.0
    return (perte_initiale - perte_finale) / perte_initiale


def _surapprentissage(courbe: tuple[MetriquesEntrainement, ...]) -> bool:
    pertes_validation = [point.perte_validation for point in courbe if point.perte_validation is not None]
    if len(pertes_validation) < 2:
        return False

    fenetre = pertes_validation[-FENETRE_PAS_VALIDATION:]
    hausse_validation = fenetre[-1] - fenetre[0]
    return hausse_validation > SEUIL_HAUSSE_VALIDATION_SURAPPRENTISSAGE
