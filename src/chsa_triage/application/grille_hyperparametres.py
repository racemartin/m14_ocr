"""
Selection sequentielle du prochain jeu d'hyperparametres a essayer,
pour `AjusterBoucleHyperparametresSftUseCase`. Fonction pure, aucun
port, meme principe que `application/verdict_convergence.py`.

Decision de conception (signature non fixee par
`docs/03_etape2_sft/02_etapes_cas_usage.md` §5, seuls le nom et le
role de `candidat_suivant(grille, historique)` y sont proposes) :
`grille` est ici une sequence ORDONNEE et deja entierement etalee de
`HyperparametresEntrainement` candidats (le produit cartesien des axes
de `recipes/sft_qwen3_lora.yaml::grille_hyperparametres` est calcule
en amont, par l'appelant qui assemble aussi `ConfigurationLora`, cf.
`training/sft_train.py`, non encore ecrit). Cette fonction se contente
de retourner le premier candidat de `grille` absent de `historique`
(l'ensemble des jeux deja essayes), ou `None` si tous l'ont ete
(grille epuisee).

Cette forme est deliberement generique : elle ne suppose rien sur les
champs qui varient dans la grille. Point ouvert documente : l'axe
`rang` de `grille_hyperparametres` (YAML) pilote `ConfigurationLora`,
pas `HyperparametresEntrainement` ; faire varier `rang` en boucle
suppose donc de faire varier aussi `config_lora` d'un essai a l'autre,
ce qu'`AjusterBoucleHyperparametresSftUseCase` ne fait pas dans cette
phase (elle recoit un `config_lora` fixe, cf. le cas d'usage). Cette
fonction reste utilisable telle quelle le jour ou ce point sera
tranche : il suffira que l'appelant etale aussi les combinaisons de
`config_lora` en parallele de `grille`.
"""

from __future__ import annotations

from collections.abc import Sequence

from chsa_triage.domain.model.configuration_entrainement import HyperparametresEntrainement


def candidat_suivant(
    grille: Sequence[HyperparametresEntrainement],
    historique: Sequence[HyperparametresEntrainement],
) -> HyperparametresEntrainement | None:
    """
    Retourne le premier element de `grille` absent de `historique`
    (comparaison par egalite de valeur, `HyperparametresEntrainement`
    etant `frozen=True`), ou `None` si `historique` couvre deja tous
    les candidats de `grille` (grille epuisee).
    """
    deja_essayes = set(historique)
    for candidat in grille:
        if candidat not in deja_essayes:
            return candidat
    return None
