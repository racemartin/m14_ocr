"""
Test d'integration reel de TrlDpoEntraineurAdapter ; pas un mock.

Auto-ignore specifiquement si `torch.cuda.is_available()` est faux (cf.
`docs/04_etape3_dpo/03_guide_implementation_pas_a_pas.md` etape 12),
meme principe exact que `tests/infrastructure/test_trl_sft_entraineur.py` :
suppose un Environnement B deja provisionne (GPU + `peft`/`trl`/
`bitsandbytes` installes, PLUS un checkpoint SFT-LoRA reel accessible
via `chemin_checkpoint_politique_depart`), saute uniquement si aucun
GPU n'est detecte. Se saute donc TOUJOURS dans l'environnement
d'ecriture (Environnement A, sans GPU) : ce test n'a jamais tourne
pour de vrai, cf. l'avertissement complet dans
`infrastructure/adapters/trl_dpo_entraineur.py`.

Objectif du run reel (le jour ou ce test s'execute sur un vrai GPU) :
quelques dizaines de pas sur un petit sous-echantillon (50-100
exemples deja reformules, cf. guide etape 12), suffisant pour
confirmer (a) qu'aucune erreur CUDA ne survient et (b) que la perte
d'entrainement baisse reellement, sans consommer un run complet facture.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "test GPU reel : necessite torch.cuda.is_available() (Environnement B), "
        "jamais execute dans Environnement A",
        allow_module_level=True,
    )

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    ConfigurationQuantification,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
from chsa_triage.infrastructure.adapters.trl_dpo_entraineur import (
    TrlDpoEntraineurAdapter,
)

NOM_MODELE = "Qwen/Qwen3-1.7B-Base"
CHECKPOINT_SFT_LORA = "mombasstic/chsa-triage-sft-lora"


def _petit_dataset(nombre: int) -> list[ExempleFormatePreference]:
    """
    Genere `nombre` ExempleFormatePreference synthetiques au format
    ChatML brut (memes marqueurs que ce que produit reellement
    `ChatMLFormateurAdapter.formater_preference()`), varies pour ne pas
    etre tous identiques.
    """
    exemples = []
    for i in range(nombre):
        texte_prompt = (
            f"<|im_start|>user\nSymptome numero {i} : douleur thoracique legere.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        texte_chosen = (
            f"<|im_start|>assistant\n<think>raisonnement {i}</think>"
            f'{{"niveau": 3, "categorie": "cardio", "ressources_estimees": "ECG"}}<|im_end|>\n'
        )
        texte_rejected = f"<|im_start|>assistant\nCe n'est rien {i}.<|im_end|>\n"
        exemples.append(
            ExempleFormatePreference(
                identifiant=f"exemple-{i}",
                texte_prompt=texte_prompt,
                texte_chosen=texte_chosen,
                texte_rejected=texte_rejected,
            )
        )
    return exemples


def test_entrainement_reel_quelques_pas_perte_baisse(tmp_path):
    dataset_train = _petit_dataset(80)
    dataset_validation = _petit_dataset(20)

    config_lora = ConfigurationLora(rang=16, alpha=32, dropout=0.05, modules_cibles=("q_proj", "v_proj"))
    hyperparametres = HyperparametresEntrainementDpo(
        beta=0.1, taux_apprentissage=5e-6, nombre_epoques=1, taille_lot=2,
        type_perte="sigmoid", precompute_ref_log_probs=False,
    )
    configuration_quantification = ConfigurationQuantification(
        bits=4, type_quantification="nf4", double_quantification=True, dtype_calcul="bfloat16"
    )

    adaptateur = TrlDpoEntraineurAdapter(
        identifiant_modele_base=NOM_MODELE,
        configuration_quantification=configuration_quantification,
        chemin_checkpoint_politique_depart=CHECKPOINT_SFT_LORA,
        repertoire_sortie=str(tmp_path / "outputs"),
    )

    resultat = adaptateur.entrainer(
        dataset_train, dataset_validation, config_lora, hyperparametres, CHECKPOINT_SFT_LORA
    )

    assert resultat.chemin_checkpoint
    assert len(resultat.courbe_metriques) > 0
    for cle in ("rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins"):
        assert cle in resultat.metriques_recompense

    for point in resultat.courbe_metriques:
        assert not math.isnan(point.perte_train)
        assert not math.isinf(point.perte_train)

    premiere_perte = resultat.courbe_metriques[0].perte_train
    derniere_perte = resultat.courbe_metriques[-1].perte_train
    assert derniere_perte < premiere_perte, (
        f"perte attendue en baisse sur ce mini-run (premiere={premiere_perte}, "
        f"derniere={derniere_perte}) : si ce n'est pas le cas, verifier "
        f"hyperparametres/donnees avant de lancer un run complet facture"
    )


def test_entrainer_refuse_un_chemin_checkpoint_politique_depart_different(tmp_path):
    """
    `entrainer()` verifie la coherence avec le checkpoint deja charge
    dans le constructeur (cf. docstring de `TrlDpoEntraineurAdapter.entrainer`).
    Reste dans ce module GPU-only (pas un test de domaine/application) :
    `TrlDpoEntraineurAdapter.__post_init__` charge deja un modele reel
    a la construction (meme discipline que `TrlSftEntraineurAdapter`),
    donc meme cette verification de garde ne peut pas s'executer sans
    GPU/checkpoint reel accessible.
    """
    configuration_quantification = ConfigurationQuantification(
        bits=4, type_quantification="nf4", double_quantification=True, dtype_calcul="bfloat16"
    )
    adaptateur = TrlDpoEntraineurAdapter(
        identifiant_modele_base=NOM_MODELE,
        configuration_quantification=configuration_quantification,
        chemin_checkpoint_politique_depart=CHECKPOINT_SFT_LORA,
        repertoire_sortie=str(tmp_path / "outputs"),
    )

    with pytest.raises(ValueError, match="ne concorde pas"):
        adaptateur.entrainer(
            [], [],
            ConfigurationLora(rang=16, alpha=32, dropout=0.05, modules_cibles=("q_proj",)),
            HyperparametresEntrainementDpo(
                beta=0.1, taux_apprentissage=5e-6, nombre_epoques=1, taille_lot=2,
                type_perte="sigmoid", precompute_ref_log_probs=False,
            ),
            "un/depot-different",
        )
