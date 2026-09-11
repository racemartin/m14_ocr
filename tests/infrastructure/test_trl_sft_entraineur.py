"""
Test d'integration reel de TrlSftEntraineurAdapter ; pas un mock.

Auto-ignore specifiquement si `torch.cuda.is_available()` est faux
(cf. `docs/03_etape2_sft/03_guide_implementation_pas_a_pas.md` etape 9),
PAS pour absence de reseau/installation : contrairement a
`tests/infrastructure/test_chatml_formateur_adapter.py` (saute si le
tokenizer n'est pas telechargeable), ce test suppose un Environnement
B deja provisionne (GPU + `peft`/`trl`/`bitsandbytes` installes) et
saute uniquement si aucun GPU n'est detecte. Se saute donc TOUJOURS
dans l'environnement d'ecriture (Environnement A, sans GPU) : ce test
n'a jamais tourne pour de vrai, cf. l'avertissement complet dans
`infrastructure/adapters/trl_sft_entraineur.py`.

Objectif du run reel (le jour ou ce test s'execute sur un vrai GPU) :
quelques dizaines de pas sur un petit sous-echantillon (50-100
exemples), suffisant pour confirmer (a) qu'aucune erreur CUDA ne
survient et (b) que la perte d'entrainement baisse reellement, sans
consommer un run complet facture.

`assistant_only_loss=False` est utilise explicitement ici (pas le
defaut de la recette, qui vaut `true`) : la forme actuelle
d'`ExempleFormate` (texte ChatML deja rendu) est verifiee INCOMPATIBLE
avec `assistant_only_loss=True` (cf. le meme avertissement), ce test
verifie donc le chemin qui peut reellement fonctionner aujourd'hui,
pas celui qui echoue par construction.
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
    HyperparametresEntrainement,
)
from chsa_triage.domain.model.exemple_formate import ExempleFormate
from chsa_triage.infrastructure.adapters.trl_sft_entraineur import (
    TrlSftEntraineurAdapter,
)

NOM_MODELE = "Qwen/Qwen3-1.7B-Base"


def _petit_dataset(nombre: int) -> list[ExempleFormate]:
    """
    Genere `nombre` ExempleFormate synthetiques au format ChatML brut
    (memes marqueurs que ce que produit reellement ChatMLFormateurAdapter),
    varies pour ne pas etre tous identiques.
    """
    exemples = []
    for i in range(nombre):
        texte = (
            f"<|im_start|>system\nTu es un assistant medical.<|im_end|>\n"
            f"<|im_start|>user\nSymptome numero {i} : douleur thoracique legere.<|im_end|>\n"
            f"<|im_start|>assistant\nRecommandation {i} : consultation sous 48h.<|im_end|>\n"
        )
        exemples.append(ExempleFormate(identifiant=f"exemple-{i}", texte=texte))
    return exemples


def test_entrainement_reel_quelques_pas_perte_baisse(tmp_path):
    dataset_train = _petit_dataset(80)
    dataset_validation = _petit_dataset(20)

    config_lora = ConfigurationLora(rang=8, alpha=16, dropout=0.05, modules_cibles=("q_proj", "v_proj"))
    hyperparametres = HyperparametresEntrainement(
        taux_apprentissage=2e-4, nombre_epoques=1, taille_lot=2, packing=False, type_perte="chunked_nll"
    )
    configuration_quantification = ConfigurationQuantification(
        bits=4, type_quantification="nf4", double_quantification=True, dtype_calcul="bfloat16"
    )

    adaptateur = TrlSftEntraineurAdapter(
        identifiant_modele_base=NOM_MODELE,
        configuration_quantification=configuration_quantification,
        repertoire_sortie=str(tmp_path / "outputs"),
        assistant_only_loss=False,
    )

    resultat = adaptateur.entrainer(dataset_train, dataset_validation, config_lora, hyperparametres)

    assert resultat.chemin_checkpoint
    assert len(resultat.courbe_metriques) > 0

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
