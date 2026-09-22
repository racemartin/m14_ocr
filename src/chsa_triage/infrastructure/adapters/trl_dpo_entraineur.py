"""
Adaptateur secondaire (Environnement B, GPU requis) : entrainement DPO
reel via `trl.DPOTrainer`, continuant le checkpoint SFT-LoRA deja
entraine, conformement a
`docs/04_etape3_dpo/03_guide_implementation_pas_a_pas.md` etape 12.

Implemente le port `EntraineurPreference`. Seul module du projet qui
importe `trl.DPOTrainer`/`trl.DPOConfig`, meme regle "seule
infrastructure/adapters importe des bibliotheques externes" deja
respectee par `TrlSftEntraineurAdapter`. Imports fonction-scoped
(`__post_init__`/`entrainer`), meme discipline : ce fichier reste
importable en Environnement A (`peft`/`trl` absents) tant qu'on
n'instancie pas `TrlDpoEntraineurAdapter` pour de vrai.

AVERTISSEMENT CRITIQUE (ecrit le 19/09/2026, jamais execute reellement) :
ce module n'a JAMAIS tourne sur un GPU, faute d'Environnement B au
moment de l'ecriture (meme statut que `TrlSftEntraineurAdapter` avant
son premier run reel). Tout ce qui suit distingue explicitement ce qui
a ete VERIFIE sans GPU (signatures reelles, lecture directe du code
source de `trl`/`peft`, appels de fonctions pures) de ce qui reste NON
VERIFIE et attend une vraie session GPU reelle.

MISE A JOUR REELLE (22/09/2026), premier entrainement DPO ayant
reellement boucle jusqu'au bout (job GPU L4, 100 exemples,
`--skip-reformulation`, 82/82 pas) : `trainer.train()` a reussi, mais
l'evaluation post-epoque a plante avec `torch.OutOfMemoryError` (traceback
exact : `DPOTrainer.evaluation_loop -> prediction_step ->
get_batch_loss_metrics -> concatenated_forward`, "Tried to allocate
9.17 GiB"). Cause racine CONFIRMEE (pas une hypothese, contrairement
aux corrections precedentes de ce module sur le comportement du
modele) : `DPOConfig(...)` fixait `per_device_train_batch_size` mais
jamais `per_device_eval_batch_size`, dont le defaut REEL verifie
(inspection directe de la signature `transformers.TrainingArguments.__init__`,
sans GPU) est **8**, le double du batch d'entrainement (4, valeur
reelle de `recipes/dpo_qwen3_lora.yaml`) deja demontre viable par ce
meme run. `concatenated_forward` traite `chosen`+`rejected`
concatenes dans la meme passe (visible dans le traceback), doublant
deja la charge memoire par element par rapport a un forward simple :
un lot d'evaluation deux fois plus grand que le lot d'entrainement,
sur une passe qui double deja la charge par element, suffit a
expliquer l'OOM observe sans autre cause necessaire. Fixe en fixant
`per_device_eval_batch_size=hyperparametres.taille_lot` dans
`entrainer()` ci-dessous (raisonnement complet en commentaire a cet
endroit). Reste NON CONFIRME jusqu'a un prochain run GPU reel :
`per_device_train_batch_size` n'a pas ete touche (il a deja fonctionne
82/82 pas).

VERIFIE SANS GPU (trl==1.13.0, peft==0.21.0, `trl` installe
temporairement dans le venv local via `uv pip install` pour cette
verification, PAS ajoute en dependance permanente d'Environnement A ;
desinstalle apres verification, meme methode deja utilisee pour
`peft.LoraConfig`/`trl.SFTConfig` en Etape 2) :

  - `trl.DPOConfig(...)` accepte bien `output_dir`, `beta`,
    `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`,
    `loss_type` (type reel `list[str]`, PAS `str` : `HyperparametresEntrainementDpo.type_perte`
    est enveloppe en liste a un element ci-dessous),
    `precompute_ref_log_probs`, `eval_strategy`, `report_to` : champs
    reels verifies (pas supposes) sur `trl.DPOConfig` (trl 1.13.0).
    Defauts reels confirmes : `beta=0.1`, `learning_rate=1e-06`,
    `precompute_ref_log_probs=False`, `loss_type=['sigmoid']` (cf.
    docs/04_etape3_dpo/00_introduction_concepts.md §5, memes valeurs).

  - **DECISION reelle sur `pi_ref`, tranchee par lecture directe du
    code source de `trl.DPOTrainer.__init__` (pas devinee)** : les deux
    options laissees ouvertes par
    docs/04_etape3_dpo/00_introduction_concepts.md §0/§2 et
    02_etapes_cas_usage.md §2 ("recharger une seconde copie gelee, ou
    deleguer a trl via ref_model=None") ne sont PAS symetriques une
    fois `peft_config` sorti de l'equation. `DPOTrainer.__init__`
    (trl==1.13.0) distingue deux chemins :
      (a) `peft_config is not None` -> `model = get_peft_model(model, peft_config)`,
          un NOUVEL adaptateur LoRA vierge, quels que soient les poids
          deja charges sur `model` : ce chemin EFFACERAIT le checkpoint
          SFT-LoRA deja entraine si on lui passait `config_lora` ici.
      (b) `elif is_peft_model(model) and ref_model is None:` -> si
          `model` est DEJA un `peft.PeftModel` (charge via
          `PeftModel.from_pretrained`, notre cas, cf. `__post_init__`
          ci-dessous) et qu'aucun `ref_model` explicite n'est fourni,
          `trl` AJOUTE LUI-MEME un second adaptateur fige ("ref"),
          copie exacte des poids de l'adaptateur "default" au moment
          du chargement (boucle `model.add_adapter("ref", default_config)`
          + copie parametre par parametre), et calcule `pi_ref` en
          desactivant temporairement l'adaptateur "default" au profit
          de "ref" pendant le passage avant de reference. AUCUN second
          modele complet n'est charge en memoire.
    Ce projet suit le chemin (b) : `TrlDpoEntraineurAdapter` charge
    `chemin_checkpoint_politique_depart` UNE SEULE FOIS (base quantifie
    + adaptateur LoRA-SFT, via `PeftModel.from_pretrained(...,
    is_trainable=True)`), ne passe JAMAIS `peft_config` a `DPOTrainer`,
    et laisse `ref_model=None` : c'est le chemin le MOINS couteux en
    memoire GPU (un seul modele de base charge, deux adaptateurs LoRA
    legers superposes), et c'est un choix imposé par le code lui-meme,
    pas une preference arbitraire (le chemin (a) DETRUIRAIT le point de
    depart post-SFT que toute la conception du DPO, §0/§2 du document
    d'introduction, exige explicitement).

  - **CONSEQUENCE, `config_lora` recu par `entrainer()` mais NON
    UTILISE pour construire un `peft.LoraConfig`** : le port
    `EntraineurPreference.entrainer()` (deja fige, etapes 1-11,
    `domain/ports/entraineur_preference.py`) impose ce parametre dans
    sa signature ; il reste accepte ici pour respecter le contrat, mais
    n'est jamais transmis a `DPOTrainer` (le passer creerait
    exactement le chemin (a) ci-dessus, un adaptateur vierge qui
    efface le SFT). L'architecture LoRA reelle (rang/alpha/modules
    cibles) est deja figee par le checkpoint SFT-LoRA charge en
    `__post_init__` (`adapter_config.json` du depot HF) : elle ne peut
    de toute facon pas etre changee en continuant un adaptateur deja
    entraine. `config_lora` reste utile COTE APPELANT (cf.
    `training/E3_03_dpo_train.py`) pour documenter, dans les metadonnees
    `CheckpointEntraine` persistees, l'architecture LoRA reellement en
    jeu (celle du checkpoint SFT de depart), pas pour cet adaptateur.

  - `peft.PeftModel.from_pretrained(model, model_id, is_trainable=...)`
    a bien un parametre `is_trainable`, DEFAUT REEL VERIFIE `False`
    (inspection de signature, peft 0.21.0) : sans le passer
    explicitement a `True`, l'adaptateur LoRA-SFT charge serait GELE
    (aucun gradient), et DPO n'entrainerait rien. `TrlDpoEntraineurAdapter`
    le passe donc explicitement.

  - `peft.prepare_model_for_kbit_training(model)` existe reellement
    (inspection de signature, peft 0.21.0) : applique AVANT
    `PeftModel.from_pretrained`, sur le modele de base quantifie brut,
    meme recette QLoRA standard que documentee par `peft` pour
    continuer l'entrainement d'un adaptateur deja sauvegarde sur un
    modele de base charge en 4-bit (cast des couches sensibles,
    `use_cache=False`, gradient checkpointing compatible). NON
    applique par `TrlSftEntraineurAdapter` (Etape 2) parce que ce
    dernier delegue `get_peft_model` a `SFTTrainer` lui-meme (qui gere
    cette preparation en interne quand `peft_config` est fourni) ;
    ici, le chemin (b) ci-dessus ne passe jamais par cette
    preparation interne de `trl`, d'ou l'appel explicite.

  - `trl.data_utils.is_conversational({"prompt": "texte", "chosen":
    "texte", "rejected": "texte"})` retourne bien `False` (verifie par
    appel reel) : confirme que le format "Explicit prompt" en
    CHAINES DE CARACTERES simples (pas des listes de messages), exactement
    la forme portee par `ExempleFormatePreference`
    (`texte_prompt`/`texte_chosen`/`texte_rejected`), est accepte par
    `DPOTrainer` sans conversion prealable : memes garanties deja
    verifiees pour `is_conversational({"text": "..."})` en Etape 2
    (SFT), point de depart de la demande de correction de ce module.

  - Cles reelles des 4 metriques de recompense propres a `trl.DPOTrainer`
    (`rewards/chosen`, `rewards/rejected`, `rewards/accuracies`,
    `rewards/margins`), verifiees par lecture directe de
    `trl/trainer/dpo_trainer.py` (grep reel du code source, pas
    suppose) : memes noms deja actes en
    docs/04_etape3_dpo/00_introduction_concepts.md §4.4 et deja cables
    cote `EntrainerDpoUseCase.CLES_METRIQUES_RECOMPENSE_DPO` (etapes
    1-11, deja fusionnees). Journalisees par `trl` dans
    `trainer.state.log_history` au meme rythme que `loss`/`grad_norm`
    (mecanisme `transformers.Trainer.log()` standard, pas surcharge) :
    `_metriques_recompense_depuis_log_history` ci-dessous retient la
    DERNIERE valeur observee de chaque cle (valeur agregee finale, pas
    une courbe, cf. docstring de `ResultatEntrainementDPO.metriques_recompense`).

NON VERIFIE, a confirmer sur une vraie session GPU avant de faire
confiance a ce module (memes categories de reserve que
`TrlSftEntraineurAdapter`) :
  - Que le modele charge en 4-bit (QLoRA NF4) + adaptateur LoRA-SFT
    continue de s'entrainer reellement sans erreur CUDA/memoire sur
    `Qwen3-1.7B-Base`, ni que la perte DPO baisse reellement.
  - Que `prepare_model_for_kbit_training` + `PeftModel.from_pretrained(...,
    is_trainable=True)` suffisent reellement (sans appel supplementaire
    a `model.enable_input_require_grads()`, que `DPOTrainer` semble
    gerer lui-meme pour un modele PEFT avec gradient checkpointing
    active, cf. lecture du code source, mais jamais confronte a un
    run reel).
  - Le mapping `Trainer.state.log_history` -> `MetriquesEntrainement`/
    `metriques_recompense` ci-dessous : reutilise
    `_courbe_depuis_log_history` de `trl_sft_entraineur.py` TEL QUEL
    (memes cles `loss`/`grad_norm`/`eval_loss`/`step`, DPOTrainer
    herite du meme mecanisme de `Trainer`, pas surcharge), jamais
    confronte a un vrai `log_history` DPO.
  - `precompute_ref_log_probs=True` (mitigation memoire documentee en
    00_introduction_concepts.md §5) : laisse au defaut `False` de la
    recette, jamais mesure.
  - Que `chemin_checkpoint` (repertoire ecrit par `trainer.save_model()`)
    est bien rechargeable ensuite (evaluation post-DPO, futur).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chsa_triage.domain.model.configuration_entrainement import (
    ConfigurationLora,
    ConfigurationQuantification,
    HyperparametresEntrainementDpo,
)
from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
from chsa_triage.domain.ports.entraineur_preference import ResultatEntrainementDPO
from chsa_triage.infrastructure.adapters.trl_sft_entraineur import (
    _courbe_depuis_log_history,
    _horodatage_nom_run,
)

CHECKPOINT_POLITIQUE_DEPART_DEFAUT = "mombasstic/chsa-triage-sft-lora"

# Cles reelles de trl.DPOTrainer (rewards/*), sans equivalent SFT, cf.
# AVERTISSEMENT en tete de module et docs/04_etape3_dpo/00_introduction_concepts.md §4.4.
CLES_METRIQUES_RECOMPENSE = (
    "rewards/chosen",
    "rewards/rejected",
    "rewards/accuracies",
    "rewards/margins",
)


def _construire_dpo_config(chemin_checkpoint: str, hyperparametres: HyperparametresEntrainementDpo) -> Any:
    """
    Construit le `trl.DPOConfig` reel a partir des hyperparametres du
    domaine. Extrait de `entrainer()` pour rester testable SANS GPU
    (construire un `DPOConfig`/`TrainingArguments` ne touche jamais le
    GPU ni ne charge de modele), cf.
    `tests/infrastructure/test_trl_dpo_entraineur_config.py`.

    `per_device_eval_batch_size` est fixe explicitement a
    `hyperparametres.taille_lot` (meme valeur que
    `per_device_train_batch_size`) : cf. l'AVERTISSEMENT "MISE A JOUR
    REELLE (22/09/2026)" en tete de module pour la cause racine
    confirmee (defaut reel de `transformers.TrainingArguments`, 8, le
    double du batch d'entrainement) de l'OOM CUDA observe pendant
    l'evaluation post-epoque sur le premier run DPO ayant reellement
    boucle son entrainement jusqu'au bout.
    """
    from trl import DPOConfig

    return DPOConfig(
        output_dir=chemin_checkpoint,
        beta=hyperparametres.beta,
        learning_rate=hyperparametres.taux_apprentissage,
        num_train_epochs=hyperparametres.nombre_epoques,
        per_device_train_batch_size=hyperparametres.taille_lot,
        per_device_eval_batch_size=hyperparametres.taille_lot,
        loss_type=[hyperparametres.type_perte],
        precompute_ref_log_probs=hyperparametres.precompute_ref_log_probs,
        eval_strategy="epoch",
        report_to="none",
    )


def _metriques_recompense_depuis_log_history(log_history: list[dict]) -> dict[str, float]:
    """
    Valeurs agregees FINALES (derniere occurrence dans `log_history`,
    PAS une courbe par etape) des 4 metriques de recompense propres a
    `trl.DPOTrainer` : c'est le mode d'agregation attendu par
    `ResultatEntrainementDPO.metriques_recompense`, cf.
    `domain/ports/entraineur_preference.py`.
    """
    valeurs: dict[str, float] = {}
    for entree in log_history:
        for cle in CLES_METRIQUES_RECOMPENSE:
            if cle in entree:
                valeurs[cle] = entree[cle]
    return valeurs


@dataclass(slots=True)
class TrlDpoEntraineurAdapter:
    """
    Adaptateur GPU implementant `EntraineurPreference` via
    `trl.DPOTrainer`, continuant l'entrainement d'un checkpoint
    SFT-LoRA deja produit. Charge le modele de base quantifie PLUS
    l'adaptateur LoRA-SFT UNE SEULE FOIS dans `__init__` (meme patron
    que `TrlSftEntraineurAdapter`/`TransformersLoraInferenceAdapter`) :
    jamais expose au-dela de la frontiere domain/application (le
    domaine ne voit que `ExempleFormatePreference`/`ConfigurationLora`/
    `HyperparametresEntrainementDpo`/`ResultatEntrainementDPO`).

    `chemin_checkpoint_politique_depart` existe a la fois ICI (constructeur,
    charge le modele une seule fois, cf. guide etape 12) et comme
    parametre de `entrainer()` (impose par le port `EntraineurPreference`
    deja fige, etapes 1-11) : `entrainer()` verifie que les deux
    concordent plutot que d'ignorer silencieusement l'un des deux (cf.
    docstring de la methode).
    """

    identifiant_modele_base            : str
    configuration_quantification         : ConfigurationQuantification
    chemin_checkpoint_politique_depart     : str = CHECKPOINT_POLITIQUE_DEPART_DEFAUT
    repertoire_sortie                        : str = "outputs/dpo-lora"
    attn_implementation                        : str = "sdpa"   # cf. AVERTISSEMENT TrlSftEntraineurAdapter, non re-mesure ici
    _modele                                      : Any = field(default=None, init=False, repr=False)
    _tokenizer                                    : Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        conf = self.configuration_quantification
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=conf.bits == 4,
            load_in_8bit=conf.bits == 8,
            bnb_4bit_quant_type=conf.type_quantification,
            bnb_4bit_use_double_quant=conf.double_quantification,
            bnb_4bit_compute_dtype=getattr(torch, conf.dtype_calcul),
        )

        modele_base = AutoModelForCausalLM.from_pretrained(
            self.identifiant_modele_base,
            quantization_config=bnb_config,
            attn_implementation=self.attn_implementation,
            trust_remote_code=True,
        )
        modele_base = prepare_model_for_kbit_training(modele_base)
        # is_trainable=True : sans ce flag (defaut reel False, cf. AVERTISSEMENT),
        # l'adaptateur LoRA-SFT charge resterait gele et DPO n'entrainerait rien.
        self._modele = PeftModel.from_pretrained(
            modele_base, self.chemin_checkpoint_politique_depart, is_trainable=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.identifiant_modele_base, trust_remote_code=True
        )

    def entrainer(
        self,
        dataset_train                        : Iterable[ExempleFormatePreference],
        dataset_validation                    : Iterable[ExempleFormatePreference],
        config_lora                             : ConfigurationLora,
        hyperparametres                           : HyperparametresEntrainementDpo,
        chemin_checkpoint_politique_depart          : str,
    ) -> ResultatEntrainementDPO:
        """
        Construit `trl.DPOConfig` + `trl.DPOTrainer` a partir des
        dataclasses pures du domaine, lance `trainer.train()`,
        sauvegarde le checkpoint LoRA, et reconstruit la courbe de
        metriques + les metriques de recompense depuis
        `trainer.state.log_history`.

        `config_lora` est accepte (contrat du port) mais N'EST JAMAIS
        transmis a `DPOTrainer` : cf. AVERTISSEMENT en tete de module,
        "CONSEQUENCE, config_lora recu mais non utilise". `ref_model`
        n'est jamais construit explicitement (`ref_model=None` transmis
        implicitement a `DPOTrainer` : `trl` derive `pi_ref` lui-meme,
        cf. AVERTISSEMENT, "DECISION reelle sur pi_ref").

        `chemin_checkpoint_politique_depart` DOIT concorder avec celui
        deja charge en `__init__` : cet adaptateur charge son modele
        UNE SEULE FOIS a la construction (meme discipline que
        `TrlSftEntraineurAdapter`), il ne peut pas recharger un autre
        checkpoint de depart a la volee ici. Leve `ValueError` sinon,
        plutot que d'ignorer silencieusement l'argument du port.
        """
        if chemin_checkpoint_politique_depart != self.chemin_checkpoint_politique_depart:
            raise ValueError(
                f"chemin_checkpoint_politique_depart={chemin_checkpoint_politique_depart!r} passe a entrainer() "
                f"ne concorde pas avec {self.chemin_checkpoint_politique_depart!r} deja charge dans le constructeur "
                "de TrlDpoEntraineurAdapter : cet adaptateur charge son modele de depart UNE SEULE FOIS a la "
                "construction, il ne peut pas en recharger un autre a la volee."
            )

        from datasets import Dataset
        from trl import DPOTrainer

        dataset_train_hf = Dataset.from_list(
            [
                {"prompt": e.texte_prompt, "chosen": e.texte_chosen, "rejected": e.texte_rejected}
                for e in dataset_train
            ]
        )
        dataset_validation_hf = Dataset.from_list(
            [
                {"prompt": e.texte_prompt, "chosen": e.texte_chosen, "rejected": e.texte_rejected}
                for e in dataset_validation
            ]
        )

        chemin_checkpoint = str(Path(self.repertoire_sortie) / f"run-{_horodatage_nom_run()}")

        dpo_config = _construire_dpo_config(chemin_checkpoint, hyperparametres)

        trainer = DPOTrainer(
            model=self._modele,
            ref_model=None,
            args=dpo_config,
            train_dataset=dataset_train_hf,
            eval_dataset=dataset_validation_hf,
            processing_class=self._tokenizer,
        )

        trainer.train()
        trainer.save_model(chemin_checkpoint)

        courbe = _courbe_depuis_log_history(trainer.state.log_history)
        metriques_recompense = _metriques_recompense_depuis_log_history(trainer.state.log_history)
        return ResultatEntrainementDPO(
            chemin_checkpoint=chemin_checkpoint,
            courbe_metriques=courbe,
            metriques_recompense=metriques_recompense,
        )
