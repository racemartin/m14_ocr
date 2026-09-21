"""
Cas d'usage : reformuler le `chosen` d'un sous-ensemble d'`ExemplePivot`
DPO vers le format cible `<think>...</think>` + JSON strict
(`niveau`/`categorie`/`ressources_estimees`, cahier des charges F3-F4),
decision tranchee en docs/04_etape3_dpo/00_introduction_concepts.md §3.2
et docs/04_etape3_dpo/02_etapes_cas_usage.md §1.

Reutilise `MoteurInference` (aucun nouveau port, cf. §1.2 du document
cas d'usage cite ci-dessus) : la reformulation est, au sens le plus
strict, "envoyer des messages, recevoir une reponse", exactement le
contrat deja expose par ce port (quatrieme reutilisation dans le
projet, apres les deux baselines zero-shot et l'evaluation post-SFT).
L'adaptateur concret attendu est `TransformersLoraInferenceAdapter`
(deja ecrit, Etape 1bis), pointe vers le checkpoint SFT-LoRA
(`mombasstic/chsa-triage-sft-lora`).

`taille_cible` est une cible CUMULATIVE (comme `E1_05_00_decouper_splits.py
--n`), pas une limite par execution (contrairement a
`AnonymiserDatasetUseCase --limite`) : cf. §1.3 du document cas d'usage,
"ce sous-ensemble reformule est PARTIEL... un futur passage pourrait
elargir la reformulation... exactement comme E1_05_00_decouper_splits.py
--n". Un exemple deja present dans le fichier de sortie
(`identifiants_existants()`) n'est jamais retraite.

`rejected` n'est ni lu ni ecrit ici (decision deliberee, §3.2 du
document d'introduction : seul `chosen` est reformule).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from chsa_triage.application.validation_reformulation_dpo import parser_reformulation_stricte
from chsa_triage.domain.model.enums import TypeExemple
from chsa_triage.domain.model.exemple_pivot import ExemplePivot
from chsa_triage.domain.model.preference_reformulee import ChosenReformule
from chsa_triage.domain.ports.dataset_repository import RepositoryLectureEcriture
from chsa_triage.domain.ports.moteur_inference import MoteurInference

PROMPT_REFORMULATION_CHOSEN = (
    "Tu es un assistant clinique charge de reformuler une reponse medicale "
    "en un format structure de triage. On te donne une reponse jugee "
    "meilleure (chosen) dans un corpus de preference medicale generale. "
    "Reformule-la, SANS changer son sens clinique ni inventer d'information "
    "absente du texte source, sous exactement cette forme :\n"
    "<think>quelques phrases de raisonnement clinique s'appuyant sur le "
    "texte source</think>"
    '{"niveau": <un entier d\'echelle ESI 1-5, deduit honnetement du texte, '
    'jamais invente sans lien avec lui>, "categorie": "<categorie clinique '
    'courte>", "ressources_estimees": "<ressources/examens vraisemblables>"}\n'
    "Ne produis rien d'autre que ce bloc <think> suivi du JSON, strictement "
    "ces trois cles, aucune autre."
)


def _horodatage_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Taille max de `echantillon_echecs_reformulation` : diagnostic, jamais une
# croissance sans limite en memoire (un lot peut compter des milliers
# d'echecs, cf. le premier job DPO reel, 90/90 echecs de format).
TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION = 5


@dataclass(slots=True)
class ReformulerPreferenceDpoUseCase:
    """Orchestre la reformulation incrementale/resumable du `chosen` d'un sous-ensemble DPO."""

    moteur                       : MoteurInference
    repository_reformule          : RepositoryLectureEcriture
    taille_cible                    : int = 5000
    horloge                          : Callable[[], str] = _horodatage_utc_iso

    nombre_echecs_reformulation : int = field(default=0, init=False)
    echantillon_echecs_reformulation: list[str] = field(default_factory=list, init=False)

    def executer(self, source_dpo: Iterable[ExemplePivot]) -> int:
        """
        Filtre `source_dpo` sur `type_exemple == TypeExemple.DPO`, exclut
        les identifiants deja presents dans `repository_reformule`
        (`identifiants_existants()`, patron incremental/resumable deja
        utilise par `AnonymiserDatasetUseCase --limite`, cf. AGENTS.md),
        et reformule au plus `taille_cible - (deja reformules)` nouveaux
        candidats via `self.moteur.generer()` +
        `parser_reformulation_stricte()`. Un echec (inference ou format,
        meme patron de resilience que
        `EvaluerBaselineZeroShotUseCase.executer()`) compte dans
        `self.nombre_echecs_reformulation` et n'ecrit jamais d'exemple
        partiel. Un echec de FORMAT (pas d'inference, qui n'a pas de texte
        a montrer) garde le texte brut renvoye par `self.moteur.generer()`
        dans `self.echantillon_echecs_reformulation`, jusqu'a
        `TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION` elements : pur
        diagnostic (voir AGENTS.md, premier job DPO reel a 0/90 succes),
        aucun impact sur le parsing/la validation elle-meme, qui reste
        inchangee. Persiste les `ChosenReformule` valides en un seul
        `sauvegarder_plusieurs()` (jamais un `sauvegarder()` par item,
        cf. AGENTS.md, cout O(n^2)). Retourne le nombre d'exemples
        effectivement reformules lors de CETTE execution.
        """
        deja_reformules = self.repository_reformule.identifiants_existants()
        candidats = [
            exemple
            for exemple in source_dpo
            if exemple.type_exemple == TypeExemple.DPO and exemple.identifiant not in deja_reformules
        ]

        nombre_restant = max(0, self.taille_cible - len(deja_reformules))
        candidats = candidats[:nombre_restant]

        self.nombre_echecs_reformulation = 0
        self.echantillon_echecs_reformulation = []
        reformules: list[ChosenReformule] = []
        for exemple in candidats:
            texte_chosen_original = "\n".join(message.contenu for message in exemple.chosen)
            messages = [
                {"role": "system", "content": PROMPT_REFORMULATION_CHOSEN},
                {"role": "user", "content": texte_chosen_original},
            ]
            try:
                reponse = self.moteur.generer(messages)
            except Exception:  # noqa: BLE001 - erreur adaptateur concrete, le domaine ne la type pas
                self.nombre_echecs_reformulation += 1
                continue

            chosen_reformule = parser_reformulation_stricte(reponse.texte)
            if chosen_reformule is None:
                self.nombre_echecs_reformulation += 1
                if len(self.echantillon_echecs_reformulation) < TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION:
                    self.echantillon_echecs_reformulation.append(reponse.texte)
                continue

            reformules.append(
                ChosenReformule(
                    identifiant=exemple.identifiant,
                    chosen_reformule=chosen_reformule,
                    horodatage=self.horloge(),
                )
            )

        self.repository_reformule.sauvegarder_plusieurs(reformules)
        return len(reformules)
