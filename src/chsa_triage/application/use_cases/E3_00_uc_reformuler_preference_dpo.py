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

**Prompt en UN SEUL tour `user`, jamais de tour `system` (corrige
21/09/2026, cf. AGENTS.md)** : le premier job DPO reel (100 exemples)
a echoue a 0/90 avec un texte genere VIDE (pas malformatte) pour les 5
echecs captures dans `echantillon_echecs_reformulation`. Verifie
directement sur `data/processed/dataset_pivot_anonymise.jsonl`
(134883 exemples reels, tous types confondus) : aucun `ExemplePivot`
n'a jamais de role `system` dans `prompt`/`chosen`/`rejected`, ce
checkpoint SFT-LoRA (`mombasstic/chsa-triage-sft-lora`) n'a donc
jamais vu de tour `system` pendant son propre entrainement. `PROMPT_REFORMULATION_CHOSEN`
et le texte a reformuler sont donc concatenes dans un unique
message `role: "user"`.

**Few-shot + parametres de generation explicites (hypothese, corrige
21/09/2026, NON CONFIRMEE par un run GPU reel reussi au moment de ce
commit)** : le deuxieme job DPO reel (meme correctif ci-dessus deja
applique) est reste a 0/90, mais avec un signal DIFFERENT dans
l'echantillon de diagnostic : 4/5 textes toujours vides, mais le 5e a
produit du texte reel (illisible, sans rapport avec le contenu
medical) plutot qu'une chaine vide. Ceci confirme que retirer le tour
`system` a bien eu un effet (ce n'est plus un EOS immediat
systematique), mais qu'un modele 1.7B base+LoRA, entraine
UNIQUEMENT sur des paires question medicale directe -> reponse
directe, jamais sur une meta-instruction de reformatage, ne suit pas
le format cible de facon fiable en pur zero-shot (description en
prose du format seule). `PROMPT_REFORMULATION_CHOSEN` inclut donc
desormais un exemple deja resolu (few-shot, contenu clinique invente
pour l'exemple, jamais presente comme un cas reel du dataset),
technique standard pour faire suivre un nouveau format a un petit
modele sans instruction-tuning general. `self.moteur.generer()` recoit
egalement des `parametres` explicites (`n_predict`/`temperature`,
vocabulaire llama.cpp deja utilise par
`_parametres_generation_transformers`, cf.
`infrastructure/adapters/transformers_inference_adapter.py`) au lieu
des defauts implicites de l'adaptateur : une temperature moderee
(`TEMPERATURE_REFORMULATION = 0.3`) plutot que purement greedy, la
degenerescence (vide ou basura) observee sur les deux premiers jobs
etant une signature typique d'un petit modele coince en decodage
glouton face a une tache nouvelle. Cette combinaison (few-shot +
sampling modere) est une HYPOTHESE bien fondee, PAS une certitude :
seul un prochain run GPU reel (job de 100) confirmera ou infirmera
qu'elle suffit a faire passer `parser_reformulation_stricte()`.

`echantillon_echecs_reformulation` capture desormais entree ET sortie
(`EchecReformulation`, une paire par echec, jamais deux listes
paralleles qui pourraient se desaligner) : les deux premiers jobs
reels n'avaient capture que la sortie brute, rendant impossible de
relier un echec (vide, illisible, etc.) au texte source qui l'a
produit sans rejouer manuellement le job.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from chsa_triage.application.validation_reformulation_dpo import parser_reformulation_stricte
from chsa_triage.domain.model.enums import TypeExemple
from chsa_triage.domain.model.exemple_pivot import ExemplePivot
from chsa_triage.domain.model.preference_reformulee import ChosenReformule
from chsa_triage.domain.ports.dataset_repository import RepositoryLectureEcriture
from chsa_triage.domain.ports.moteur_inference import MoteurInference


@dataclass(frozen=True, slots=True)
class EchecReformulation:
    """
    Un element de `ReformulerPreferenceDpoUseCase.echantillon_echecs_reformulation` :
    entree ET sortie emparieees (jamais deux listes paralleles qui
    pourraient se desaligner). `entree` est `texte_chosen_original` (le
    `chosen` source, PAS le message complet envoye au modele : celui-ci
    repete `PROMPT_REFORMULATION_CHOSEN`, constant et deja connu, a
    chaque appel, l'y rejouer par echec serait pure redondance dans le
    diagnostic). `sortie_brute` est `reponse.texte` tel quel, jamais
    retouche.
    """

    entree      : str
    sortie_brute : str

# Exemple few-shot : contenu clinique ENTIEREMENT INVENTE pour illustrer le
# format, jamais presente comme un cas reel du dataset (cf. docstring du
# module). `json.dumps` garantit que cet exemple respecte lui-meme, au mot
# pres, le format exige par `parser_reformulation_stricte()` (exactement les
# cles `niveau`/`categorie`/`ressources_estimees`).
_EXEMPLE_FEW_SHOT_ENTREE = (
    "Prenez du paracetamol toutes les 6 heures et reposez-vous. Si la fievre "
    "depasse 39 degres ou persiste plus de 3 jours, consultez un medecin."
)
_EXEMPLE_FEW_SHOT_SORTIE = (
    "<think>Fievre isolee geree en ambulatoire par antipyretique et repos ; "
    "aucun signe de gravite mentionne dans le texte, donc pas d'urgence "
    "immediate mais une consultation est deja conseillee en cas de "
    "persistance.</think>"
    + json.dumps(
        {
            "niveau": 4,
            "categorie": "fievre_ambulatoire",
            "ressources_estimees": "antipyretique en vente libre, pas d'examen complementaire immediat",
        }
    )
)

PROMPT_REFORMULATION_CHOSEN = (
    "Reformule la reponse medicale ci-dessous en un format structure de "
    "triage. C'est une reponse jugee meilleure (chosen) dans un corpus de "
    "preference medicale generale. Reformule-la, SANS changer son sens "
    "clinique ni inventer d'information absente du texte source, sous "
    "exactement cette forme :\n"
    "<think>quelques phrases de raisonnement clinique s'appuyant sur le "
    "texte source</think>"
    '{"niveau": <un entier d\'echelle ESI 1-5, deduit honnetement du texte, '
    'jamais invente sans lien avec lui>, "categorie": "<categorie clinique '
    'courte>", "ressources_estimees": "<ressources/examens vraisemblables>"}\n'
    "Ne produis rien d'autre que ce bloc <think> suivi du JSON, strictement "
    "ces trois cles, aucune autre.\n\n"
    "Voici un exemple deja resolu, a titre d'illustration UNIQUEMENT "
    "(contenu invente pour cet exemple, ne provient pas du dataset reel) :\n\n"
    "Reponse a reformuler :\n"
    f"{_EXEMPLE_FEW_SHOT_ENTREE}\n\n"
    "Reponse attendue :\n"
    f"{_EXEMPLE_FEW_SHOT_SORTIE}\n\n"
    "Reformule maintenant, EXACTEMENT selon ce meme format (uniquement le "
    "bloc <think> suivi du JSON, rien d'autre, aucun texte avant ou apres), "
    "la reponse suivante :"
)

# Vocabulaire llama.cpp (`n_predict`/`temperature`), deja celui traduit par
# `_parametres_generation_transformers` (cf.
# infrastructure/adapters/transformers_inference_adapter.py) vers les kwargs
# reels de `GenerationMixin.generate`. Passes explicitement plutot que de
# laisser l'adaptateur sur ses defauts implicites (cf. docstring du module) :
# - `TEMPERATURE_REFORMULATION = 0.3` : sampling modere plutot que purement
#   glouton. Les deux premiers jobs DPO reels ont produit du texte vide puis
#   du texte illisible, une signature typique d'un petit modele coince en
#   decodage glouton face a une tache nouvelle ; 0.3 vise a laisser une marge
#   d'echappement sans derailler trop loin du texte source clinique (une
#   temperature plus haute, ex. 0.7-1.0, risquerait d'halluciner un contenu
#   clinique absent du texte a reformuler).
# - `NOMBRE_TOKENS_GENERES_REFORMULATION = 256` : le few-shot ajoute au
#   PROMPT n'allonge pas la sortie attendue (elle reste bornee par
#   l'exemple, un court <think> + un petit JSON) ; 256 tokens restait deja
#   large pour ce format avant le few-shot et le reste apres.
TEMPERATURE_REFORMULATION = 0.3
NOMBRE_TOKENS_GENERES_REFORMULATION = 256


def _horodatage_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Taille max de `echantillon_echecs_reformulation` : diagnostic, jamais une
# croissance sans limite en memoire (un lot peut compter des milliers
# d'echecs, cf. le premier job DPO reel, 90/90 echecs de format). Portee de
# 5 a 20 (21/09/2026) : une fenetre de 5 sur un job de 100 exemples donnait
# une vue trop etroite de la distribution reelle des echecs (vide vs. basura
# vs. quasi-correct) pour diagnostiquer en une seule execution, alors que
# chaque job reel a un cout (GPU paye).
TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION = 20


@dataclass(slots=True)
class ReformulerPreferenceDpoUseCase:
    """Orchestre la reformulation incrementale/resumable du `chosen` d'un sous-ensemble DPO."""

    moteur                       : MoteurInference
    repository_reformule          : RepositoryLectureEcriture
    taille_cible                    : int = 5000
    horloge                          : Callable[[], str] = _horodatage_utc_iso

    nombre_echecs_reformulation : int = field(default=0, init=False)
    echantillon_echecs_reformulation: list[EchecReformulation] = field(default_factory=list, init=False)

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
        a montrer) garde une `EchecReformulation` (entree ET sortie brute
        emparieees, jamais deux listes paralleles) dans
        `self.echantillon_echecs_reformulation`, jusqu'a
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
            contenu_utilisateur = (
                f"{PROMPT_REFORMULATION_CHOSEN}\n\nReponse a reformuler :\n{texte_chosen_original}"
            )
            messages = [{"role": "user", "content": contenu_utilisateur}]
            parametres_generation = {
                "n_predict": NOMBRE_TOKENS_GENERES_REFORMULATION,
                "temperature": TEMPERATURE_REFORMULATION,
            }
            try:
                reponse = self.moteur.generer(messages, parametres_generation)
            except Exception:  # noqa: BLE001 - erreur adaptateur concrete, le domaine ne la type pas
                self.nombre_echecs_reformulation += 1
                continue

            chosen_reformule = parser_reformulation_stricte(reponse.texte)
            if chosen_reformule is None:
                self.nombre_echecs_reformulation += 1
                if len(self.echantillon_echecs_reformulation) < TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION:
                    self.echantillon_echecs_reformulation.append(
                        EchecReformulation(entree=texte_chosen_original, sortie_brute=reponse.texte)
                    )
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
