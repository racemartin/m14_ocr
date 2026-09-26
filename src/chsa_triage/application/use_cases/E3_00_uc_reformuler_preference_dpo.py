"""
Cas d'usage : reformuler le `chosen` d'un sous-ensemble d'`ExemplePivot`
DPO vers le format cible `<think>...</think>` + JSON strict
(`niveau`/`categorie`/`ressources_estimees`). Reutilise `MoteurInference`
(aucun nouveau port), vers `TransformersLoraInferenceAdapter` pointe sur
le checkpoint SFT-LoRA. `taille_cible` est CUMULATIVE (comme
`E1_05_00_decouper_splits.py --n`) : un exemple deja reformule
(`identifiants_existants()`) n'est jamais retraite. `rejected` n'est ni
lu ni ecrit ici (seul `chosen` est reformule).

Journal de debogage reel (5 jobs DPO successifs, tous a l'origine 0/90
succes de format) qui explique les constantes ci-dessous :
1. Prompt en un seul tour `user`, jamais `system` : le checkpoint
   n'avait jamais vu de role `system` a l'entrainement, d'ou un texte
   genere VIDE avec un tour `system`.
2. Few-shot + `temperature=0.3` : un modele 1.7B base+LoRA jamais
   entraine sur une meta-instruction de reformatage ne suit pas un
   format decrit en pur zero-shot ; le few-shot + sampling modere corrige
   le texte illisible observe au job 2.
3. Troncature de l'entree (hypothese testee au job 4, longueurs 300-800+
   mots) : ECARTEE ensuite comme cause (job 5, entrees deja courtes,
   toujours en echec).
4. Cause reelle trouvee au job 5 : `nombre_tokens_sortie == 1` sur tous
   les echecs, le modele emet un seul token (EOS) quel que soit
   l'entree. Correctif : `min_new_tokens` (supprime l'option EOS tant
   que ce minimum n'est pas atteint) — risque connu et accepte : peut
   produire du remplissage incoherent si le modele n'a rien a dire
   au-dela de son arret naturel.
Etat au moment de l'ecriture de ce commentaire : correctif applique,
resultat d'un run GPU reel pas encore confirme a ce stade (voir le
README pour l'issue reelle si depuis resolue).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from chsa_triage.application.validation_reformulation_dpo import (
    parser_reformulation_stricte,
)
from chsa_triage.domain.model.enums import TypeExemple
from chsa_triage.domain.model.exemple_pivot import ExemplePivot
from chsa_triage.domain.model.preference_reformulee import ChosenReformule
from chsa_triage.domain.ports.dataset_repository import (
    RepositoryLectureEcriture,
)
from chsa_triage.domain.ports.moteur_inference import MoteurInference


@dataclass(frozen=True, slots=True)
class EchecReformulation:
    """Un element de `echantillon_echecs_reformulation` : entree ET
    sortie appariees (jamais deux listes paralleles). `entree` est le
    texte source tel qu'envoye (apres troncature, sans repeter le prompt
    constant). `nombre_tokens_*` sont recopies depuis `ReponseModele`,
    diagnostic pour la degenerescence de generation (cf. docstring du
    module)."""

    entree: str
    sortie_brute: str
    nombre_tokens_entree: int = 0
    nombre_tokens_sortie: int = 0


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

# Vocabulaire llama.cpp (n_predict/temperature), traduit par
# _parametres_generation_transformers vers les kwargs de GenerationMixin.generate.
# Voir la docstring du module pour l'historique de debogage qui justifie
# ces valeurs (temperature moderee, plafond de sortie, minimum anti-EOS).
TEMPERATURE_REFORMULATION = 0.3
NOMBRE_TOKENS_GENERES_REFORMULATION = 256
MIN_TOKENS_GENERES_REFORMULATION = 24

# Longueur max (caracteres) de l'entree avant insertion dans le message :
# suffisant pour capturer le contenu clinique principal (typiquement dans
# le premier tiers d'une reponse longue) sans reproduire un texte au style
# essai en entier. Hypothese testee puis ecartee comme cause de
# degenerescence (cf. docstring du module) ; gardee par prudence.
LONGUEUR_MAX_ENTREE_REFORMULATION = 1500


def _tronquer_texte_chosen(texte: str) -> str:
    """
    Coupe `texte` a `LONGUEUR_MAX_ENTREE_REFORMULATION` caracteres au
    maximum, sur une limite de phrase (`. `/`! `/`? `) ou, a defaut, de
    paragraphe (`\n`), jamais en plein mot : un texte coupe de facon
    abrupte pourrait perturber le modele davantage qu'aider. En dernier
    recours (aucune limite de phrase/paragraphe trouvee dans la fenetre),
    coupe simplement a la longueur max.
    """
    if len(texte) <= LONGUEUR_MAX_ENTREE_REFORMULATION:
        return texte

    extrait = texte[:LONGUEUR_MAX_ENTREE_REFORMULATION]
    for separateur in (". ", "! ", "? "):
        position = extrait.rfind(separateur)
        if position > 0:
            return extrait[: position + 1]

    position_paragraphe = extrait.rfind("\n")
    if position_paragraphe > 0:
        return extrait[:position_paragraphe].rstrip()

    return extrait.rstrip()


def _horodatage_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Taille max de `echantillon_echecs_reformulation` : diagnostic borne en
# memoire (un lot peut compter des milliers d'echecs), pas un echantillon
# statistique. 20 (pas 5) pour voir la distribution reelle des echecs
# (vide vs. charabia vs. quasi-correct) sans multiplier les runs GPU.
TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION = 20


@dataclass(slots=True)
class ReformulerPreferenceDpoUseCase:
    """Orchestre la reformulation incrementale/resumable du `chosen` d'un sous-ensemble DPO."""

    moteur: MoteurInference
    repository_reformule: RepositoryLectureEcriture
    taille_cible: int = 5000
    horloge: Callable[[], str] = _horodatage_utc_iso

    nombre_echecs_reformulation: int = field(default=0, init=False)
    echantillon_echecs_reformulation: list[EchecReformulation] = field(
        default_factory=list, init=False
    )

    def executer(self, source_dpo: Iterable[ExemplePivot]) -> int:
        """
        Filtre `source_dpo` sur `TypeExemple.DPO`, exclut les
        identifiants deja dans `repository_reformule`, et reformule au
        plus `taille_cible - (deja reformules)` nouveaux candidats. Un
        echec (inference ou format) compte dans
        `nombre_echecs_reformulation` et n'ecrit jamais d'exemple
        partiel ; un echec de format garde un `EchecReformulation`
        jusqu'a `TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION` elements,
        pur diagnostic. Persiste en une seule `sauvegarder_plusieurs()`.
        Retourne le nombre reformule lors de CETTE execution.
        """
        deja_reformules = self.repository_reformule.identifiants_existants()
        candidats = [
            exemple
            for exemple in source_dpo
            if exemple.type_exemple == TypeExemple.DPO
            and exemple.identifiant not in deja_reformules
        ]

        nombre_restant = max(0, self.taille_cible - len(deja_reformules))
        candidats = candidats[:nombre_restant]

        self.nombre_echecs_reformulation = 0
        self.echantillon_echecs_reformulation = []
        reformules: list[ChosenReformule] = []
        for exemple in candidats:
            texte_chosen_original = "\n".join(
                message.contenu for message in exemple.chosen
            )
            texte_chosen_envoye = _tronquer_texte_chosen(texte_chosen_original)
            contenu_utilisateur = f"{PROMPT_REFORMULATION_CHOSEN}\n\nReponse a reformuler :\n{texte_chosen_envoye}"
            messages = [{"role": "user", "content": contenu_utilisateur}]
            parametres_generation = {
                "n_predict": NOMBRE_TOKENS_GENERES_REFORMULATION,
                "temperature": TEMPERATURE_REFORMULATION,
                "min_new_tokens": MIN_TOKENS_GENERES_REFORMULATION,
            }
            try:
                reponse = self.moteur.generer(messages, parametres_generation)
            except Exception:  # noqa: BLE001 - erreur adaptateur concrete, le domaine ne la type pas
                self.nombre_echecs_reformulation += 1
                continue

            chosen_reformule = parser_reformulation_stricte(reponse.texte)
            if chosen_reformule is None:
                self.nombre_echecs_reformulation += 1
                if (
                    len(self.echantillon_echecs_reformulation)
                    < TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION
                ):
                    self.echantillon_echecs_reformulation.append(
                        EchecReformulation(
                            entree=texte_chosen_envoye,
                            sortie_brute=reponse.texte,
                            nombre_tokens_entree=reponse.nombre_tokens_entree,
                            nombre_tokens_sortie=reponse.nombre_tokens_sortie,
                        )
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
