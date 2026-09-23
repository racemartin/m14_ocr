"""
Cas d'usage : evaluer la baseline ZERO-SHOT de `Qwen/Qwen3-1.7B-Base`
(SANS entrainement) sur le split `test` deja existant du pivot
anonymise, pour disposer d'un point de comparaison mesurable avant
SFT/DPO (cahier des charges §9 : "l'accuracy de classification ESI...
depasse la baseline zero-shot de facon mesurable").

Etape 1bis (pas Etape 2) : ne forme, n'entraine, ne modifie rien ;
lit des `ExemplePivot` deja repartis en split (§7 README), genere une
reponse zero-shot pour chacun via `MoteurInference`, et compare chaque
generation a la `completion` reelle via les metriques pures de
`application/metriques_evaluation_baseline.py`.
"""

from __future__ import annotations  # Annotations de type differees

# Bibliotheque standard
from dataclasses import dataclass, field  # dataclass et champ a valeur par defaut

# Bibliotheques du projet (couche application/domaine)
from chsa_triage.application.metriques_evaluation_baseline import (  # Metriques de comparaison reponse/reference
    ResultatExactitudeNiveau,
    exactitude_classification_niveau,
    f1_moyen,
    taux_exact_match,
)
from chsa_triage.domain.model.enums                        import TypeExemple, TypeSplit  # Enums de filtrage du repository
from chsa_triage.domain.model.exemple_pivot                import ExemplePivot  # Entite pivot (prompt/completion)
from chsa_triage.domain.ports.dataset_repository           import RepositoryLectureEcriture  # Port de lecture du dataset
from chsa_triage.domain.ports.formateur_invite_zero_shot   import FormateurInviteZeroShot  # Port : formate l'invite zero-shot
from chsa_triage.domain.ports.moteur_inference             import MoteurInference  # Port : moteur d'inference
from chsa_triage.domain.ports.suivi_experimentation        import SuiviExperimentation  # Port : suivi du run (MLflow)
from tools.rafael.log_tool                                 import LogTool  # Log warning par exemple en echec, sans abandonner le run

NOM_RUN_PAR_DEFAUT = "baseline-zero-shot"

log = LogTool(origin="evaluer_baseline_zero_shot")

# Taille max de `echantillon_generations` : diagnostic (inspecter reellement
# ce que le modele genere, ex. verbosite/format, plutot que deviner a partir
# des seules metriques agregees), jamais une garantie de couverture ; meme
# principe et meme taille que `TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION`
# dans `E3_00_uc_reformuler_preference_dpo.py` (23/09/2026).
TAILLE_MAX_ECHANTILLON_GENERATIONS = 20


@dataclass(frozen=True, slots=True)
class ResultatEvaluationBaseline:
    """Resultat agrege d'un run d'evaluation baseline zero-shot."""

    nombre_exemples          : int
    exact_match              : float
    f1_moyen                 : float
    exactitude_niveau        : ResultatExactitudeNiveau
    latence_ms_moyenne       : float
    nombre_echecs_inference  : int = 0


# ##############################################################################
def _extraire_texte_reponse_reelle(exemple: ExemplePivot) -> str:
    """
    `completion` est un tuple de `Message` (assistant seul en
    pratique) ; concatene leur `contenu`.
    """
    return "\n".join(message.contenu for message in exemple.completion)


@dataclass(slots=True)
class EvaluerBaselineZeroShotUseCase:
    """
    Orchestre l'evaluation baseline zero-shot d'un `MoteurInference` sur
    le split test, et relaie le resultat agrege vers `SuiviExperimentation`
    (meme mecanisme que `EntrainerSftUseCase`, pas un nouveau systeme de
    tracking : demarrer_run/logger_metrique/terminer_run).
    """

    repository             : RepositoryLectureEcriture
    formateur              : FormateurInviteZeroShot
    moteur                 : MoteurInference
    suivi                  : SuiviExperimentation
    parametres_generation  : dict = field(default_factory=dict)
    nom_run                : str  = NOM_RUN_PAR_DEFAUT

    # Jusqu'a TAILLE_MAX_ECHANTILLON_GENERATIONS paires (texte genere, texte
    # reference), dans l'ordre du split test.
    echantillon_generations: list[tuple[str, str]] = field(default_factory=list, init=False)

    # ##########################################################################
    def executer(self) -> ResultatEvaluationBaseline:
        """
        Leve `ValueError` si le split test ne contient aucun exemple
        SFT (rien a evaluer) : jamais un resultat agrege silencieusement
        vide, cf. la meme regle explicite que les autres cas d'usage de
        ce projet (ex. `DecouperSplitsUseCase`). Le run de suivi n'est
        demarre qu'APRES cette verification (rien a logger si on leve).
        """
        self.echantillon_generations = []

        # ----- CHARGEMENT ET VALIDATION DU SPLIT TEST -------------------------
        exemples = list(
            self.repository.lister(
                filtre={
                    "split": TypeSplit.TEST_CLINIQUE,
                    "type_exemple": TypeExemple.SFT,
                }
            )
        )
        if not exemples:
            raise ValueError(
                "aucun exemple type_exemple=SFT avec split=test dans le "
                "repository : rien a evaluer"
            )

        # ----- DEMARRAGE DU SUIVI D'EXPERIMENTATION ---------------------------
        self.suivi.demarrer_run(
            self.nom_run,
            {"nombre_exemples": len(exemples), **self.parametres_generation},
        )

        # ----- GENERATION ZERO-SHOT ET COLLECTE DES PAIRES --------------------
        # Un exemple degenere (generation corrompue rejetee par le serveur
        # d'inference, ex. 500 sur un checkpoint BASE pres de n_predict) ne
        # doit jamais abandonner tout le run : on logue et on continue.
        paires      : list[tuple[str, str]] = []
        latences_ms : list[float]           = []
        nombre_echecs_inference = 0
        for indice, exemple in enumerate(exemples):
            invite = self.formateur.formater_invite_zero_shot(exemple)
            parametres = {
                "invite_deja_rendue": True,
                **self.parametres_generation,
            }
            try:
                reponse = self.moteur.generer(
                    [{"role": "user", "content": invite}], parametres
                )
            except Exception as erreur:
                nombre_echecs_inference += 1
                log.LEVEL_5_WARNING(
                    "EvaluerBaselineZeroShotUseCase",
                    f"echec d'inference sur l'exemple {indice} "
                    f"(identifiant={exemple.identifiant}) : {erreur} ; "
                    "exemple ignore, run poursuivi",
                )
                continue
            paire = (reponse.texte, _extraire_texte_reponse_reelle(exemple))
            paires.append(paire)
            if len(self.echantillon_generations) < TAILLE_MAX_ECHANTILLON_GENERATIONS:
                self.echantillon_generations.append(paire)
            latences_ms.append(reponse.latence_ms)

        if not paires:
            raise ValueError(
                "tous les exemples ont echoue a l'inference "
                f"({nombre_echecs_inference}/{len(exemples)}) : rien a agreger"
            )

        # ----- AGREGATION DES METRIQUES ---------------------------------------
        resultat = ResultatEvaluationBaseline(
            nombre_exemples=len(exemples),
            exact_match=taux_exact_match(paires),
            f1_moyen=f1_moyen(paires),
            exactitude_niveau=exactitude_classification_niveau(paires),
            latence_ms_moyenne=sum(latences_ms) / len(latences_ms),
            nombre_echecs_inference=nombre_echecs_inference,
        )

        # ----- JOURNALISATION DES METRIQUES ET CLOTURE DU RUN -----------------
        self.suivi.logger_metrique("exact_match", resultat.exact_match, 0)
        self.suivi.logger_metrique("f1_moyen", resultat.f1_moyen, 0)
        if resultat.exactitude_niveau.exactitude is not None:
            self.suivi.logger_metrique(
                "exactitude_niveau_triage",
                resultat.exactitude_niveau.exactitude,
                0,
            )
        self.suivi.logger_metrique(
            "latence_ms_moyenne", resultat.latence_ms_moyenne, 0
        )
        self.suivi.logger_metrique(
            "nombre_echecs_inference", resultat.nombre_echecs_inference, 0
        )
        self.suivi.terminer_run()

        return resultat
