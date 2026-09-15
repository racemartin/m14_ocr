"""
Tests de EvaluerBaselineZeroShotUseCase, avec de faux adaptateurs en
memoire (meme patron que `test_E1_05_02_extraire_sous_ensemble_sft.py`) :
aucun reseau, aucun serveur llama.cpp reel, aucun GPU.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from chsa_triage.application.use_cases import (
    EvaluerBaselineZeroShotUseCase,
)
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple, TypeSplit
from chsa_triage.domain.ports.moteur_inference import ReponseModele


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire."""

    def __init__(self, items: list[ExemplePivot]) -> None:
        self.items: dict[str, ExemplePivot] = {item.identifiant: item for item in items}

    def sauvegarder(self, item: ExemplePivot) -> None:
        self.items[item.identifiant] = item

    def sauvegarder_plusieurs(self, items: Iterable[ExemplePivot]) -> None:
        for item in items:
            self.items[item.identifiant] = item

    def trouver_par_id(self, identifiant: str):
        return self.items.get(identifiant)

    def lister(self, filtre: dict | None = None):
        for exemple in self.items.values():
            if filtre is None or all(getattr(exemple, cle) == valeur for cle, valeur in filtre.items()):
                yield exemple

    def compter(self, filtre: dict | None = None) -> int:
        return sum(1 for _ in self.lister(filtre))


class FauxFormateur:
    """Rend juste le contenu concatene du prompt, precede d'un marqueur, sans vrai tokenizer."""

    def formater_invite_zero_shot(self, exemple: ExemplePivot) -> str:
        return "INVITE::" + " ".join(m.contenu for m in exemple.prompt)


class FauxSuivi:
    """Faux adaptateur SuiviExperimentation, en memoire : capture les appels sans MLflow reel."""

    def __init__(self) -> None:
        self.runs_demarres: list[tuple[str, dict]] = []
        self.metriques: list[tuple[str, float, int]] = []
        self.nombre_fins = 0

    def demarrer_run(self, nom: str, parametres: dict) -> None:
        self.runs_demarres.append((nom, parametres))

    def logger_metrique(self, nom: str, valeur: float, etape: int) -> None:
        self.metriques.append((nom, valeur, etape))

    def terminer_run(self) -> None:
        self.nombre_fins += 1


class FauxMoteurInference:
    """Retourne une reponse pre-configuree par identifiant d'invite (ou une valeur par defaut)."""

    def __init__(self, reponses_par_invite: dict[str, str] | None = None, reponse_defaut: str = "") -> None:
        self._reponses_par_invite = reponses_par_invite or {}
        self._reponse_defaut = reponse_defaut
        self.appels: list[tuple[list[dict], dict]] = []

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        self.appels.append((messages, parametres or {}))
        invite = messages[0]["content"]
        texte = self._reponses_par_invite.get(invite, self._reponse_defaut)
        return ReponseModele(texte=texte, nombre_tokens_entree=10, nombre_tokens_sortie=5, latence_ms=42.0)


def _exemple_sft_test(identifiant_suffixe: str, question: str, reponse: str) -> ExemplePivot:
    exemple = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test", identifiant_suffixe),
        source="Test",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        prompt=(Message(role="user", contenu=question),),
        completion=(Message(role="assistant", contenu=reponse),),
    )
    from dataclasses import replace

    return replace(exemple, anonymise=True, split=TypeSplit.TEST_CLINIQUE)


def test_executer_leve_si_aucun_exemple_test_sft():
    repository = FauxRepository([])
    suivi = FauxSuivi()
    cas_usage = EvaluerBaselineZeroShotUseCase(
        repository=repository, formateur=FauxFormateur(), moteur=FauxMoteurInference(), suivi=suivi
    )
    with pytest.raises(ValueError):
        cas_usage.executer()

    assert suivi.runs_demarres == []  # jamais demarre : rien a evaluer, rien a logger


def test_executer_ignore_les_exemples_hors_split_test_ou_hors_sft():
    from dataclasses import replace

    exemple_train = replace(_exemple_sft_test("a", "Q1 ?", "R1."), split=TypeSplit.TRAIN)
    exemple_dpo_test = ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("test", "b"),
        source="Test",
        type_exemple=TypeExemple.DPO,
        langue=Langue.ANGLAIS,
        prompt=(Message(role="user", contenu="Q2 ?"),),
        chosen=(Message(role="assistant", contenu="bonne"),),
        rejected=(Message(role="assistant", contenu="mauvaise"),),
    )
    exemple_dpo_test = replace(exemple_dpo_test, anonymise=True, split=TypeSplit.TEST_CLINIQUE)
    exemple_valide = _exemple_sft_test("c", "Q3 ?", "R3.")

    repository = FauxRepository([exemple_train, exemple_dpo_test, exemple_valide])
    moteur = FauxMoteurInference(reponse_defaut="R3.")
    cas_usage = EvaluerBaselineZeroShotUseCase(
        repository=repository, formateur=FauxFormateur(), moteur=moteur, suivi=FauxSuivi()
    )

    resultat = cas_usage.executer()

    assert resultat.nombre_exemples == 1
    assert len(moteur.appels) == 1


def test_executer_calcule_exact_match_1_quand_generation_egale_reference():
    exemple = _exemple_sft_test("a", "Quel est le niveau ESI ?", "Niveau 2, a surveiller.")
    repository = FauxRepository([exemple])
    moteur = FauxMoteurInference(reponse_defaut="Niveau 2, a surveiller.")

    cas_usage = EvaluerBaselineZeroShotUseCase(
        repository=repository, formateur=FauxFormateur(), moteur=moteur, suivi=FauxSuivi()
    )
    resultat = cas_usage.executer()

    assert resultat.nombre_exemples == 1
    assert resultat.exact_match == 1.0
    assert resultat.f1_moyen == 1.0
    assert resultat.latence_ms_moyenne == 42.0


def test_executer_exact_match_0_quand_generation_incorrecte_zero_shot():
    """
    Cas realiste d'une baseline zero-shot (modele non entraine) :
    generation totalement hors-sujet par rapport a la reference.
    """
    exemple = _exemple_sft_test("a", "Quel est le niveau ESI ?", "Niveau 2, a surveiller.")
    repository = FauxRepository([exemple])
    moteur = FauxMoteurInference(reponse_defaut="Je ne sais pas repondre a cette question.")

    cas_usage = EvaluerBaselineZeroShotUseCase(
        repository=repository, formateur=FauxFormateur(), moteur=moteur, suivi=FauxSuivi()
    )
    resultat = cas_usage.executer()

    assert resultat.exact_match == 0.0
    assert resultat.exactitude_niveau.exactitude is None  # aucun JSON dans ni l'un ni l'autre texte


def test_executer_passe_invite_deja_rendue_true_au_moteur():
    """Verifie que le cas d'usage utilise bien le mode `/completion` (invite pre-rendue) de l'adaptateur."""
    exemple = _exemple_sft_test("a", "Q ?", "R.")
    repository = FauxRepository([exemple])
    moteur = FauxMoteurInference(reponse_defaut="R.")

    cas_usage = EvaluerBaselineZeroShotUseCase(
        repository=repository, formateur=FauxFormateur(), moteur=moteur, suivi=FauxSuivi()
    )
    cas_usage.executer()

    messages, parametres = moteur.appels[0]
    assert parametres["invite_deja_rendue"] is True
    assert messages[0]["content"] == "INVITE::Q ?"


def test_executer_fusionne_parametres_generation_personnalises():
    exemple = _exemple_sft_test("a", "Q ?", "R.")
    repository = FauxRepository([exemple])
    moteur = FauxMoteurInference(reponse_defaut="R.")

    cas_usage = EvaluerBaselineZeroShotUseCase(
        repository=repository,
        formateur=FauxFormateur(),
        moteur=moteur,
        suivi=FauxSuivi(),
        parametres_generation={"temperature": 0.0, "n_predict": 128},
    )
    cas_usage.executer()

    _, parametres = moteur.appels[0]
    assert parametres["temperature"] == 0.0
    assert parametres["n_predict"] == 128


def test_executer_relaie_les_metriques_agregees_vers_suivi_experimentation():
    """
    Meme mecanisme que EntrainerSftUseCase (§B3) : demarrer_run puis
    logger_metrique par metrique agregee puis terminer_run, jamais un
    nouveau systeme de tracking.
    """
    exemple = _exemple_sft_test("a", "Quel est le niveau ESI ?", "Niveau 2, a surveiller.")
    repository = FauxRepository([exemple])
    moteur = FauxMoteurInference(reponse_defaut="Niveau 2, a surveiller.")
    suivi = FauxSuivi()

    cas_usage = EvaluerBaselineZeroShotUseCase(
        repository=repository, formateur=FauxFormateur(), moteur=moteur, suivi=suivi
    )
    cas_usage.executer()

    assert len(suivi.runs_demarres) == 1
    nom_run, parametres_run = suivi.runs_demarres[0]
    assert nom_run == "baseline-zero-shot"
    assert parametres_run["nombre_exemples"] == 1

    noms_metriques = {nom for nom, _, _ in suivi.metriques}
    assert noms_metriques == {"exact_match", "f1_moyen", "latence_ms_moyenne"}  # exactitude_niveau non loggee : None
    assert suivi.nombre_fins == 1
