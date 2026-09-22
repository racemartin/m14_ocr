"""
Tests de `FormaterDatasetChatMLPreferenceUseCase`, avec de faux
adaptateurs en memoire (pas de tokenizer, pas de fichier reel), meme
patron que `tests/application/test_E2_00_uc_formater_dataset_chatml.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from chsa_triage.application.use_cases.E3_01_uc_formater_dataset_chatml_preference import (
    FormaterDatasetChatMLPreferenceUseCase,
)
from chsa_triage.domain.model.enums import Langue, TypeExemple, TypeSplit
from chsa_triage.domain.model.exemple_formate_preference import ExempleFormatePreference
from chsa_triage.domain.model.exemple_pivot import ExemplePivot, Message
from chsa_triage.domain.model.preference_reformulee import ChosenReformule


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire, filtre inclus."""

    def __init__(self, items=None) -> None:
        self.items: dict[str, object] = {item.identifiant: item for item in (items or [])}

    def sauvegarder(self, item) -> None:
        self.items[item.identifiant] = item

    def sauvegarder_plusieurs(self, items: Iterable) -> None:
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

    def identifiants_existants(self) -> set[str]:
        return set(self.items.keys())


class FauxFormateurPreference:
    """Faux adaptateur FormateurPreference : concatene prompt/chosen/rejected sans tokenizer reel."""

    def __init__(self) -> None:
        self.appels: list[ExemplePivot] = []

    def formater_preference(self, exemple: ExemplePivot) -> ExempleFormatePreference:
        self.appels.append(exemple)
        return ExempleFormatePreference(
            identifiant=exemple.identifiant,
            texte_prompt=" ".join(m.contenu for m in exemple.prompt),
            texte_chosen=" ".join(m.contenu for m in exemple.chosen),
            texte_rejected=" ".join(m.contenu for m in exemple.rejected),
        )


def _exemple_dpo(split: TypeSplit | None, chosen_original: str = "chosen original") -> ExemplePivot:
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("UltraMedical-Preference", cle),
        identifiant_source_brute=cle,
        source="UltraMedical-Preference",
        type_exemple=TypeExemple.DPO,
        langue=Langue.ANGLAIS,
        prompt=(Message(role="user", contenu="Question ?"),),
        chosen=(Message(role="assistant", contenu=chosen_original),),
        rejected=(Message(role="assistant", contenu="mauvaise reponse"),),
        split=split,
    )


def _exemple_sft(split: TypeSplit | None) -> ExemplePivot:
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("MediQAl", cle),
        identifiant_source_brute=cle,
        source="MediQAl",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        prompt=(Message(role="user", contenu="Question ?"),),
        completion=(Message(role="assistant", contenu="Reponse."),),
        split=split,
    )


def _chosen_reformule(identifiant: str, texte: str = "chosen reformule <think>...</think>{}") -> ChosenReformule:
    return ChosenReformule(
        identifiant=identifiant,
        chosen_reformule=(Message(role="assistant", contenu=texte),),
        horodatage="2026-09-19T00:00:00Z",
    )


def test_exemple_sans_chosen_reformule_utilise_chosen_original_en_fallback():
    """DECISION DESACOPLADA : un exemple sans ChosenReformule correspondant
    (hors du sous-ensemble reformule) est PERSISTE avec le chosen original,
    ne sera plus exclus. Cela desacouple le formatage DPO de la reformulation.
    """
    exemple_avec_reformulation = _exemple_dpo(TypeSplit.TRAIN, chosen_original="chosen original A")
    exemple_sans_reformulation = _exemple_dpo(TypeSplit.TRAIN, chosen_original="chosen original B")
    repository_pivot = FauxRepository([exemple_avec_reformulation, exemple_sans_reformulation])
    repository_reformule = FauxRepository([_chosen_reformule(exemple_avec_reformulation.identifiant, texte="chosen reformule A")])
    repository_formate = FauxRepository()
    formateur = FauxFormateurPreference()

    cas_usage = FormaterDatasetChatMLPreferenceUseCase(
        repository_pivot=repository_pivot,
        repository_reformule=repository_reformule,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    nombre = cas_usage.executer(TypeSplit.TRAIN)

    assert nombre == 2
    assert set(repository_formate.items) == {exemple_avec_reformulation.identifiant, exemple_sans_reformulation.identifiant}


def test_repository_reformule_none_utilise_tous_les_chosen_originaux():
    """DECISION DESACOPLADA : quand repository_reformule est None (mode
    --skip-reformulation), tous les exemples DPO sont formates avec leur
    chosen original tel quel — aucune exclusion.
    """
    exemple_1 = _exemple_dpo(TypeSplit.TRAIN, chosen_original="chosen original 1")
    exemple_2 = _exemple_dpo(TypeSplit.TRAIN, chosen_original="chosen original 2")
    repository_pivot = FauxRepository([exemple_1, exemple_2])
    repository_formate = FauxRepository()
    formateur = FauxFormateurPreference()

    cas_usage = FormaterDatasetChatMLPreferenceUseCase(
        repository_pivot=repository_pivot,
        repository_reformule=None,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    nombre = cas_usage.executer(TypeSplit.TRAIN)

    assert nombre == 2
    assert set(repository_formate.items) == {exemple_1.identifiant, exemple_2.identifiant}
    # Verifie que le chosen original a ete utilise (pas de reformulation)
    for item in repository_formate.items.values():
        assert "chosen original" in item.texte_chosen


def test_exclut_les_exemples_sft():
    """Meme bug de fuite deja documente et corrige en Etape 1, cf. AGENTS.md "SFT/DPO type leak"."""
    exemple_dpo = _exemple_dpo(TypeSplit.TRAIN)
    exemple_sft = _exemple_sft(TypeSplit.TRAIN)
    repository_pivot = FauxRepository([exemple_dpo, exemple_sft])
    repository_reformule = FauxRepository([_chosen_reformule(exemple_dpo.identifiant)])
    repository_formate = FauxRepository()
    formateur = FauxFormateurPreference()

    cas_usage = FormaterDatasetChatMLPreferenceUseCase(
        repository_pivot=repository_pivot,
        repository_reformule=repository_reformule,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    nombre = cas_usage.executer(TypeSplit.TRAIN)

    assert nombre == 1
    assert set(repository_formate.items) == {exemple_dpo.identifiant}


def test_ne_traite_que_le_split_demande():
    exemple_train = _exemple_dpo(TypeSplit.TRAIN)
    exemple_val = _exemple_dpo(TypeSplit.VALIDATION)
    repository_pivot = FauxRepository([exemple_train, exemple_val])
    repository_reformule = FauxRepository(
        [_chosen_reformule(exemple_train.identifiant), _chosen_reformule(exemple_val.identifiant)]
    )
    repository_formate = FauxRepository()
    formateur = FauxFormateurPreference()

    cas_usage = FormaterDatasetChatMLPreferenceUseCase(
        repository_pivot=repository_pivot,
        repository_reformule=repository_reformule,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    nombre = cas_usage.executer(TypeSplit.TRAIN)

    assert nombre == 1
    assert set(repository_formate.items) == {exemple_train.identifiant}


def test_le_chosen_reformule_remplace_le_chosen_original_avant_formatage():
    exemple = _exemple_dpo(TypeSplit.TEST_CLINIQUE, chosen_original="chosen original jamais vu")
    texte_reformule = "<think>raisonnement</think>{\"niveau\": 2}"
    repository_pivot = FauxRepository([exemple])
    repository_reformule = FauxRepository([_chosen_reformule(exemple.identifiant, texte=texte_reformule)])
    repository_formate = FauxRepository()
    formateur = FauxFormateurPreference()

    cas_usage = FormaterDatasetChatMLPreferenceUseCase(
        repository_pivot=repository_pivot,
        repository_reformule=repository_reformule,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    cas_usage.executer(TypeSplit.TEST_CLINIQUE)

    assert len(formateur.appels) == 1
    exemple_recu = formateur.appels[0]
    assert exemple_recu.chosen == (Message(role="assistant", contenu=texte_reformule),)
    assert "chosen original jamais vu" not in " ".join(m.contenu for m in exemple_recu.chosen)

    resultat = repository_formate.items[exemple.identifiant]
    assert texte_reformule in resultat.texte_chosen
    assert "chosen original jamais vu" not in resultat.texte_chosen


def test_ne_mute_jamais_le_pivot_ni_le_registre_de_reformulation_source():
    exemple = _exemple_dpo(TypeSplit.TRAIN, chosen_original="chosen original")
    chosen_reformule = _chosen_reformule(exemple.identifiant)
    repository_pivot = FauxRepository([exemple])
    repository_reformule = FauxRepository([chosen_reformule])
    repository_formate = FauxRepository()
    formateur = FauxFormateurPreference()

    cas_usage = FormaterDatasetChatMLPreferenceUseCase(
        repository_pivot=repository_pivot,
        repository_reformule=repository_reformule,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    cas_usage.executer(TypeSplit.TRAIN)

    assert repository_pivot.items[exemple.identifiant].chosen == (Message(role="assistant", contenu="chosen original"),)
    assert repository_reformule.items[exemple.identifiant] == chosen_reformule


def test_sans_exemple_dans_le_split_ne_persiste_rien():
    repository_pivot = FauxRepository([_exemple_dpo(TypeSplit.TRAIN)])
    repository_reformule = FauxRepository()
    repository_formate = FauxRepository()
    formateur = FauxFormateurPreference()

    cas_usage = FormaterDatasetChatMLPreferenceUseCase(
        repository_pivot=repository_pivot,
        repository_reformule=repository_reformule,
        repository_formate=repository_formate,
        formateur=formateur,
    )
    nombre = cas_usage.executer(TypeSplit.VALIDATION)

    assert nombre == 0
    assert len(repository_formate.items) == 0
    assert formateur.appels == []
