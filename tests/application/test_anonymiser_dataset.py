"""
Tests de la couche application pour AnonymiserDatasetUseCase, avec de
faux adaptateurs en memoire (pas de Presidio, pas de fichier reel).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from chsa_triage.application.use_cases import AnonymiserDatasetUseCase
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple
from chsa_triage.domain.ports.anonymiseur import EntiteDetectee, ResultatAnonymisation


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire, filtre inclus."""

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


class FauxAnonymiseur:
    """Faux adaptateur Anonymiseur : marque le texte sans dependance a Presidio."""

    def __init__(self) -> None:
        self.appels = 0

    def anonymiser(self, texte: str, langue: str) -> ResultatAnonymisation:
        self.appels += 1
        if not texte:
            return ResultatAnonymisation(texte_original=texte, texte_anonymise=texte, entites_detectees=())
        return ResultatAnonymisation(
            texte_original=texte,
            texte_anonymise=f"[ANON]{texte}",
            entites_detectees=(EntiteDetectee(type_entite="PERSON", debut=0, fin=1, score=1.0),),
        )


def _exemple(source: str, type_exemple: TypeExemple = TypeExemple.SFT, langue: Langue = Langue.FRANCAIS) -> ExemplePivot:
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant(source),
        source=source,
        type_exemple=type_exemple,
        langue=langue,
        symptomes="Fievre et toux",
        prompt=(Message(role="user", contenu="Question ?"),),
        completion=(Message(role="assistant", contenu="Reponse."),),
    )


def test_anonymiser_dataset_traite_tout_sans_limite():
    exemples = [_exemple("MediQAl") for _ in range(5)]
    repository = FauxRepository(exemples)
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur, limite=None)
    nombre = cas_usage.executer()

    assert nombre == 5
    assert all(e.anonymise for e in repository.items.values())
    assert all("[ANON]" in e.symptomes for e in repository.items.values())


def test_anonymiser_dataset_ne_retraite_jamais_les_exemples_deja_anonymises():
    deja_fait = replace(_exemple("MediQAl"), anonymise=True)
    a_faire = _exemple("MediQAl")
    repository = FauxRepository([deja_fait, a_faire])
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur, limite=None)
    nombre = cas_usage.executer()

    assert nombre == 1
    assert anonymiseur.appels > 0
    # L'exemple deja anonymise n'a pas ete retouche (toujours son texte d'origine).
    assert repository.items[deja_fait.identifiant].symptomes == "Fievre et toux"


def test_anonymiser_dataset_limite_selectionne_un_echantillon_stratifie():
    exemples = (
        [_exemple("MediQAl", TypeExemple.SFT) for _ in range(80)]
        + [_exemple("MedQuAD", TypeExemple.SFT) for _ in range(15)]
        + [_exemple("UltraMedical-Preference", TypeExemple.DPO) for _ in range(5)]
    )
    repository = FauxRepository(exemples)
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur, limite=20, graine_aleatoire=42)
    nombre = cas_usage.executer()

    assert nombre == 20
    traites = [e for e in repository.items.values() if e.anonymise]
    assert len(traites) == 20
    # Chaque strate doit etre representee dans l'echantillon (80/15/5 sur 100 -> ~16/3/1).
    sources_traitees = {e.source for e in traites}
    assert sources_traitees == {"MediQAl", "MedQuAD", "UltraMedical-Preference"}
    restants = [e for e in repository.items.values() if not e.anonymise]
    assert len(restants) == 80


def test_anonymiser_dataset_limite_superieure_au_reste_traite_tout():
    exemples = [_exemple("MediQAl") for _ in range(3)]
    repository = FauxRepository(exemples)
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur, limite=5000)
    nombre = cas_usage.executer()

    assert nombre == 3


def test_anonymiser_dataset_accumule_les_statistiques_rgpd_par_source():
    """
    Instrumentation minimale (section 3 du rapport RGPD) : le cas
    d'usage doit exposer, par source, le nombre de registres traites,
    le nombre de registres avec au moins une entite detectee, et le
    detail des entites par type -- sans quoi ces chiffres devraient
    etre estimes a la main.
    """
    exemples = (
        [_exemple("MediQAl", TypeExemple.SFT) for _ in range(2)]
        + [_exemple("MedQuAD", TypeExemple.SFT) for _ in range(1)]
    )
    repository = FauxRepository(exemples)
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur, limite=None)
    cas_usage.executer()

    assert set(cas_usage.statistiques) == {"MediQAl", "MedQuAD"}

    stats_mediqal = cas_usage.statistiques["MediQAl"]
    assert stats_mediqal.registres_traites == 2
    assert stats_mediqal.registres_avec_entite == 2
    # FauxAnonymiseur detecte un PERSON par champ non vide (symptomes + prompt + completion).
    assert stats_mediqal.entites_par_type == {"PERSON": 6}

    stats_medquad = cas_usage.statistiques["MedQuAD"]
    assert stats_medquad.registres_traites == 1
    assert stats_medquad.registres_avec_entite == 1


def test_anonymiser_dataset_fonctionne_sans_barre_de_progression():
    """Sans `envelopper_iterable`, le comportement est inchange (mode non interactif/tests)."""
    exemples = [_exemple("MediQAl") for _ in range(2)]
    repository = FauxRepository(exemples)
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur, limite=None)
    nombre = cas_usage.executer(envelopper_iterable=None)

    assert nombre == 2


def test_anonymiser_dataset_accepte_un_envelopper_iterable_pour_la_progression():
    """`envelopper_iterable` (typiquement une barre tqdm) enveloppe l'echantillon deja stratifie/limite."""
    exemples = (
        [_exemple("MediQAl", TypeExemple.SFT) for _ in range(80)]
        + [_exemple("MedQuAD", TypeExemple.SFT) for _ in range(15)]
        + [_exemple("UltraMedical-Preference", TypeExemple.DPO) for _ in range(5)]
    )
    repository = FauxRepository(exemples)
    anonymiseur = FauxAnonymiseur()

    appels: list[int] = []

    def envelopper(iterable):
        elements = list(iterable)
        appels.append(len(elements))
        return elements

    cas_usage = AnonymiserDatasetUseCase(repository=repository, anonymiseur=anonymiseur, limite=20, graine_aleatoire=42)
    nombre = cas_usage.executer(envelopper_iterable=envelopper)

    assert nombre == 20
    # L'iterable enveloppe est bien l'echantillon deja stratifie (20), pas les 100 candidats.
    assert appels == [20]
