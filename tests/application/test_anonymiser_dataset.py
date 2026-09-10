"""
Tests de la couche application pour AnonymiserDatasetUseCase, avec de
faux adaptateurs en memoire (pas de Presidio, pas de fichier reel).

Design source/sortie separes (08/09/2026) : `repository_source` n'est
JAMAIS ecrit par le cas d'usage ; seul `repository_sortie` recoit les
versions anonymisees. "Deja traite" se determine par la presence de
l'identifiant dans `repository_sortie` (`identifiants_existants()`),
pas par un booleen mute sur l'exemple source.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from chsa_triage.application.use_cases import AnonymiserDatasetUseCase
from chsa_triage.domain.model import ExemplePivot, Langue, Message, TypeExemple
from chsa_triage.domain.ports.anonymiseur import EntiteDetectee, ResultatAnonymisation


class FauxRepository:
    """Faux adaptateur RepositoryLectureEcriture, en memoire, filtre inclus."""

    def __init__(self, items: list[ExemplePivot] | None = None) -> None:
        self.items: dict[str, ExemplePivot] = {item.identifiant: item for item in (items or [])}

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

    def identifiants_existants(self) -> set[str]:
        return set(self.items.keys())


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
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant(source, cle),
        identifiant_source_brute=cle,
        source=source,
        type_exemple=type_exemple,
        langue=langue,
        symptomes="Fievre et toux",
        prompt=(Message(role="user", contenu="Question ?"),),
        completion=(Message(role="assistant", contenu="Reponse."),),
    )


def test_anonymiser_dataset_traite_tout_sans_limite():
    exemples = [_exemple("MediQAl") for _ in range(5)]
    source = FauxRepository(exemples)
    sortie = FauxRepository()
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(
        repository_source=source, repository_sortie=sortie, anonymiseur=anonymiseur, limite=None
    )
    nombre = cas_usage.executer()

    assert nombre == 5
    assert len(sortie.items) == 5
    assert all(e.anonymise for e in sortie.items.values())
    assert all("[ANON]" in e.symptomes for e in sortie.items.values())
    # Le pivot source n'est JAMAIS modifie.
    assert all(not e.anonymise for e in source.items.values())
    assert all(e.symptomes == "Fievre et toux" for e in source.items.values())


def test_anonymiser_dataset_ne_retraite_jamais_les_exemples_deja_presents_en_sortie():
    exemple_deja_fait = _exemple("MediQAl")
    exemple_a_faire = _exemple("MediQAl")
    source = FauxRepository([exemple_deja_fait, exemple_a_faire])
    # Le fichier de sortie contient DEJA une version anonymisee de exemple_deja_fait.
    sortie = FauxRepository([
        ExemplePivot(
            identifiant=exemple_deja_fait.identifiant,
            identifiant_source_brute=exemple_deja_fait.identifiant_source_brute,
            source=exemple_deja_fait.source,
            type_exemple=exemple_deja_fait.type_exemple,
            langue=exemple_deja_fait.langue,
            symptomes="[DEJA ANONYMISE]",
            anonymise=True,
        )
    ])
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(
        repository_source=source, repository_sortie=sortie, anonymiseur=anonymiseur, limite=None
    )
    nombre = cas_usage.executer()

    assert nombre == 1  # seul exemple_a_faire est traite
    assert sortie.items[exemple_deja_fait.identifiant].symptomes == "[DEJA ANONYMISE]"  # inchange
    assert "[ANON]" in sortie.items[exemple_a_faire.identifiant].symptomes


def test_anonymiser_dataset_ne_modifie_jamais_le_repository_source():
    exemples = [_exemple("MediQAl") for _ in range(3)]
    source = FauxRepository(exemples)
    sortie = FauxRepository()
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(
        repository_source=source, repository_sortie=sortie, anonymiseur=anonymiseur, limite=None
    )
    cas_usage.executer()

    # Aucune ecriture sur `source` : memes objets, meme texte, `anonymise` toujours False.
    for identifiant, exemple_original in {e.identifiant: e for e in exemples}.items():
        assert source.items[identifiant] is exemple_original
        assert source.items[identifiant].anonymise is False


def test_anonymiser_dataset_limite_selectionne_un_echantillon_stratifie():
    exemples = (
        [_exemple("MediQAl", TypeExemple.SFT) for _ in range(80)]
        + [_exemple("MedQuAD", TypeExemple.SFT) for _ in range(15)]
        + [_exemple("UltraMedical-Preference", TypeExemple.DPO) for _ in range(5)]
    )
    source = FauxRepository(exemples)
    sortie = FauxRepository()
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(
        repository_source=source, repository_sortie=sortie, anonymiseur=anonymiseur, limite=20, graine_aleatoire=42
    )
    nombre = cas_usage.executer()

    assert nombre == 20
    assert len(sortie.items) == 20
    # Chaque strate doit etre representee dans l'echantillon (80/15/5 sur 100 -> ~16/3/1).
    sources_traitees = {e.source for e in sortie.items.values()}
    assert sources_traitees == {"MediQAl", "MedQuAD", "UltraMedical-Preference"}
    # Rien dans le fichier source n'a ete touche, les 100 exemples y restent.
    assert len(source.items) == 100


def test_anonymiser_dataset_limite_superieure_au_reste_traite_tout():
    exemples = [_exemple("MediQAl") for _ in range(3)]
    source = FauxRepository(exemples)
    sortie = FauxRepository()
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(
        repository_source=source, repository_sortie=sortie, anonymiseur=anonymiseur, limite=5000
    )
    nombre = cas_usage.executer()

    assert nombre == 3


def test_anonymiser_dataset_accumule_les_statistiques_rgpd_par_source():
    """
    Instrumentation minimale (section 3 du rapport RGPD) : le cas
    d'usage doit exposer, par source, le nombre de registres traites,
    le nombre de registres avec au moins une entite detectee, et le
    detail des entites par type ; sans quoi ces chiffres devraient
    etre estimes a la main.
    """
    exemples = (
        [_exemple("MediQAl", TypeExemple.SFT) for _ in range(2)]
        + [_exemple("MedQuAD", TypeExemple.SFT) for _ in range(1)]
    )
    source = FauxRepository(exemples)
    sortie = FauxRepository()
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(
        repository_source=source, repository_sortie=sortie, anonymiseur=anonymiseur, limite=None
    )
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
    source = FauxRepository(exemples)
    sortie = FauxRepository()
    anonymiseur = FauxAnonymiseur()

    cas_usage = AnonymiserDatasetUseCase(
        repository_source=source, repository_sortie=sortie, anonymiseur=anonymiseur, limite=None
    )
    nombre = cas_usage.executer(envelopper_iterable=None)

    assert nombre == 2


def test_anonymiser_dataset_accepte_un_envelopper_iterable_pour_la_progression():
    """`envelopper_iterable` (typiquement une barre tqdm) enveloppe l'echantillon deja stratifie/limite."""
    exemples = (
        [_exemple("MediQAl", TypeExemple.SFT) for _ in range(80)]
        + [_exemple("MedQuAD", TypeExemple.SFT) for _ in range(15)]
        + [_exemple("UltraMedical-Preference", TypeExemple.DPO) for _ in range(5)]
    )
    source = FauxRepository(exemples)
    sortie = FauxRepository()
    anonymiseur = FauxAnonymiseur()

    appels: list[int] = []

    def envelopper(iterable):
        elements = list(iterable)
        appels.append(len(elements))
        return elements

    cas_usage = AnonymiserDatasetUseCase(
        repository_source=source, repository_sortie=sortie, anonymiseur=anonymiseur, limite=20, graine_aleatoire=42
    )
    nombre = cas_usage.executer(envelopper_iterable=envelopper)

    assert nombre == 20
    # L'iterable enveloppe est bien l'echantillon deja stratifie (20), pas les 100 candidats.
    assert appels == [20]
