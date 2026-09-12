"""
Tests de `FormaterDatasetChatMLUseCase`, avec de faux adaptateurs en
memoire (pas de tokenizer, pas de fichier reel), meme patron que
`tests/application/test_E1_04_00_anonymiser_dataset.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from chsa_triage.application.use_cases import FormaterDatasetChatMLUseCase
from chsa_triage.domain.model import ExempleFormate, ExemplePivot, Langue, Message, TypeExemple, TypeSplit


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


class FauxFormateurConversation:
    """Faux adaptateur FormateurConversation : concatene prompt/completion sans tokenizer reel."""

    def __init__(self) -> None:
        self.appels = 0

    def formater(self, exemple: ExemplePivot) -> ExempleFormate:
        self.appels += 1
        texte_prompt = " ".join(m.contenu for m in exemple.prompt)
        texte_completion = " ".join(m.contenu for m in exemple.completion)
        return ExempleFormate(
            identifiant=exemple.identifiant,
            texte=f"<|im_start|>user\n{texte_prompt}<|im_end|>\n<|im_start|>assistant\n{texte_completion}<|im_end|>",
        )


def _exemple(split: TypeSplit | None) -> ExemplePivot:
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


def _exemple_dpo(split: TypeSplit | None) -> ExemplePivot:
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("UltraMedical-Preference", cle),
        identifiant_source_brute=cle,
        source="UltraMedical-Preference",
        type_exemple=TypeExemple.DPO,
        langue=Langue.ANGLAIS,
        prompt=(Message(role="user", contenu="Question ?"),),
        chosen=(Message(role="assistant", contenu="Bonne reponse."),),
        rejected=(Message(role="assistant", contenu="Mauvaise reponse."),),
        split=split,
    )


def test_formater_dataset_chatml_exclut_les_exemples_dpo():
    """
    Reproduit le trou du 12/09/2026 : `executer()` ne filtrait que sur
    `split`, sans filtrer `type_exemple`, laissant passer des
    `ExemplePivot` DPO (`completion` vide) dans le rendu ChatML SFT.
    """
    exemples_sft = [_exemple(TypeSplit.TRAIN) for _ in range(2)]
    exemples_dpo = [_exemple_dpo(TypeSplit.TRAIN) for _ in range(3)]
    repository_pivot = FauxRepository(exemples_sft + exemples_dpo)
    repository_formate = FauxRepository()
    formateur = FauxFormateurConversation()

    cas_usage = FormaterDatasetChatMLUseCase(
        repository_pivot=repository_pivot, repository_formate=repository_formate, formateur=formateur
    )
    nombre = cas_usage.executer(TypeSplit.TRAIN)

    assert nombre == 2
    assert set(repository_formate.items) == {e.identifiant for e in exemples_sft}


def test_formater_dataset_chatml_ne_traite_que_le_split_demande():
    exemples_train = [_exemple(TypeSplit.TRAIN) for _ in range(3)]
    exemples_val = [_exemple(TypeSplit.VALIDATION) for _ in range(2)]
    exemples_sans_split = [_exemple(None) for _ in range(1)]
    repository_pivot = FauxRepository(exemples_train + exemples_val + exemples_sans_split)
    repository_formate = FauxRepository()
    formateur = FauxFormateurConversation()

    cas_usage = FormaterDatasetChatMLUseCase(
        repository_pivot=repository_pivot, repository_formate=repository_formate, formateur=formateur
    )
    nombre = cas_usage.executer(TypeSplit.TRAIN)

    assert nombre == 3
    assert len(repository_formate.items) == 3
    assert set(repository_formate.items) == {e.identifiant for e in exemples_train}


def test_formater_dataset_chatml_produit_un_rendu_chatml_par_exemple():
    exemple = _exemple(TypeSplit.TEST_CLINIQUE)
    repository_pivot = FauxRepository([exemple])
    repository_formate = FauxRepository()
    formateur = FauxFormateurConversation()

    cas_usage = FormaterDatasetChatMLUseCase(
        repository_pivot=repository_pivot, repository_formate=repository_formate, formateur=formateur
    )
    cas_usage.executer(TypeSplit.TEST_CLINIQUE)

    resultat = repository_formate.items[exemple.identifiant]
    assert resultat.identifiant == exemple.identifiant
    assert "<|im_start|>assistant" in resultat.texte
    assert "Question ?" in resultat.texte
    assert "Reponse." in resultat.texte
    assert formateur.appels == 1


def test_formater_dataset_chatml_sans_exemple_dans_le_split_ne_persiste_rien():
    repository_pivot = FauxRepository([_exemple(TypeSplit.TRAIN)])
    repository_formate = FauxRepository()
    formateur = FauxFormateurConversation()

    cas_usage = FormaterDatasetChatMLUseCase(
        repository_pivot=repository_pivot, repository_formate=repository_formate, formateur=formateur
    )
    nombre = cas_usage.executer(TypeSplit.VALIDATION)

    assert nombre == 0
    assert len(repository_formate.items) == 0
    assert formateur.appels == 0
