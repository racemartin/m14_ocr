"""
Tests du masquage de perte `assistant_only_loss` de
`TrlSftEntraineurAdapter`, via la fonction pure `_tokeniser_exemple_masque`.

Aucune dependance a `torch`/`trl`/`transformers` : `_tokeniser_exemple_masque`
ne les importe pas (seuls `__post_init__`/`entrainer` le font, cf.
docstring du module), donc ce test tourne partout avec un FAUX
tokenizer (mot = un token, cf. `_FauxTokenizer`), contrairement a
`tests/infrastructure/test_trl_sft_entraineur.py` qui necessite un GPU
reel et ne s'execute jamais en Environnement A.
"""

from __future__ import annotations

from chsa_triage.domain.model.exemple_formate import ExempleFormate, LimiteTour
from chsa_triage.infrastructure.adapters.trl_sft_entraineur import (
    IGNORE_INDEX,
    _tokeniser_exemple_masque,
)


class _FauxTokenizer:
    """Tokenise par mot (split sur l'espace), un mot -> un identifiant
    entier stable (position dans un vocabulaire construit a la volee).
    Suffisant pour verifier l'alignement input_ids/labels, pas le
    comportement reel d'un tokenizer BPE."""

    def __init__(self) -> None:
        self._vocabulaire: dict[str, int] = {}

    def __call__(self, texte: str, add_special_tokens: bool = True) -> dict:
        ids = [self._id_pour(mot) for mot in texte.split()]
        return {"input_ids": ids}

    def _id_pour(self, mot: str) -> int:
        if mot not in self._vocabulaire:
            self._vocabulaire[mot] = len(self._vocabulaire)
        return self._vocabulaire[mot]


def _exemple_deux_tours() -> ExempleFormate:
    texte_user = "question utilisateur ici"
    texte_assistant = "reponse assistant la"
    texte = f"{texte_user} {texte_assistant}"
    tours = (
        LimiteTour(role="user", debut=0, fin=len(texte_user)),
        LimiteTour(
            role="assistant", debut=len(texte_user) + 1, fin=len(texte)
        ),
    )
    return ExempleFormate(identifiant="ex-1", texte=texte, tours=tours)


def test_assistant_only_loss_faux_labels_egalent_input_ids():
    exemple = _exemple_deux_tours()
    tokenizer = _FauxTokenizer()

    resultat = _tokeniser_exemple_masque(
        exemple, tokenizer, assistant_only_loss=False
    )

    assert resultat["labels"] == resultat["input_ids"]
    assert IGNORE_INDEX not in resultat["labels"]
    assert len(resultat["attention_mask"]) == len(resultat["input_ids"])
    assert all(bit == 1 for bit in resultat["attention_mask"])


def test_assistant_only_loss_vrai_masque_le_tour_non_assistant():
    exemple = _exemple_deux_tours()
    tokenizer = _FauxTokenizer()

    resultat = _tokeniser_exemple_masque(
        exemple, tokenizer, assistant_only_loss=True
    )

    nombre_mots_user = len(["question", "utilisateur", "ici"])
    nombre_mots_assistant = len(["reponse", "assistant", "la"])

    labels_user = resultat["labels"][:nombre_mots_user]
    labels_assistant = resultat["labels"][nombre_mots_user:]

    assert labels_user == [IGNORE_INDEX] * nombre_mots_user
    assert IGNORE_INDEX not in labels_assistant
    assert labels_assistant == resultat["input_ids"][nombre_mots_user:]
    assert len(resultat["input_ids"]) == nombre_mots_user + nombre_mots_assistant


def test_assistant_only_loss_vrai_sans_tours_retombe_sur_perte_pleine_sequence():
    """`tours == ()` (ExempleFormate construit sans bornes, cas legacy) :
    pas de masquage silencieusement errone, retombe explicitement sur
    `labels = input_ids`."""
    exemple = ExempleFormate(identifiant="ex-legacy", texte="un texte sans tours")
    tokenizer = _FauxTokenizer()

    resultat = _tokeniser_exemple_masque(
        exemple, tokenizer, assistant_only_loss=True
    )

    assert resultat["labels"] == resultat["input_ids"]
    assert IGNORE_INDEX not in resultat["labels"]


def test_assistant_only_loss_vrai_masque_plusieurs_tours_assistant():
    """Deux tours assistant non contigus (cas general au-dela du simple
    prompt/completion unique) : chacun doit survivre non masque dans
    `labels`, le reste masque."""
    tokenizer = _FauxTokenizer()
    texte = "sys usr asstA usr asstB"
    tours = (
        LimiteTour(role="system", debut=0, fin=3),
        LimiteTour(role="user", debut=4, fin=7),
        LimiteTour(role="assistant", debut=8, fin=13),
        LimiteTour(role="user", debut=14, fin=17),
        LimiteTour(role="assistant", debut=18, fin=23),
    )
    exemple = ExempleFormate(identifiant="ex-multi", texte=texte, tours=tours)

    resultat = _tokeniser_exemple_masque(
        exemple, tokenizer, assistant_only_loss=True
    )

    assert resultat["labels"] == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        tokenizer._vocabulaire["asstA"],
        IGNORE_INDEX,
        tokenizer._vocabulaire["asstB"],
    ]
