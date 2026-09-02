"""
Entite ExemplePivot : schema pivot commun aux quatre corpus, tel que
defini dans 01_cahier_des_charges.md paragraphe 5.2.

C'est le coeur du domaine : la seule entite qui porte du vocabulaire
medical. Aucune dependance externe (pas de pandas, pas de pydantic
ici : la validation de schema est un souci d'infrastructure/interface,
pas de domaine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from chsa_triage.domain.model.enums import (
    Langue,
    NiveauConfiance,
    TypeExemple,
    TypeSplit,
)


@dataclass(frozen=True, slots=True)
class Message:
    """Un tour de dialogue (role/contenu), independant du format ChatML."""

    role    : str   # "system" | "user" | "assistant"
    contenu : str


@dataclass(frozen=True, slots=True)
class ConstantesVitales:
    """Constantes vitales optionnelles associees a un exemple clinique."""

    pression_arterielle : str | None = None   # ex. "150/95"
    frequence_cardiaque  : int | None = None   # bpm
    saturation_o2        : int | None = None   # SpO2 %
    frequence_respiratoire: int | None = None  # rpm


@dataclass(frozen=True, slots=True)
class ExemplePivot:
    """
    Unite atomique du dataset une fois convertie au format pivot,
    quelle que soit sa source d'origine (MediQAl, FrenchMedMCQA,
    MedQuAD, UltraMedical-Preference).

    - Pour un exemple SFT : `prompt` + `completion` sont renseignes.
    - Pour un exemple DPO : `prompt` + `chosen` + `rejected` sont
      renseignes, `completion` reste vide.
    """

    identifiant          : str
    source                : str                 # nom du CorpusSource d'origine
    type_exemple          : TypeExemple
    langue                : Langue

    symptomes             : str = ""
    antecedents           : str | None = None
    constantes_vitales    : ConstantesVitales | None = None

    prompt                : tuple[Message, ...] = field(default_factory=tuple)
    completion             : tuple[Message, ...] = field(default_factory=tuple)
    chosen                 : tuple[Message, ...] = field(default_factory=tuple)
    rejected                : tuple[Message, ...] = field(default_factory=tuple)

    niveau_confiance       : NiveauConfiance = NiveauConfiance.MOYENNE
    anonymise               : bool = False
    split                   : TypeSplit | None = None

    @staticmethod
    def nouvel_identifiant(source: str) -> str:
        """Genere un identifiant conforme au schema `chsa-<source>-<uuid>`."""
        return f"chsa-{source.lower()}-{uuid4().hex[:12]}"

    def est_complet_pour_sft(self) -> bool:
        """Verifie qu'un exemple SFT a bien un prompt et une completion."""
        return self.type_exemple == TypeExemple.SFT and bool(self.prompt) and bool(self.completion)

    def est_complet_pour_dpo(self) -> bool:
        """Verifie qu'un exemple DPO a bien un prompt, un chosen et un rejected."""
        return (
            self.type_exemple == TypeExemple.DPO
            and bool(self.prompt)
            and bool(self.chosen)
            and bool(self.rejected)
        )
