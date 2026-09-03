"""
Cas d'usage : anonymiser les champs texte libre (symptomes,
antecedents, messages) d'un ensemble d'ExemplePivot deja persiste,
et re-sauvegarder les versions anonymisees.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from chsa_triage.domain.model import ExemplePivot, Message
from chsa_triage.domain.ports import Anonymiseur, RepositoryLectureEcriture


@dataclass(slots=True)
class AnonymiserDatasetUseCase:
    """Orchestre l'anonymisation RGPD d'un dataset pivot."""

    repository  : RepositoryLectureEcriture
    anonymiseur : Anonymiseur

    def executer(self) -> int:
        """
        Parcourt tous les ExemplePivot non encore anonymises, masque
        les entites sensibles dans les champs texte libre, et
        persiste la version anonymisee (anonymise=True).

        Retourne le nombre d'exemples traites.
        """
        compteur = 0

        for exemple in self.repository.lister(filtre={"anonymise": False}):
            exemple_anonymise = self._anonymiser_exemple(exemple)
            self.repository.sauvegarder(exemple_anonymise)
            compteur += 1

        return compteur

    def _anonymiser_exemple(self, exemple: ExemplePivot) -> ExemplePivot:
        """Applique l'anonymisation a tous les champs texte libre."""
        langue = exemple.langue.value

        symptomes_anon = self.anonymiseur.anonymiser(exemple.symptomes, langue).texte_anonymise

        antecedents_anon = None
        if exemple.antecedents:
            antecedents_anon = self.anonymiseur.anonymiser(exemple.antecedents, langue).texte_anonymise

        messages_anonymises = tuple(
            self._anonymiser_messages(groupe, langue)
            for groupe in (exemple.prompt, exemple.completion, exemple.chosen, exemple.rejected)
        )

        return replace(
            exemple,
            symptomes=symptomes_anon,
            antecedents=antecedents_anon,
            prompt=messages_anonymises[0],
            completion=messages_anonymises[1],
            chosen=messages_anonymises[2],
            rejected=messages_anonymises[3],
            anonymise=True,
        )

    def _anonymiser_messages(self, messages: tuple[Message, ...], langue: str) -> tuple[Message, ...]:
        return tuple(
            replace(m, contenu=self.anonymiseur.anonymiser(m.contenu, langue).texte_anonymise)
            for m in messages
        )
