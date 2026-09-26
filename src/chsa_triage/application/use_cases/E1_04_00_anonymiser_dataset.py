"""
Cas d'usage : anonymiser les champs texte libre (symptomes,
antecedents, messages) d'un ensemble d'ExemplePivot, en LISANT le
pivot original (jamais modifie) et en ECRIVANT le resultat dans un
fichier de SORTIE separe. Le pivot original reste intact : on peut le
regenerer sans perdre le texte original, et comparer les deux fichiers
pour le controle qualite (`E1_04_02_controler_qualite_anonymisation.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from chsa_triage.application.echantillonnage import echantillon_stratifie
from chsa_triage.domain.model import ExemplePivot, Message
from chsa_triage.domain.ports import Anonymiseur, RepositoryLectureEcriture
from chsa_triage.domain.ports.anonymiseur import ResultatAnonymisation


@dataclass(slots=True)
class StatistiquesSource:
    """Compteurs RGPD accumules pendant une passe d'anonymisation, par
    source (cf. rapport de justification RGPD, section 3) : chiffres
    reels sur les entites detectees, jamais estimes."""

    registres_traites: int = 0
    registres_avec_entite: int = 0
    entites_par_type: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class AnonymiserDatasetUseCase:
    """Orchestre l'anonymisation RGPD d'un dataset pivot.

    `repository_source` est LU SEUL, jamais ecrit. "Deja traite" se
    determine par la presence de l'`identifiant` dans
    `repository_sortie` (pas un booleen sur l'exemple source), ce qui
    permet de traiter le dataset en plusieurs passes (`--limite`) sans
    jamais retraiter un exemple deja anonymise.
    """

    repository_source: RepositoryLectureEcriture
    repository_sortie: RepositoryLectureEcriture
    anonymiseur: Anonymiseur
    limite: int | None = None
    graine_aleatoire: int = 42
    statistiques: dict[str, StatistiquesSource] = field(
        default_factory=dict, init=False
    )

    def executer(
        self,
        envelopper_iterable: Callable[
            [Iterable[ExemplePivot]], Iterable[ExemplePivot]
        ]
        | None = None,
    ) -> int:
        """
        Traite les ExemplePivot pas encore dans le fichier de sortie
        (au plus `self.limite`, en echantillon stratifie si depasse),
        masque les entites sensibles et persiste le resultat en UNE
        seule ecriture (`sauvegarder_plusieurs`). Retourne le nombre
        d'exemples traites lors de CETTE execution.

        `envelopper_iterable` permet a l'appelant de brancher une barre
        de progression sans dependance de presentation ici.

        Piege reel rencontre sur le dataset complet (147204 exemples) :
        appeler `sauvegarder(...)` a chaque iteration relit/reecrit tout
        le JSONL a chaque exemple (O(n^2), infaisable) — d'ou
        `sauvegarder_plusieurs` en une seule passe. L'anonymisation
        complete prend par ailleurs ~19h (cout NLP) : `--limite` existe
        pour traiter le dataset par vagues successives incrementales.
        """
        deja_traites = self.repository_sortie.identifiants_existants()
        candidats = [
            e
            for e in self.repository_source.lister()
            if e.identifiant not in deja_traites
        ]

        if self.limite is None or self.limite >= len(candidats):
            a_traiter = candidats
        else:
            a_traiter = self._echantillon_stratifie(candidats, self.limite)

        iterable = (
            envelopper_iterable(a_traiter) if envelopper_iterable else a_traiter
        )

        exemples_anonymises = [
            self._anonymiser_exemple(exemple) for exemple in iterable
        ]
        self.repository_sortie.sauvegarder_plusieurs(exemples_anonymises)
        return len(exemples_anonymises)

    def _echantillon_stratifie(
        self, candidats: list[ExemplePivot], taille: int
    ) -> list[ExemplePivot]:
        return echantillon_stratifie(candidats, taille, self.graine_aleatoire)

    def _anonymiser_exemple(self, exemple: ExemplePivot) -> ExemplePivot:
        """Applique l'anonymisation a tous les champs texte libre."""
        langue = exemple.langue.value
        resultats_champ: list[ResultatAnonymisation] = []

        resultat_symptomes = self.anonymiseur.anonymiser(
            exemple.symptomes, langue
        )
        resultats_champ.append(resultat_symptomes)
        symptomes_anon = resultat_symptomes.texte_anonymise

        antecedents_anon = None
        if exemple.antecedents:
            resultat_antecedents = self.anonymiseur.anonymiser(
                exemple.antecedents, langue
            )
            resultats_champ.append(resultat_antecedents)
            antecedents_anon = resultat_antecedents.texte_anonymise

        prompt_anon, resultats_prompt = self._anonymiser_messages(
            exemple.prompt, langue
        )
        completion_anon, resultats_completion = self._anonymiser_messages(
            exemple.completion, langue
        )
        chosen_anon, resultats_chosen = self._anonymiser_messages(
            exemple.chosen, langue
        )
        rejected_anon, resultats_rejected = self._anonymiser_messages(
            exemple.rejected, langue
        )
        resultats_champ.extend(
            resultats_prompt
            + resultats_completion
            + resultats_chosen
            + resultats_rejected
        )

        self._enregistrer_statistiques(exemple.source, resultats_champ)

        return replace(
            exemple,
            symptomes=symptomes_anon,
            antecedents=antecedents_anon,
            prompt=prompt_anon,
            completion=completion_anon,
            chosen=chosen_anon,
            rejected=rejected_anon,
            anonymise=True,
        )

    def _anonymiser_messages(
        self, messages: tuple[Message, ...], langue: str
    ) -> tuple[tuple[Message, ...], list[ResultatAnonymisation]]:
        resultats = [
            self.anonymiseur.anonymiser(m.contenu, langue) for m in messages
        ]
        nouveaux_messages = tuple(
            replace(m, contenu=resultat.texte_anonymise)
            for m, resultat in zip(messages, resultats)
        )
        return nouveaux_messages, resultats

    def _enregistrer_statistiques(
        self, source: str, resultats: list[ResultatAnonymisation]
    ) -> None:
        """Accumule, pour `source`, les compteurs RGPD du rapport de justification (section 3)."""
        stats = self.statistiques.setdefault(source, StatistiquesSource())
        stats.registres_traites += 1

        au_moins_une_entite = False
        for resultat in resultats:
            if resultat.entites_detectees:
                au_moins_une_entite = True
            for entite in resultat.entites_detectees:
                stats.entites_par_type[entite.type_entite] = (
                    stats.entites_par_type.get(entite.type_entite, 0) + 1
                )

        if au_moins_une_entite:
            stats.registres_avec_entite += 1
