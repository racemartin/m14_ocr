"""
Cas d'usage : anonymiser les champs texte libre (symptomes,
antecedents, messages) d'un ensemble d'ExemplePivot deja persiste,
et re-sauvegarder les versions anonymisees.
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
    """
    Compteurs RGPD accumules pendant une passe d'anonymisation, par
    source (cf. section 3 -- resultats quantitatifs -- du rapport de
    justification RGPD). Auparavant, `ResultatAnonymisation` etait
    calcule puis jete a chaque champ anonymise : ces compteurs sont
    l'instrumentation minimale necessaire pour produire des chiffres
    reels (et non estimes) sur les entites detectees.
    """

    registres_traites      : int = 0
    registres_avec_entite  : int = 0
    entites_par_type       : dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class AnonymiserDatasetUseCase:
    """Orchestre l'anonymisation RGPD d'un dataset pivot.

    Incremental et reprenable : ne touche que les exemples encore
    marques `anonymise=False`. Un exemple deja anonymise (`anonymise=True`)
    n'est jamais retraite, meme si `executer()` est rappele plus tard --
    c'est ce qui permet d'anonymiser en plusieurs passes successives
    (`--limite`) sans jamais refaire le travail deja fait.
    """

    repository          : RepositoryLectureEcriture
    anonymiseur         : Anonymiseur
    limite              : int | None = None
    graine_aleatoire    : int = 42
    statistiques        : dict[str, StatistiquesSource] = field(default_factory=dict, init=False)

    def executer(
        self,
        envelopper_iterable: Callable[[Iterable[ExemplePivot]], Iterable[ExemplePivot]] | None = None,
    ) -> int:
        """
        Parcourt les ExemplePivot non encore anonymises (au plus
        `self.limite`, tire en echantillon stratifie par
        (type_exemple, source) si le nombre en attente depasse la
        limite), masque les entites sensibles dans les champs texte
        libre, et persiste les versions anonymisees en une seule
        operation. Les exemples non selectionnes restent
        `anonymise=False`, prets pour un appel ulterieur avec une
        limite plus grande (ou `limite=None` pour tout traiter).

        `envelopper_iterable` permet a l'appelant (typiquement l'interface
        CLI) de brancher une barre de progression sans que ce cas d'usage
        depende d'une librairie de presentation : par defaut, l'iterable
        n'est pas modifie.

        Retourne le nombre d'exemples effectivement traites.

        NOTE (07/09/2026, decouvert en executant le pipeline sur le
        dataset pivot reel, 147204 exemples/624 Mo) : appeler
        `self.repository.sauvegarder(...)` a chaque iteration relit et
        reecrit tout le fichier JSONL a CHAQUE exemple (cf.
        `JsonlDatasetRepository.sauvegarder`) -- sur 147204 exemples
        c'est un O(n^2) totalement infaisable (des heures, voire des
        jours). `sauvegarder_plusieurs` fait le meme travail de
        fusion par identifiant mais en une seule lecture/ecriture du
        fichier, quel que soit le nombre d'exemples traites.

        NOTE (08/09/2026, decision produit) : mesure reelle sur le
        dataset pivot complet (147204 exemples) -- l'anonymisation
        Presidio/spaCy complete prendrait ~19h (cout NLP, pas I/O).
        Decision du capitaine : ne pas trancher entre "echantillon" et
        "complet", mais rendre le processus incremental via le champ
        `anonymise` deja present sur `ExemplePivot`. `--limite`
        (cf. `interfaces/cli/anonymiser_dataset.py`) permet de traiter
        le dataset par vagues successives, chacune stratifiee pour
        rester representative de toutes les (type_exemple, source).
        """
        candidats = list(self.repository.lister(filtre={"anonymise": False}))

        if self.limite is None or self.limite >= len(candidats):
            a_traiter = candidats
        else:
            a_traiter = self._echantillon_stratifie(candidats, self.limite)

        iterable = envelopper_iterable(a_traiter) if envelopper_iterable else a_traiter

        exemples_anonymises = [self._anonymiser_exemple(exemple) for exemple in iterable]
        self.repository.sauvegarder_plusieurs(exemples_anonymises)
        return len(exemples_anonymises)

    def _echantillon_stratifie(self, candidats: list[ExemplePivot], taille: int) -> list[ExemplePivot]:
        return echantillon_stratifie(candidats, taille, self.graine_aleatoire)

    def _anonymiser_exemple(self, exemple: ExemplePivot) -> ExemplePivot:
        """Applique l'anonymisation a tous les champs texte libre."""
        langue = exemple.langue.value
        resultats_champ: list[ResultatAnonymisation] = []

        resultat_symptomes = self.anonymiseur.anonymiser(exemple.symptomes, langue)
        resultats_champ.append(resultat_symptomes)
        symptomes_anon = resultat_symptomes.texte_anonymise

        antecedents_anon = None
        if exemple.antecedents:
            resultat_antecedents = self.anonymiseur.anonymiser(exemple.antecedents, langue)
            resultats_champ.append(resultat_antecedents)
            antecedents_anon = resultat_antecedents.texte_anonymise

        prompt_anon, resultats_prompt         = self._anonymiser_messages(exemple.prompt, langue)
        completion_anon, resultats_completion = self._anonymiser_messages(exemple.completion, langue)
        chosen_anon, resultats_chosen         = self._anonymiser_messages(exemple.chosen, langue)
        rejected_anon, resultats_rejected     = self._anonymiser_messages(exemple.rejected, langue)
        resultats_champ.extend(resultats_prompt + resultats_completion + resultats_chosen + resultats_rejected)

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
        resultats = [self.anonymiseur.anonymiser(m.contenu, langue) for m in messages]
        nouveaux_messages = tuple(
            replace(m, contenu=resultat.texte_anonymise) for m, resultat in zip(messages, resultats)
        )
        return nouveaux_messages, resultats

    def _enregistrer_statistiques(self, source: str, resultats: list[ResultatAnonymisation]) -> None:
        """Accumule, pour `source`, les compteurs RGPD du rapport de justification (section 3)."""
        stats = self.statistiques.setdefault(source, StatistiquesSource())
        stats.registres_traites += 1

        au_moins_une_entite = False
        for resultat in resultats:
            if resultat.entites_detectees:
                au_moins_une_entite = True
            for entite in resultat.entites_detectees:
                stats.entites_par_type[entite.type_entite] = stats.entites_par_type.get(entite.type_entite, 0) + 1

        if au_moins_une_entite:
            stats.registres_avec_entite += 1
