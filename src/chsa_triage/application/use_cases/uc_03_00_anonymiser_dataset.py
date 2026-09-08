"""
Cas d'usage : anonymiser les champs texte libre (symptomes,
antecedents, messages) d'un ensemble d'ExemplePivot, en LISANT le
pivot original (jamais modifie) et en ECRIVANT le resultat dans un
fichier de SORTIE separe.

Design (08/09/2026, decision du capitaine -- remplace un design
precedent qui mutait le pivot en place) : le pivot original reste
intact pour toujours, ce qui permet (a) de regenerer/reutiliser le
pivot source sans jamais perdre le texte original d'un exemple deja
anonymise, et (b) un controle qualite a posteriori par simple
comparaison de deux fichiers (cf. `controler_qualite_anonymisation.py`),
plutot qu'un enganche en direct dans la boucle d'anonymisation.
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

    `repository_source` est LU SEUL, jamais ecrit -- c'est le pivot
    original, immuable. `repository_sortie` est le fichier separe ou
    sont persistes les exemples anonymises (`anonymise=True`).

    "Deja traite" se determine desormais par la presence de
    l'`identifiant` dans `repository_sortie` (pas par un booleen mute
    sur l'exemple source) : c'est ce qui permet de traiter le dataset
    en plusieurs passes successives (`--limite`) sans jamais retraiter
    un exemple deja anonymise, tout en laissant le pivot source
    intact.
    """

    repository_source    : RepositoryLectureEcriture
    repository_sortie     : RepositoryLectureEcriture
    anonymiseur            : Anonymiseur
    limite                  : int | None = None
    graine_aleatoire         : int = 42
    statistiques             : dict[str, StatistiquesSource] = field(default_factory=dict, init=False)

    def executer(
        self,
        envelopper_iterable: Callable[[Iterable[ExemplePivot]], Iterable[ExemplePivot]] | None = None,
    ) -> int:
        """
        Parcourt les ExemplePivot du pivot source dont l'identifiant
        n'est PAS encore present dans le fichier de sortie (au plus
        `self.limite`, tire en echantillon stratifie par
        (type_exemple, source) si le nombre en attente depasse la
        limite), masque les entites sensibles dans les champs texte
        libre, et persiste les versions anonymisees dans le fichier de
        SORTIE en une seule operation. Le pivot source n'est jamais
        modifie. Les exemples non selectionnes ne sont pas encore dans
        le fichier de sortie, prets pour un appel ulterieur avec une
        limite plus grande (ou `limite=None` pour tout traiter).

        `envelopper_iterable` permet a l'appelant (typiquement l'interface
        CLI) de brancher une barre de progression sans que ce cas d'usage
        depende d'une librairie de presentation : par defaut, l'iterable
        n'est pas modifie.

        Retourne le nombre d'exemples effectivement traites lors de
        CETTE execution.

        NOTE (07/09/2026, decouvert en executant le pipeline sur le
        dataset pivot reel, 147204 exemples/624 Mo) : appeler
        `self.repository_sortie.sauvegarder(...)` a chaque iteration
        relit et reecrit tout le fichier JSONL a CHAQUE exemple (cf.
        `JsonlDatasetRepository.sauvegarder`) -- sur un dataset de
        cette taille c'est un O(n^2) totalement infaisable (des
        heures, voire des jours). `sauvegarder_plusieurs` fait le meme
        travail de fusion par identifiant mais en une seule
        lecture/ecriture du fichier, quel que soit le nombre
        d'exemples traites.

        NOTE (08/09/2026, decision produit) : mesure reelle sur le
        dataset pivot complet (147204 exemples) -- l'anonymisation
        Presidio/spaCy complete prendrait ~19h (cout NLP, pas I/O).
        Decision du capitaine : ne pas trancher entre "echantillon" et
        "complet", mais rendre le processus incremental. `--limite`
        (cf. `interfaces/cli/anonymiser_dataset.py`) permet de traiter
        le dataset par vagues successives, chacune stratifiee pour
        rester representative de toutes les (type_exemple, source).
        """
        deja_traites = self.repository_sortie.identifiants_existants()
        candidats = [e for e in self.repository_source.lister() if e.identifiant not in deja_traites]

        if self.limite is None or self.limite >= len(candidats):
            a_traiter = candidats
        else:
            a_traiter = self._echantillon_stratifie(candidats, self.limite)

        iterable = envelopper_iterable(a_traiter) if envelopper_iterable else a_traiter

        exemples_anonymises = [self._anonymiser_exemple(exemple) for exemple in iterable]
        self.repository_sortie.sauvegarder_plusieurs(exemples_anonymises)
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
