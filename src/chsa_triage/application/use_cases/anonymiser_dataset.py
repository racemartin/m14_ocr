"""
Cas d'usage : anonymiser les champs texte libre (symptomes,
antecedents, messages) d'un ensemble d'ExemplePivot deja persiste,
et re-sauvegarder les versions anonymisees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from chsa_triage.domain.model import ExemplePivot, Message
from chsa_triage.domain.ports import Anonymiseur, RepositoryLectureEcriture


@dataclass(slots=True)
class AnonymiserDatasetUseCase:
    """Orchestre l'anonymisation RGPD d'un dataset pivot.

    Incremental et reprenable : ne touche que les exemples encore
    marques `anonymise=False`. Un exemple deja anonymise (`anonymise=True`)
    n'est jamais retraite, meme si `executer()` est rappele plus tard --
    c'est ce qui permet d'anonymiser en plusieurs passes successives
    (`--limite`) sans jamais refaire le travail deja fait.
    """

    repository        : RepositoryLectureEcriture
    anonymiseur        : Anonymiseur
    limite              : int | None = None
    graine_aleatoire    : int = 42

    def executer(self) -> int:
        """
        Parcourt les ExemplePivot non encore anonymises (au plus
        `self.limite`, tire en echantillon stratifie par
        (type_exemple, source) si le nombre en attente depasse la
        limite), masque les entites sensibles dans les champs texte
        libre, et persiste les versions anonymisees en une seule
        operation. Les exemples non selectionnes restent
        `anonymise=False`, prets pour un appel ulterieur avec une
        limite plus grande (ou `limite=None` pour tout traiter).

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

        exemples_anonymises = [self._anonymiser_exemple(exemple) for exemple in a_traiter]
        self.repository.sauvegarder_plusieurs(exemples_anonymises)
        return len(exemples_anonymises)

    def _echantillon_stratifie(self, candidats: list[ExemplePivot], taille: int) -> list[ExemplePivot]:
        """
        Selectionne `taille` exemples parmi `candidats`, en respectant
        au mieux la proportion de chaque strate (type_exemple, source)
        dans l'echantillon (methode du plus grand reste, tirage
        aleatoire reproductible via `graine_aleatoire` au sein de
        chaque strate).
        """
        rng = random.Random(self.graine_aleatoire)

        groupes: dict[tuple[str, str], list[ExemplePivot]] = {}
        for exemple in candidats:
            cle = (exemple.type_exemple.value, exemple.source)
            groupes.setdefault(cle, []).append(exemple)

        total = len(candidats)
        quotas: dict[tuple[str, str], int] = {}
        restes: list[tuple[float, tuple[str, str]]] = []
        for cle, groupe in groupes.items():
            part_exacte = taille * (len(groupe) / total)
            quotas[cle] = min(int(part_exacte), len(groupe))
            restes.append((part_exacte - int(part_exacte), cle))

        # Methode du plus grand reste : distribue les unites manquantes
        # (arrondis vers le bas ci-dessus) aux strates dont le reste
        # fractionnaire est le plus grand, dans la limite de leur taille.
        deficit = taille - sum(quotas.values())
        for _, cle in sorted(restes, key=lambda r: r[0], reverse=True):
            if deficit <= 0:
                break
            if quotas[cle] < len(groupes[cle]):
                quotas[cle] += 1
                deficit -= 1

        selection: list[ExemplePivot] = []
        for cle in sorted(groupes):
            groupe = list(groupes[cle])
            rng.shuffle(groupe)
            selection.extend(groupe[: quotas[cle]])

        return selection

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
