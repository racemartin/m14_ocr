"""
Cas d'usage : profiler un corpus brut avant de decider s'il faut le
nettoyer.

Ne connait que des ports (LecteurCorpus, Profileur), jamais une
implementation concrete. L'injection des adaptateurs se fait a la
construction (voir interfaces/cli).
"""

from __future__ import annotations

from dataclasses import dataclass

from chsa_triage.domain.ports import LecteurCorpus, Profileur, RapportProfilage


@dataclass(slots=True)
class ProfilerCorpusUseCase:
    """Orchestre la lecture d'un corpus et son profilage."""

    lecteur  : LecteurCorpus
    profileur : Profileur

    def executer(self, nom_corpus: str) -> RapportProfilage:
        """
        Lit l'integralite du corpus via le port `LecteurCorpus`, puis
        produit un rapport de profilage via le port `Profileur`.
        """
        enregistrements = list(self.lecteur.lire_enregistrements())
        return self.profileur.profiler(enregistrements, nom_corpus)
