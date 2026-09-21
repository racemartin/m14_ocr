"""
Tests de `ReformulerPreferenceDpoUseCase`, avec un faux `MoteurInference`
en memoire (scripte pour retourner tantot un JSON valide, tantot un
format degenere) et un faux `RepositoryLectureEcriture` en memoire, meme
patron que `tests/application/test_E2_01_uc_entrainer_sft.py`. Aucun
GPU necessaire : seule la logique d'orchestration/exclusion/comptage est
validee ici.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from uuid import uuid4

from chsa_triage.application.use_cases.E3_00_uc_reformuler_preference_dpo import (
    LONGITUD_MAX_ENTREE_REFORMULATION,
    MIN_TOKENS_GENERES_REFORMULATION,
    NOMBRE_TOKENS_GENERES_REFORMULATION,
    PROMPT_REFORMULATION_CHOSEN,
    TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION,
    TEMPERATURE_REFORMULATION,
    EchecReformulation,
    ReformulerPreferenceDpoUseCase,
)
from chsa_triage.domain.model.enums import Langue, TypeExemple
from chsa_triage.domain.model.exemple_pivot import ExemplePivot, Message
from chsa_triage.domain.model.preference_reformulee import ChosenReformule
from chsa_triage.domain.ports.moteur_inference import ReponseModele

JSON_CIBLE = json.dumps({"niveau": 3, "categorie": "respiratoire", "ressources_estimees": "oxygenotherapie"})
TEXTE_REFORMULATION_VALIDE = f"<think>raisonnement clinique</think>{JSON_CIBLE}"


class FauxMoteurInference:
    """
    Faux adaptateur MoteurInference : retourne une reponse scriptee par
    identifiant, sans reseau/GPU. `reponses_speciales` est cle par une
    sous-chaine attendue dans le seul message `user` (prompt + texte a
    reformuler concatenes, cf. E3_00_uc_reformuler_preference_dpo.py),
    jamais une egalite exacte, puisque ce message contient desormais
    PROMPT_REFORMULATION_CHOSEN en plus du texte source.
    """

    def __init__(self, reponses_par_defaut: str = TEXTE_REFORMULATION_VALIDE) -> None:
        self.reponses_par_defaut = reponses_par_defaut
        self.reponses_speciales: dict[str, str | Exception | ReponseModele] = {}
        self.appels: list[list[dict]] = []
        self.appels_parametres: list[dict | None] = []

    def generer(self, messages: list[dict], parametres: dict | None = None) -> ReponseModele:
        self.appels.append(messages)
        self.appels_parametres.append(parametres)
        contenu_utilisateur = messages[-1]["content"]
        reponse = self.reponses_par_defaut
        for cle, valeur in self.reponses_speciales.items():
            if cle in contenu_utilisateur:
                reponse = valeur
                break
        if isinstance(reponse, Exception):
            raise reponse
        if isinstance(reponse, ReponseModele):
            return reponse
        return ReponseModele(texte=reponse)


class FauxRepositoryReformule:
    """Faux adaptateur RepositoryLectureEcriture, en memoire, parametre sur ChosenReformule."""

    def __init__(self, items: list[ChosenReformule] | None = None) -> None:
        self.items: dict[str, ChosenReformule] = {item.identifiant: item for item in (items or [])}

    def sauvegarder(self, item: ChosenReformule) -> None:
        self.items[item.identifiant] = item

    def sauvegarder_plusieurs(self, items: Iterable[ChosenReformule]) -> None:
        for item in items:
            self.items[item.identifiant] = item

    def trouver_par_id(self, identifiant: str):
        return self.items.get(identifiant)

    def lister(self, filtre: dict | None = None):
        yield from self.items.values()

    def compter(self, filtre: dict | None = None) -> int:
        return len(self.items)

    def identifiants_existants(self) -> set[str]:
        return set(self.items.keys())


def _exemple_dpo(texte_chosen: str = "reponse choisie originale") -> ExemplePivot:
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("UltraMedical-Preference", cle),
        identifiant_source_brute=cle,
        source="UltraMedical-Preference",
        type_exemple=TypeExemple.DPO,
        langue=Langue.ANGLAIS,
        prompt=(Message(role="user", contenu="Question clinique ?"),),
        chosen=(Message(role="assistant", contenu=texte_chosen),),
        rejected=(Message(role="assistant", contenu="mauvaise reponse"),),
    )


def _exemple_sft() -> ExemplePivot:
    cle = uuid4().hex
    return ExemplePivot(
        identifiant=ExemplePivot.nouvel_identifiant("MediQAl", cle),
        identifiant_source_brute=cle,
        source="MediQAl",
        type_exemple=TypeExemple.SFT,
        langue=Langue.FRANCAIS,
        prompt=(Message(role="user", contenu="Question ?"),),
        completion=(Message(role="assistant", contenu="Reponse."),),
    )


def test_reformule_uniquement_les_exemples_dpo():
    exemples = [_exemple_dpo(), _exemple_dpo(), _exemple_sft()]
    moteur = FauxMoteurInference()
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    nombre = cas_usage.executer(exemples)

    assert nombre == 2
    assert cas_usage.nombre_echecs_reformulation == 0
    assert len(moteur.appels) == 2


def test_exclut_les_identifiants_deja_reformules():
    deja_reformule = ChosenReformule(
        identifiant="chsa-deja-reformule",
        chosen_reformule=(Message(role="assistant", contenu=TEXTE_REFORMULATION_VALIDE),),
        horodatage="2026-09-19T00:00:00Z",
    )
    exemple_existant = replace(_exemple_dpo(), identifiant="chsa-deja-reformule")
    nouvel_exemple = _exemple_dpo()

    moteur = FauxMoteurInference()
    repository = FauxRepositoryReformule([deja_reformule])
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository, taille_cible=5000)

    nombre = cas_usage.executer([exemple_existant, nouvel_exemple])

    assert nombre == 1
    assert set(repository.items) == {"chsa-deja-reformule", nouvel_exemple.identifiant}
    assert len(moteur.appels) == 1


def test_taille_cible_est_cumulative_pas_un_plafond_par_execution():
    """
    `taille_cible` est une cible CUMULATIVE (comme `E1_05_00_decouper_splits.py
    --n`) : si 2 exemples sont deja reformules et `taille_cible=3`, un
    seul nouvel exemple doit etre traite, meme si plus de candidats sont
    disponibles.
    """
    deja_reformules = [
        ChosenReformule(
            identifiant=f"chsa-deja-{i}",
            chosen_reformule=(Message(role="assistant", contenu=TEXTE_REFORMULATION_VALIDE),),
            horodatage="2026-09-19T00:00:00Z",
        )
        for i in range(2)
    ]
    candidats = [_exemple_dpo() for _ in range(5)]

    moteur = FauxMoteurInference()
    repository = FauxRepositoryReformule(deja_reformules)
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository, taille_cible=3)

    nombre = cas_usage.executer(candidats)

    assert nombre == 1
    assert len(moteur.appels) == 1


def test_echec_de_parsing_est_compte_et_exclut_du_resultat_sans_ecrire_un_exemple_partiel():
    exemple_ok = _exemple_dpo(texte_chosen="reponse ok")
    exemple_degenere = _exemple_dpo(texte_chosen="reponse degeneree")

    moteur = FauxMoteurInference()
    moteur.reponses_speciales["reponse degeneree"] = "texte libre, pas de <think> ni de JSON"
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    nombre = cas_usage.executer([exemple_ok, exemple_degenere])

    assert nombre == 1
    assert cas_usage.nombre_echecs_reformulation == 1
    assert exemple_ok.identifiant in repository.items
    assert exemple_degenere.identifiant not in repository.items


def test_echec_de_parsing_capture_entree_et_sortie_emparieees_dans_l_echantillon():
    """
    Entree ET sortie, emparieees (jamais deux listes paralleles qui
    pourraient se desaligner, cf. AGENTS.md) : sans l'entree, un echec
    (vide, illisible, etc.) ne peut pas etre relie au texte source qui
    l'a produit.
    """
    exemple_degenere = _exemple_dpo(texte_chosen="reponse degeneree")
    texte_brut = "preambule inattendu, pas de <think> ni de JSON valide"

    moteur = FauxMoteurInference()
    moteur.reponses_speciales["reponse degeneree"] = texte_brut
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer([exemple_degenere])

    assert cas_usage.echantillon_echecs_reformulation == [
        EchecReformulation(entree="reponse degeneree", sortie_brute=texte_brut)
    ]


def test_echec_d_inference_ne_capture_rien_dans_l_echantillon():
    """Une exception d'inference n'a pas de texte a montrer (cf. FauxMoteurInference.generer)."""
    exemple_en_echec = _exemple_dpo(texte_chosen="reponse en echec")

    moteur = FauxMoteurInference()
    moteur.reponses_speciales["reponse en echec"] = RuntimeError("500 Internal Server Error")
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer([exemple_en_echec])

    assert cas_usage.echantillon_echecs_reformulation == []


def test_echantillon_echecs_reformulation_est_acote_meme_avec_plus_d_echecs():
    """
    Jamais de croissance sans limite en memoire (cf. AGENTS.md, premier
    job DPO reel : 90/90 echecs de format) : au-dela de
    `TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION`, les echecs
    supplementaires sont toujours comptes mais plus captures.
    """
    nombre_candidats = TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION + 3
    exemples = [_exemple_dpo(texte_chosen=f"reponse degeneree {i}") for i in range(nombre_candidats)]

    moteur = FauxMoteurInference()
    for i in range(nombre_candidats):
        moteur.reponses_speciales[f"reponse degeneree {i}"] = f"texte libre {i}, pas de format cible"
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer(exemples)

    assert cas_usage.nombre_echecs_reformulation == nombre_candidats
    assert len(cas_usage.echantillon_echecs_reformulation) == TAILLE_MAX_ECHANTILLON_ECHECS_REFORMULATION


def test_echec_d_inference_est_compte_sans_abandonner_le_lot():
    exemple_ok = _exemple_dpo(texte_chosen="reponse ok")
    exemple_en_echec = _exemple_dpo(texte_chosen="reponse en echec")

    moteur = FauxMoteurInference()
    moteur.reponses_speciales["reponse en echec"] = RuntimeError("500 Internal Server Error")
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    nombre = cas_usage.executer([exemple_ok, exemple_en_echec])

    assert nombre == 1
    assert cas_usage.nombre_echecs_reformulation == 1
    assert exemple_en_echec.identifiant not in repository.items


def test_le_prompt_est_un_unique_tour_user_jamais_system():
    """
    Cf. AGENTS.md/docstring du module : le checkpoint SFT-LoRA n'a jamais
    vu de tour `system` pendant son propre entrainement (confirme sur
    dataset_pivot_anonymise.jsonl, 134883 exemples reels, aucun role
    `system`) ; un tour `system` separe est une hypothese bien fondee
    pour l'echec du premier job DPO reel (0/90, texte genere VIDE).
    """
    exemple = _exemple_dpo(texte_chosen="reponse choisie originale")
    moteur = FauxMoteurInference()
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer([exemple])

    (messages,) = moteur.appels
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert PROMPT_REFORMULATION_CHOSEN in messages[0]["content"]
    assert "reponse choisie originale" in messages[0]["content"]


def test_le_prompt_inclut_un_exemple_few_shot_deja_resolu():
    """
    Cf. AGENTS.md/docstring du module : le 2e job DPO reel (0/90, texte
    illisible plutot que vide) a montre qu'une description en prose du
    format seule ne suffit pas pour un petit modele non-instruct ; le
    prompt doit inclure un exemple deja resolu (few-shot) respectant
    lui-meme le format exact attendu par `parser_reformulation_stricte()`.
    """
    assert "deja resolu" in PROMPT_REFORMULATION_CHOSEN
    assert "invente" in PROMPT_REFORMULATION_CHOSEN

    marqueur = "Reponse attendue :\n"
    debut_exemple = PROMPT_REFORMULATION_CHOSEN.index(marqueur) + len(marqueur)
    bloc_exemple = PROMPT_REFORMULATION_CHOSEN[debut_exemple:]
    assert bloc_exemple.startswith("<think>")

    debut_json = bloc_exemple.index("</think>") + len("</think>")
    fin_json = bloc_exemple.index("\n\n", debut_json)
    objet_exemple = json.loads(bloc_exemple[debut_json:fin_json])
    assert set(objet_exemple.keys()) == {"niveau", "categorie", "ressources_estimees"}


def test_generer_recoit_des_parametres_de_generation_explicites():
    """
    Cf. AGENTS.md/docstring du module : ne plus laisser l'adaptateur sur
    ses defauts implicites (potentiellement du decodage glouton, cause
    plausible de la degenerescence observee sur les deux premiers jobs
    DPO reels).
    """
    exemple = _exemple_dpo()
    moteur = FauxMoteurInference()
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer([exemple])

    (parametres,) = moteur.appels_parametres
    assert parametres == {
        "n_predict": NOMBRE_TOKENS_GENERES_REFORMULATION,
        "temperature": TEMPERATURE_REFORMULATION,
        "min_new_tokens": MIN_TOKENS_GENERES_REFORMULATION,
    }
    assert 0.0 < TEMPERATURE_REFORMULATION < 1.0
    assert 0 < MIN_TOKENS_GENERES_REFORMULATION < NOMBRE_TOKENS_GENERES_REFORMULATION


def test_ne_lit_ni_n_ecrit_jamais_rejected():
    exemple = _exemple_dpo()
    moteur = FauxMoteurInference()
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer([exemple])

    for messages in moteur.appels:
        for message in messages:
            assert "mauvaise reponse" not in message["content"]


def test_tronque_les_entrees_tres_longues_avant_de_les_envoyer_au_modele():
    """
    Cf. AGENTS.md/docstring du module : le 4e job DPO reel a montre 20/20
    echecs de diagnostic avec une sortie VIDE sur une entree tres longue
    (300-800+ mots), jamais teste jusque-la. `_tronquer_texte_chosen` coupe
    l'entree AVANT de l'inserer dans le message envoye au modele, sur une
    limite de phrase, jamais en plein mot/plein phrase.
    """
    phrase = "Ceci est une phrase clinique de test qui se repete plusieurs fois. "
    texte_tres_long = phrase * 50
    assert len(texte_tres_long) > LONGITUD_MAX_ENTREE_REFORMULATION

    exemple = _exemple_dpo(texte_chosen=texte_tres_long)
    moteur = FauxMoteurInference()
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer([exemple])

    (messages,) = moteur.appels
    contenu_envoye = messages[0]["content"]
    assert texte_tres_long not in contenu_envoye

    marqueur = "Reponse a reformuler :\n"
    entree_envoyee = contenu_envoye[contenu_envoye.rindex(marqueur) + len(marqueur) :]
    assert len(entree_envoyee) <= LONGITUD_MAX_ENTREE_REFORMULATION
    assert entree_envoyee.endswith(".")


def test_ne_tronque_pas_les_entrees_courtes():
    exemple = _exemple_dpo(texte_chosen="reponse choisie originale")
    moteur = FauxMoteurInference()
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer([exemple])

    (messages,) = moteur.appels
    assert "reponse choisie originale" in messages[0]["content"]


def test_echec_de_parsing_capture_les_tokens_entree_sortie_pour_diagnostic():
    """
    Cf. AGENTS.md : diagnostic de longueur (21/09/2026, quatrieme job DPO
    reel) - les tokens entree/sortie de `ReponseModele` (deja calcules par
    `TransformersInferenceAdapter.generer()`, jamais exposes ici avant ce
    correctif) sont recopies dans `EchecReformulation`, pour confirmer ou
    infirmer l'hypothese de degenerescence liee a la longueur avec des
    donnees reelles au prochain run GPU, plutot que de deviner encore.
    """
    exemple_degenere = _exemple_dpo(texte_chosen="reponse degeneree")
    reponse_scriptee = ReponseModele(
        texte="texte libre, pas de <think> ni de JSON",
        nombre_tokens_entree=812,
        nombre_tokens_sortie=0,
    )

    moteur = FauxMoteurInference()
    moteur.reponses_speciales["reponse degeneree"] = reponse_scriptee
    repository = FauxRepositoryReformule()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer([exemple_degenere])

    (echec,) = cas_usage.echantillon_echecs_reformulation
    assert echec.nombre_tokens_entree == 812
    assert echec.nombre_tokens_sortie == 0


def test_persiste_en_un_seul_sauvegarder_plusieurs():
    """Cf. AGENTS.md : jamais un sauvegarder() par item, cout O(n^2)."""
    exemples = [_exemple_dpo() for _ in range(4)]
    moteur = FauxMoteurInference()

    class RepositoryCompteAppels(FauxRepositoryReformule):
        def __init__(self):
            super().__init__()
            self.appels_sauvegarder = 0
            self.appels_sauvegarder_plusieurs = 0

        def sauvegarder(self, item):
            self.appels_sauvegarder += 1
            super().sauvegarder(item)

        def sauvegarder_plusieurs(self, items):
            self.appels_sauvegarder_plusieurs += 1
            super().sauvegarder_plusieurs(items)

    repository = RepositoryCompteAppels()
    cas_usage = ReformulerPreferenceDpoUseCase(moteur=moteur, repository_reformule=repository)

    cas_usage.executer(exemples)

    assert repository.appels_sauvegarder == 0
    assert repository.appels_sauvegarder_plusieurs == 1
