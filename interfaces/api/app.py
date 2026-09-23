"""
Fabrique de l'application FastAPI (Etape 4, F1/F2/F3/F4/F6/F7). Aucune
lecture de variable d'environnement ici (fait dans `main.py`, le point
d'entree ASGI reel) : `creer_application()` recoit toutes ses
dependances par injection explicite, testable avec des faux
adaptateurs en memoire sans reseau/GPU (`tests/interfaces/test_app_api.py`),
meme discipline que les cas d'usage `application/use_cases/`.

Trois endpoints metier (F1, "questionnaire intelligent adaptatif" +
bouton explicite "obtenir le diagnostic", cf. cahier des charges §3
note du 23/09/2026) :
- `POST /conversations` : demarre un entretien (historique vide).
- `POST /conversations/{id}/messages` : poursuit l'entretien (un tour).
- `POST /conversations/{id}/diagnostic` : declenche l'appel diagnostic
  (F2/F3/F4), jamais automatique, toujours sur decision de l'infirmier.
- `GET /conversations/{id}` : relit l'historique courant (utilitaire
  cote client, ex. reaffichage apres reconnexion).

Toutes les routes metier exigent la cle API (`securite.py`) ; `/sante`
reste ouverte.

`/sante` sert un double usage, distingue explicitement dans son corps
de reponse plutot que par le code HTTP : verification de vivacite du
conteneur (Docker HEALTHCHECK, `Dockerfile`) ET verification de
disponibilite du moteur d'inference distant (vLLM, cf. Space
Streamlit/CPU de test qui sonde cet endpoint en boucle jusqu'a ce que
le Space Docker/GPU compagnon soit pret, `interfaces/web/`). L'endpoint
retourne donc TOUJOURS HTTP 200 (jamais 503) : un Docker HEALTHCHECK
qui redemarrerait le conteneur API parce que vLLM met du temps a
charger le modele n'aiderait en rien (redemarrer l'API ne demarre pas
vLLM plus vite) ; seul le champ `disponible` du corps JSON distingue
les deux etats. `verificateur_sante_moteur` est injecte (meme
discipline que `moteur_inference`/`journal_audit`) : `main.py` le
branche sur `VllmEndpointInferenceAdapter.verifier_sante()` en mode
`distant`, ou laisse le defaut (`disponible=True`, aucune verification
pertinente) en mode `local`/llama.cpp.
"""

from __future__ import annotations

from typing import Callable

from fastapi import Depends, FastAPI, HTTPException

from chsa_triage.application.use_cases.E4_00_uc_poursuivre_entretien import (
    PoursuivreEntretienUseCase,
)
from chsa_triage.application.use_cases.E4_01_uc_obtenir_diagnostic import (
    ObtenirDiagnosticUseCase,
)
from chsa_triage.domain.ports.journal_audit import JournalAudit
from chsa_triage.domain.ports.moteur_inference import MoteurInference
from interfaces.api.magasin_conversations import MagasinConversationsMemoire
from interfaces.api.schemas import (
    ConversationReponse,
    DemarrerConversationReponse,
    DiagnosticReponse,
    MessageEntretienReponse,
    MessageEntretienRequete,
    SanteReponse,
    TourHistorique,
)
from interfaces.api.securite import creer_dependance_verification_cle_api

VERSION_MODELE_PAR_DEFAUT = "mombasstic/chsa-triage-dpo-lora"


def _verificateur_sante_par_defaut() -> dict:
    return {"disponible": True, "detail": "aucune verification specifique pour ce moteur d'inference"}


def creer_application(
    *,
    moteur_inference: MoteurInference,
    journal_audit: JournalAudit,
    cle_api: str,
    version_modele: str = VERSION_MODELE_PAR_DEFAUT,
    verificateur_sante_moteur: Callable[[], dict] | None = None,
) -> FastAPI:
    app = FastAPI(
        title="CHSA Triage API",
        description="API de demonstration de l'agent IA de triage medical (POC, Etape 4).",
        version="0.1.0",
    )

    magasin = MagasinConversationsMemoire()
    poursuivre_entretien = PoursuivreEntretienUseCase(
        moteur=moteur_inference, journal=journal_audit, version_modele=version_modele
    )
    obtenir_diagnostic = ObtenirDiagnosticUseCase(
        moteur=moteur_inference, journal=journal_audit, version_modele=version_modele
    )
    verifier_cle_api = creer_dependance_verification_cle_api(cle_api)
    verifier_sante_moteur = verificateur_sante_moteur or _verificateur_sante_par_defaut

    @app.get("/sante", response_model=SanteReponse)
    def sante() -> SanteReponse:
        resultat = verifier_sante_moteur()
        return SanteReponse(disponible=resultat["disponible"], detail=resultat["detail"])

    @app.post(
        "/conversations",
        response_model=DemarrerConversationReponse,
        dependencies=[Depends(verifier_cle_api)],
    )
    def demarrer_conversation() -> DemarrerConversationReponse:
        conversation_id = magasin.creer()
        return DemarrerConversationReponse(conversation_id=conversation_id)

    @app.get(
        "/conversations/{conversation_id}",
        response_model=ConversationReponse,
        dependencies=[Depends(verifier_cle_api)],
    )
    def lire_conversation(conversation_id: str) -> ConversationReponse:
        historique = magasin.obtenir(conversation_id)
        if historique is None:
            raise HTTPException(status_code=404, detail="Conversation introuvable")
        return ConversationReponse(
            conversation_id=conversation_id,
            historique=[TourHistorique(role=m.role, contenu=m.contenu) for m in historique],
        )

    @app.post(
        "/conversations/{conversation_id}/messages",
        response_model=MessageEntretienReponse,
        dependencies=[Depends(verifier_cle_api)],
    )
    def poursuivre_conversation(
        conversation_id: str, requete: MessageEntretienRequete
    ) -> MessageEntretienReponse:
        historique = magasin.obtenir(conversation_id)
        if historique is None:
            raise HTTPException(status_code=404, detail="Conversation introuvable")

        resultat = poursuivre_entretien.executer(conversation_id, historique, requete.message)
        magasin.ajouter(conversation_id, resultat.message_utilisateur, resultat.message_assistant)

        return MessageEntretienReponse(
            conversation_id=conversation_id, message_assistant=resultat.message_assistant.contenu
        )

    @app.post(
        "/conversations/{conversation_id}/diagnostic",
        response_model=DiagnosticReponse,
        dependencies=[Depends(verifier_cle_api)],
    )
    def obtenir_diagnostic_conversation(conversation_id: str) -> DiagnosticReponse:
        historique = magasin.obtenir(conversation_id)
        if historique is None:
            raise HTTPException(status_code=404, detail="Conversation introuvable")
        if not historique:
            raise HTTPException(status_code=400, detail="L'entretien est vide, aucun diagnostic possible")

        resultat = obtenir_diagnostic.executer(conversation_id, historique)

        if resultat.diagnostic is None:
            return DiagnosticReponse(
                conversation_id=conversation_id,
                format_respecte=False,
                texte_brut=resultat.texte_brut,
            )

        return DiagnosticReponse(
            conversation_id=conversation_id,
            format_respecte=True,
            niveau=resultat.diagnostic.niveau,
            categorie=resultat.diagnostic.categorie,
            ressources_estimees=resultat.diagnostic.ressources_estimees,
            raisonnement=resultat.diagnostic.raisonnement,
            texte_brut=resultat.texte_brut,
        )

    return app
