"""
Banc de test latence/robustesse/tracabilite contre une API CHSA Triage
reellement en cours d'execution (locale ou le Space HF deploye).

Comble la partie "tests de latence, de robustesse, ainsi que des audits
de tracabilite des interactions" de la mission (PDF source, Etape 3,
page 8), cf. l'audit `m14-ocr-nfr-latencia-robustez-trazabilidad` §5
etapes 7-8 : d'abord valide a cout nul contre le mode local
(`CHSA_MOTEUR_INFERENCE=local`), puis pointe vers l'endpoint reel une
fois le harnais confirme correct.

N'importe aucun module du projet non standard (httpx seul) : ce script
est un outil CLI ponctuel, pas un composant de l'architecture
hexagonale, jamais importe par `src/`/`interfaces/`.

Usage :
    uv run python scripts/benchmarker_latence_robustesse.py \
        --base-url http://127.0.0.1:7860 --cle-api change-moi \
        --sortie-json resultats.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass

import httpx

SCENARIOS_CLINIQUES = [
    {
        "nom": "urgent_douleur_thoracique",
        "tours": [
            "Homme de 62 ans, douleur thoracique intense depuis 30 minutes, "
            "irradiant dans le bras gauche, antecedent d'infarctus il y a 3 ans.",
            "La douleur est oppressante, 8/10, avec sueurs et nausees.",
        ],
    },
    {
        "nom": "modere_fievre_toux",
        "tours": [
            "Femme de 34 ans, fievre a 38,5°C depuis 2 jours, toux seche, "
            "pas d'antecedent particulier.",
            "Pas de difficulte respiratoire, fatigue moderee.",
        ],
    },
    {
        "nom": "leger_entorse_cheville",
        "tours": [
            "Homme de 28 ans, entorse de cheville en jouant au foot il y a 1 heure, "
            "douleur moderee, peut poser le pied.",
            "Pas de deformation visible, gonflement leger.",
        ],
    },
]


@dataclass
class MesureRequete:
    nom: str
    code_http: int
    latence_ms: float
    succes: bool
    detail: str = ""


@dataclass
class ResultatBenchmark:
    latences_entretien: list[MesureRequete]
    latences_diagnostic: list[MesureRequete]
    robustesse: list[MesureRequete]
    exemples_qualite: list[dict]


async def _appel_mesure(
    client: httpx.AsyncClient,
    nom: str,
    methode: str,
    url: str,
    **kwargs,
) -> tuple[MesureRequete, httpx.Response | None]:
    debut = time.perf_counter()
    try:
        reponse = await client.request(methode, url, **kwargs)
    except httpx.HTTPError as erreur:
        latence_ms = (time.perf_counter() - debut) * 1000
        return (
            MesureRequete(
                nom=nom, code_http=0, latence_ms=latence_ms, succes=False, detail=str(erreur)
            ),
            None,
        )
    latence_ms = (time.perf_counter() - debut) * 1000
    return (
        MesureRequete(
            nom=nom,
            code_http=reponse.status_code,
            latence_ms=latence_ms,
            succes=reponse.status_code < 400,
        ),
        reponse,
    )


async def mesurer_latence_entretiens(
    client: httpx.AsyncClient, base_url: str, headers: dict, nombre_scenarios: int
) -> tuple[list[MesureRequete], list[dict]]:
    """Deroule `nombre_scenarios` entretiens reels (plusieurs tours +
    diagnostic), mesure chaque requete, et capture les reponses pour le
    jugement qualitatif (Phase 2 point 5)."""
    mesures: list[MesureRequete] = []
    exemples: list[dict] = []

    for scenario in SCENARIOS_CLINIQUES[:nombre_scenarios]:
        mesure, reponse = await _appel_mesure(
            client, f"creer_conversation[{scenario['nom']}]", "POST",
            f"{base_url}/conversations", headers=headers,
        )
        mesures.append(mesure)
        if reponse is None or not mesure.succes:
            continue
        conversation_id = reponse.json()["conversation_id"]

        historique_reponses = []
        for i, tour in enumerate(scenario["tours"]):
            mesure, reponse = await _appel_mesure(
                client, f"message[{scenario['nom']}][{i}]", "POST",
                f"{base_url}/conversations/{conversation_id}/messages",
                headers=headers, json={"message": tour},
            )
            mesures.append(mesure)
            if reponse is not None and mesure.succes:
                historique_reponses.append(reponse.json()["message_assistant"])

        mesure, reponse = await _appel_mesure(
            client, f"diagnostic[{scenario['nom']}]", "POST",
            f"{base_url}/conversations/{conversation_id}/diagnostic",
            headers=headers,
        )
        mesures.append(mesure)
        diagnostic_json = reponse.json() if reponse is not None else None

        exemples.append(
            {
                "scenario": scenario["nom"],
                "tours_envoyes": scenario["tours"],
                "reponses_entretien": historique_reponses,
                "diagnostic": diagnostic_json,
                "latences_ms": [
                    m.latence_ms
                    for m in mesures
                    if scenario["nom"] in m.nom
                ],
            }
        )

    return mesures, exemples


async def executer_batterie_robustesse(
    client: httpx.AsyncClient, base_url: str, headers: dict
) -> list[MesureRequete]:
    """Concurrence (N creations simultanees), payload malforme,
    conversation_id inexistant : cf. audit §5 etape 8."""
    resultats: list[MesureRequete] = []

    # Requetes concurrentes reelles (meme endpoint, en parallele).
    taches = [
        _appel_mesure(
            client, f"concurrence[{i}]", "POST", f"{base_url}/conversations", headers=headers
        )
        for i in range(5)
    ]
    for mesure, _ in await asyncio.gather(*taches):
        resultats.append(mesure)

    # Payload malforme : champ "message" absent.
    mesure, reponse = await _appel_mesure(
        client, "payload_malforme_champ_absent", "POST",
        f"{base_url}/conversations", headers=headers,
    )
    resultats.append(mesure)
    if reponse is not None and mesure.succes:
        conversation_id = reponse.json()["conversation_id"]
        mesure, _ = await _appel_mesure(
            client, "payload_malforme_message_manquant", "POST",
            f"{base_url}/conversations/{conversation_id}/messages",
            headers=headers, json={"pas_le_bon_champ": "oops"},
        )
        resultats.append(mesure)

    # conversation_id inexistant.
    mesure, _ = await _appel_mesure(
        client, "conversation_id_inexistant_messages", "POST",
        f"{base_url}/conversations/conv-inexistante-123/messages",
        headers=headers, json={"message": "test"},
    )
    resultats.append(mesure)
    mesure, _ = await _appel_mesure(
        client, "conversation_id_inexistant_diagnostic", "POST",
        f"{base_url}/conversations/conv-inexistante-123/diagnostic",
        headers=headers,
    )
    resultats.append(mesure)

    return resultats


def resumer_latences(mesures: list[MesureRequete], prefixe: str) -> dict:
    valeurs = [m.latence_ms for m in mesures if prefixe in m.nom and m.succes]
    if not valeurs:
        return {"nombre": 0}
    return {
        "nombre": len(valeurs),
        "moyenne_ms": statistics.mean(valeurs),
        "mediane_ms": statistics.median(valeurs),
        "min_ms": min(valeurs),
        "max_ms": max(valeurs),
    }


async def main_async(args: argparse.Namespace) -> ResultatBenchmark:
    headers = {"X-API-Key": args.cle_api}
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        latences_entretien, exemples = await mesurer_latence_entretiens(
            client, args.base_url, headers, args.nombre_scenarios
        )
        robustesse = await executer_batterie_robustesse(
            client, args.base_url, headers
        )

    return ResultatBenchmark(
        latences_entretien=latences_entretien,
        latences_diagnostic=[],
        robustesse=robustesse,
        exemples_qualite=exemples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cle-api", required=True)
    parser.add_argument("--nombre-scenarios", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sortie-json", default=None)
    args = parser.parse_args()

    resultat = asyncio.run(main_async(args))

    print("\n=== Latences entretien/diagnostic ===")
    print(json.dumps(resumer_latences(resultat.latences_entretien, "message"), indent=2))
    print(json.dumps(resumer_latences(resultat.latences_entretien, "diagnostic"), indent=2))

    print("\n=== Robustesse ===")
    for mesure in resultat.robustesse:
        print(f"{mesure.nom:45s} http={mesure.code_http:4d} latence_ms={mesure.latence_ms:8.1f} succes={mesure.succes}")

    if args.sortie_json:
        donnees = {
            "latences_entretien": [asdict(m) for m in resultat.latences_entretien],
            "robustesse": [asdict(m) for m in resultat.robustesse],
            "exemples_qualite": resultat.exemples_qualite,
        }
        with open(args.sortie_json, "w", encoding="utf-8") as f:
            json.dump(donnees, f, indent=2, ensure_ascii=False)
        print(f"\nResultats complets ecrits dans {args.sortie_json}")


if __name__ == "__main__":
    main()
