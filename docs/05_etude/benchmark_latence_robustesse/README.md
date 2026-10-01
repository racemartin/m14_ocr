# Benchmark latence / robustesse / traçabilité contre l'API déployée réelle

Réponse à la partie "chiffrée en conditions réelles" de la mission (PDF
source, Étape 3, page 8/9 : "Réaliser des tests de latence, de
robustesse, ainsi que des audits de traçabilité des interactions" +
"Mesurer la latence et le temps de réponse en conditions réalistes"),
reste du plan de l'audit `m14-ocr-nfr-latencia-robustez-trazabilidad`
(§5, étapes 7-8), suite de la tâche `m14-ocr-metricas-graficos-benchmark-latencia`.

Date du run réel : 2026-10-02. Space testé : `mombasstic/chsa-triage-api`
(Docker/GPU, vLLM + checkpoint DPO-LoRA `mombasstic/chsa-triage-dpo-lora`).

## Constat de départ : le Space était déjà allumé, pas en pause

Avant toute action, l'état réel du Space a été vérifié (`hf spaces
restart`/`hf spaces wait` n'ont donc pas été utilisés) : l'API HF
Spaces renvoyait `runtime.stage: RUNNING` sur du `t4-small`, pas en
pause comme l'audit le supposait. Le modèle était déjà chaud (`/sante`
répondait `disponible: true`), donc les chiffres ci-dessous ne
comptent aucun temps de démarrage de conteneur, exactement l'objectif
du `hf spaces wait` prévu par le plan. Le Space a été **repassé en
pause immédiatement après ce run** (`hf spaces pause`, confirmé
`stage=PAUSED`).

## Constat d'architecture : aucune API REST publique

`deploy/space_gpu_api_vllm/demarrer.sh` documente explicitement qu'un
Space Docker n'expose qu'un seul port public (7860) : depuis le
01/10/2026, c'est Streamlit qui y est branché, l'API FastAPI (port
8001 interne) et vLLM (port 8000 interne) ne sont **jamais** joignables
depuis l'extérieur du conteneur. Conséquence directe pour ce
benchmark : impossible d'envoyer un payload JSON malformé ou un
`conversation_id` arbitraire par un simple `curl` contre le Space
réel, comme le ferait un banc de test REST classique. Seule
l'interface Streamlit publique peut être pilotée depuis l'extérieur.

Outil utilisé pour la piloter : **Playwright + Chrome système**, pas
`chrome-devtools-axi` comme prévu initialement. `chrome-devtools-axi`
s'est révélé en panne dans cet environnement (`pageId: Invalid input:
expected number, received undefined` sur `open`/`newpage`/
`selectpage`/`eval`/`run`, y compris après redémarrage du bridge,
testé via 6 approches différentes) ; contournement : `uv run --with
playwright` (installation éphémère, même discipline que
`python-pptx` ailleurs dans ce projet), piloté contre le binaire
`google-chrome` déjà présent sur la machine (`channel="chrome"`, aucun
téléchargement de navigateur par Playwright).

## Phase 0 (gratuite) : harnais validé en local avant toute dépense

Avant de toucher au Space payant, le même harnais (script
`scripts/benchmarker_latence_robustesse.py`, HTTP direct via `httpx`
contre l'API) a été validé contre le mode 100% gratuit
(`CHSA_MOTEUR_INFERENCE=local`, `llama-server` CPU déjà en cache sur
cette machine, aucun téléchargement). Résultat réel obtenu :

- `tour_entretien` et `diagnostic` journalisés avec `latence_ms`/
  `nombre_tokens_sortie` dans `metadonnees` (G1, confirmé réel, pas
  supposé).
- Un second tour a réellement échoué (`500` du serveur llama.cpp local
  sur ce modèle de base non fine-tuné pour ce format de prompt) et a
  été correctement journalisé comme `echec_inference` avec le vrai
  type d'exception (`HTTPStatusError`) avant d'être relayé en HTTP 502
  côté client (G4, confirmé réel sur un vrai échec, pas simulé).
- Robustesse : payload malformé -> `422`, `conversation_id` inexistant
  -> `404`, 5 créations de conversation concurrentes -> `200` x5.

Ce test local confirme que G1/G4 fonctionnent de bout en bout ; le
Space actuellement déployé ne les montrera qu'une fois cette branche
fusionnée et déployée (cf. section traçabilité ci-dessous).

## Latence réelle (3 scénarios cliniques, contre le Space GPU réel)

| Scénario | Tour 1 | Tour 2 | Diagnostic |
|---|---|---|---|
| Urgent (douleur thoracique) | 3,3 s | 3,7 s | 13,3 s |
| Modéré (fièvre/toux) | 3,8 s | 1,4 s | 16,4 s |
| Léger (entorse cheville) | 3,0 s | (un seul tour) | 13,2 s |

Mesure : temps entre l'envoi du message (clic "Entrée" dans le chat
Streamlit réel) et l'apparition de la réponse complète à l'écran, donc
inclut le round-trip Streamlit + API interne + vLLM, pas seulement
l'inférence brute. Tours d'entretien rapides (1,4-3,8 s), diagnostic
nettement plus lent (13-16 s, génération plus longue + structuration
JSON demandée).

## Robustesse

**Concurrence (3 sessions réelles simultanées, vrai test de charge)** :
3 conversations distinctes lancées en parallèle (3 threads Python, 3
navigateurs Chrome séparés) contre le même backend vLLM partagé, qui
n'a aucune protection de concurrence côté code
(`VllmEndpointInferenceAdapter`, cf. audit G3). Résultat réel : les 3
ont abouti sans erreur ni contamination croisée (chaque réponse reste
cohérente avec son propre message d'entrée), latences individuelles
3,8/4,8/5,0 s, confirmées réellement **simultanées** par les horodatages
du journal d'audit (écarts de 0,5 s et 0,75 s entre les 3 écritures,
pas 3,8+4,8+5,0 s cumulés en série). Durée totale de bout en bout des 3
sessions (navigation + inférence) : 24,4 s. **Nuance honnête** :
l'absence de protection de concurrence documentée par l'audit reste
vraie dans le code (aucune garantie pour une charge plus importante) ;
ce test confirme seulement qu'elle ne casse rien à ce niveau de charge
(3 requêtes), pas qu'elle tiendrait à 30 ou 300.

**Payload malformé / `conversation_id` inexistant contre le Space réel** :
**non réalisable** contre le réseau public réel, pour la raison
d'architecture documentée plus haut (aucune API REST publique sur ce
Space depuis le 01/10/2026). Cette partie de la batterie reste
couverte par les 5 tests d'intégration ajoutés dans `e6c799a`
(`tests/interfaces/test_app_api.py`, faux `MoteurInference` en
panne/timeout/malformé), qui tournent en CI contre l'application
FastAPI directement (sans passer par le réseau du Space), mais ne
peuvent pas être rejoués tels quels contre l'URL publique du Space.

**Smoke-test post-déploiement (G6)** : `GET /_stcore/health` (seul
endpoint public du Space) a répondu `200 ok` pendant toute la durée du
test, confirmant que le smoke-test déjà câblé dans `deploy.yml`
pointe bien vers un endpoint réellement atteignable.

## Traçabilité (F6) : confirmée réelle, mais sur l'ancien code déployé

Le dépôt dataset `mombasstic/chsa-triage-audit-journal` (mode
`hf_dataset`, confirmé via les métadonnées du Space) contient bien une
entrée par tour d'entretien et par appel diagnostic pour les 6
conversations réelles de ce test (3 scénarios séquentiels + 3
sessions concurrentes), avec le bon `type_evenement`/`conversation_id`/
`horodatage`, et même le bon `format_respecte` par diagnostic (`True`
pour les scénarios urgent/léger, `False` pour le scénario modéré,
cohérent avec ce qui a été observé à l'écran).

**Point important, honnête** : `metadonnees` de ces entrées réelles est
vide (`{}`) pour `tour_entretien`, et ne contient que `format_respecte`
(jamais `latence_ms`) pour `diagnostic`. Ce n'est **pas** un bug
nouveau : le Space actuellement déployé tourne sur une version du code
**antérieure** à cette tâche (le fix G1 n'a pas encore été poussé sur
`origin`/redéployé, cf. consigne de la tâche : ce push est fait par
firstmate APRÈS fusion). Le fix lui-même est déjà validé réellement en
local (section Phase 0 ci-dessus) ; une fois ce code déployé, les
mêmes interactions produiront `latence_ms`/`nombre_tokens_sortie` dans
ce même dépôt, sans action supplémentaire.

## Exemples réels (question -> réponse -> diagnostic) et mon évaluation qualité

Captures d'écran réelles : `screenshot_urgent_douleur_thoracique.png`,
`screenshot_modere_fievre_toux.png`, `screenshot_leger_entorse_cheville.png`
(ce dossier).

### Scénario urgent (douleur thoracique, 62 ans, ATCD infarctus)

> Entretien -> "Quels sont ses autres signes symptomatiques à part
> cette douleur ? [...]"
> **Diagnostic : Niveau ESI 4, Cardiovasculaire, Infarctus
> myocardiophile.**

Mon évaluation : la réponse JSON est bien formée et le raisonnement
textuel reste globalement cohérent avec une hypothèse cardiaque.
**Mais le niveau ESI 4 est cliniquement préoccupant** : une douleur
thoracique oppressive avec irradiation au bras gauche, sueurs,
nausées, et antécédent d'infarctus est un tableau classique de
haute acuité (ESI 2 attendu, pas 4). Un ESI 4 sous-triage ce cas de
façon potentiellement dangereuse en conditions réelles. Ceci confirme
concrètement, sur un exemple réel, pourquoi NF4 (garde-fou de sécurité
clinique, juge LLM) reste une décision produit encore ouverte et non
implémentée (cf. cahier des charges) : ce POC ne doit pas être utilisé
sans supervision humaine systématique.

### Scénario modéré (fièvre/toux, 34 ans)

> Diagnostic brut (format NON respecté) : `<think>[...]</think>
> {"niveau": 3, "categorie": "Infection", "ressources_estimees":
> "[Examination physique complète] [Radiographie thoracique]}` (JSON
> invalide : crochet non fermé dans la valeur de chaîne).

Mon évaluation : le parseur strict a correctement rejeté ce JSON
invalide et affiché le texte brut au lieu de planter ou d'inventer des
champs (comportement attendu, déjà testé). Le niveau 3/Infection
sous-jacent reste cliniquement plausible pour une fièvre isolée sans
détresse respiratoire. Bon exemple réel du chemin de secours
(`format_respecte=False`) fonctionnant comme conçu.

### Scénario léger (entorse cheville, 28 ans)

> **Diagnostic : Niveau ESI 3, [Traumatologie], Examen physiologique.**

Mon évaluation : JSON bien formé. Niveau 3 un peu élevé pour une
entorse modérée sans déformation (4-5 aurait été plus attendu), mais
dans le sens prudent (sur-triage, pas sous-triage) : moins
préoccupant que le cas urgent ci-dessus.

### Synthèse qualité

Entretien : questions de suivi plausibles mais peu ciblées sur le
motif principal (demande systématique du poids, questions parfois
tangentielles), reflet connu des limites d'un modèle base 1,7B. Sortie
structurée : 2/3 bien formée, 1/3 correctement rejetée plutôt que
silencieusement corrompue. Sécurité clinique : le cas le plus urgent
des 3 scénarios a été le moins bien triagé (ESI 4 au lieu de 2) ;
risque réel documenté ici avec un exemple concret à l'appui, pas une
crainte abstraite.

## Fichiers

- `scripts/benchmarker_latence_robustesse.py` : harnais HTTP direct
  (httpx), valide contre le mode local gratuit ; réutilisable une fois
  l'API REST à nouveau joignable (ou en local).
- `screenshot_*.png` (ce dossier) : preuve visuelle des 3 scénarios
  réels.
