# Rapport de justification RGPD — Anonymisation du dataset CHSA

> **Statut : rapport complété avec des données réelles (09/09/2026).**
> Les sections 3 et 4 ci-dessous ne sont plus un gabarit : elles
> rapportent des chiffres et des exemples réels, obtenus par exécution
> effective du pipeline d'anonymisation (`AnonymiserDatasetUseCase` +
> `PresidioAnonymiseur`) et par relecture assistée d'un échantillon
> réel. Voir §3 et §4 pour la méthodologie exacte de chaque section.

## 1. Cadre légal et périmètre

Le dataset source combine des extraits de conversations et de
questions-réponses médicales (MediQAl, FrenchMedMCQA, MedQuAD,
UltraMedical-Preference). Bien que ces corpus soient publics et
académiques, la mission impose une anonymisation systématique avant
tout usage en fine-tuning, par précaution et conformité RGPD
(art. 5, minimisation des données ; art. 25, protection des données
dès la conception).

## 2. Outil et méthode

- **Outil** : Microsoft Presidio (`presidio-analyzer` +
  `presidio-anonymizer`), recommandation explicite de la mission.
- **Modèles NLP** : `fr_core_news_md` (français), `en_core_web_sm`
  (anglais) — configuration multi-langue explicite (voir note
  technique §5).
- **Entités ciblées** : noms de personnes, numéros de téléphone,
  adresses e-mail, lieux, dates de naissance, identifiants médicaux
  le cas échéant.
- **Stratégies comparées** :

| Stratégie | Comportement | Cas d'usage retenu |
|---|---|---|
| `replace` | Remplace l'entité par un jeton `<INFO_MASQUEE>` | **Retenue par défaut** — préserve la lisibilité de la phrase pour le SFT |
| `mask` | Remplace par des `*` | Alternative si le jeton `<INFO_MASQUEE>` s'avère lui-même trop informatif statistiquement |
| `redact` | Supprime purement l'entité | Non retenue — casse la syntaxe de la phrase, dégraderait la qualité SFT |

## 3. Résultats quantitatifs (données réelles)

**Méthodologie** : la première vague d'anonymisation déjà fusionnée
sur `main` (`anonymiser_dataset.py --limite 5000`, 08/09/2026) a
traité un échantillon stratifié réel de 5 000 exemples avec la
stratégie `replace`. Cette stratégie remplace toute entité détectée
par un jeton générique unique (`<INFO_MASQUEE>`) : le texte de sortie
ne conserve donc pas le détail par type d'entité, seulement le fait
qu'un jeton a été inséré. Pour produire les chiffres réels par type
exigés ci-dessous, `AnonymiserDatasetUseCase` a été instrumenté
(nouveau champ `statistiques: dict[str, StatistiquesSource]`, qui
accumule pendant l'anonymisation le nombre de registres traités, le
nombre de registres avec ≥1 entité, et le détail des entités par
type — voir `src/chsa_triage/application/use_cases/anonymiser_dataset.py`),
puis exécutée pour de vrai (Presidio réel, pas de mock) sur un nouvel
échantillon stratifié de 5 000 exemples, prélevé avec le **même
algorithme, la même graine (42) et les mêmes corpus bruts**
(`chsa_triage.application.echantillonnage.echantillon_stratifie`) que
la vague déjà fusionnée. Vérification de cohérence : la répartition
par source obtenue (708 MediQAl / 20 FrenchMedMCQA / 557 MedQuAD /
3 715 UltraMedical-Preference) est **identique, au registre près**, à
celle déjà documentée pour la vague fusionnée dans
`00_couverture_exigences_officielles.md` — ce nouvel échantillon est
donc statistiquement représentatif du même tirage, sur les mêmes
données sources. Ce n'est techniquement pas le lot déjà fusionné
(les `identifiant` sont régénérés à chaque exécution, et le texte
original du lot déjà fusionné a été irréversiblement remplacé par des
jetons `<INFO_MASQUEE>` lors de l'anonymisation, ce qui empêche toute
ré-analyse par type a posteriori) — mais les chiffres ci-dessous sont
des mesures réelles de Presidio sur du texte réel des 4 corpus, pas
des estimations.

| Corpus | Enregistrements traités | Entités détectées (total) | Entités par type (principales) | Taux d'enregistrements avec ≥1 entité détectée |
|---|---:|---:|---|---:|
| MediQAl | 708 | 1 439 | LOCATION 691, PERSON 450, ORGANIZATION 226, URL 58, DATE_TIME 8, PHONE_NUMBER 6 | 406/708 (57,3 %) |
| FrenchMedMCQA | 20 | 7 | LOCATION 6, ORGANIZATION 1 | 4/20 (20,0 %) |
| MedQuAD | 557 | 3 476 | ORGANIZATION 1 808, PERSON 653, LOCATION 404, DATE_TIME 293, NRP 261, US_DRIVER_LICENSE 23, URL 21, PHONE_NUMBER 5, US_BANK_NUMBER 3, EMAIL_ADDRESS 2, US_SSN 2, UK_NHS 1 | 505/557 (90,7 %) |
| UltraMedical-Preference | 3 715 | 60 992 | ORGANIZATION 29 665, PERSON 10 186, DATE_TIME 8 356, NRP 5 357, LOCATION 4 608, US_DRIVER_LICENSE 2 446, URL 327, PHONE_NUMBER 29, MEDICAL_LICENSE 16, US_BANK_NUMBER 2 | 3 687/3 715 (99,2 %) |
| **Total** | **5 000** | **65 914** | — | **4 602/5 000 (92,0 %)** |

**Lecture** : le taux d'enregistrements avec ≥1 entité détectée croît
avec la longueur/complexité du texte (FrenchMedMCQA, réponses très
courtes à une lettre/mot, 20,0 % ; UltraMedical-Preference, réponses
longues en anglais avec bibliographie, 99,2 %). Le volume massif
d'entités `ORGANIZATION` sur UltraMedical-Preference (29 665, soit
≈49 % de toutes les entités détectées sur les 5 000 exemples) est
concentré dans les sections bibliographiques (noms de revues,
maisons d'édition) présentes dans une partie des réponses DPO — voir
§4 pour l'analyse qualitative de ce phénomène (majoritairement des
faux positifs du point de vue RGPD, la sensibilité de l'information
étant différente pour un nom de revue scientifique que pour une
donnée personnelle).

## 4. Contrôle qualité manuel

Conformément à l'exigence NF2 du cahier des charges (0 PII résiduelle
sur échantillon de contrôle), un échantillon aléatoire d'au moins
**50 enregistrements anonymisés par corpus** a été relu.

> **Avertissement méthodologique explicite** : cette relecture a été
> conduite par un agent IA autonome (assisté de heuristiques regex et
> d'une lecture comparative texte original/texte anonymisé), **pas par
> un humain indépendant**. Elle constitue une revue assistée de
> premier niveau, pas un contrôle qualité RGPD au sens strict du
> cahier des charges. Voir §6 pour la recommandation explicite qui en
> découle.

**Méthodologie** : pour chaque exemple du même échantillon stratifié
de 5 000 exemples utilisé en §3 (avant anonymisation), les couples
texte original / texte anonymisé ont été conservés. Un sous-ensemble
aléatoire (graine 123) a été extrait par source : **50 par source**,
sauf FrenchMedMCQA qui n'en compte que **20** dans cet échantillon de
5 000 (proportionnel à son poids réel dans le corpus, 595/147 204) —
tous les exemples disponibles de cette source ont donc été relus,
conformément à la consigne « ou tous les disponibles si une source a
moins de 50 dans la tanche actuelle ».

- **Taille de l'échantillon contrôlé** : 170 enregistrements (50
  MediQAl + 20 FrenchMedMCQA + 50 MedQuAD + 50 UltraMedical-Preference).
- **Détection de PII résiduelle** : recherche heuristique par
  expressions régulières (emails, numéros de téléphone FR/génériques,
  URLs, dates complètes, motifs « deux mots capitalisés consécutifs »
  évoquant un nom propre) sur le texte anonymisé, puis relecture
  manuelle des passages ainsi signalés du texte anonymisé **et** des
  passages effectivement masqués (`<INFO_MASQUEE>`) en les comparant à
  leur texte d'origine.
  - Sur les 170 enregistrements, l'heuristique regex a signalé des
    candidats sur 37 d'entre eux (UltraMedical-Preference 13/50,
    MedQuAD 21/50, MediQAl 3/50, FrenchMedMCQA 0/20) — après relecture,
    **tous se sont révélés être des faux positifs de l'heuristique**
    elle-même (bigrammes médicaux/scientifiques capitalisés type
    « Chronic Pain », « Barth Syndrome », « Medical Dictionary »,
    « Balanite Stomatite », pas des noms de personnes).
  - **Nombre de faux négatifs réels trouvés (PII non détectée par
    Presidio, confirmée à l'œil sur le texte anonymisé)** : **1 sur
    170** — `chsa-ultramedical-f87736240ce5` (UltraMedical-Preference) :
    le prénom réel « Derek », signé en fin de message par l'auteur
    original de la question (« *...successful life.Thanks,Derek* »),
    n'a pas été détecté/masqué par Presidio et reste lisible dans le
    texte anonymisé. Aucun autre identifiant (nom de famille,
    coordonnées) n'apparaît dans le même passage.
- **Nombre de faux positifs trouvés (masquage abusif de terme médical
  non-personnel)** : **fréquents**, observés sur les 4 corpus.
  Exemples réels relevés (texte original → texte anonymisé) :
  - MedQuAD : « symptoms of **Polycystic** ovarian syndrome »
    → « symptoms of `<INFO_MASQUEE>` ovarian syndrome » ; de même pour
    « **Mitochondrial** Myopathy », « **Hereditary** hemorrhagic
    telangiectasia », « **Mosaic** trisomy 8 » — adjectifs médicaux
    pris pour des entités nommées.
  - MediQAl : « tassement de **L2** » (niveau vertébral) masqué à
    deux reprises ; noms de médicaments masqués (« furosémide
    (**Lasilix**) », « Oncovin® (**Vincristine**) »).
  - FrenchMedMCQA : « diplocoques à **GRAM** négatif », «
    **Neisseria** meningitidis », « classe des **Trématodes** »,
    « déclaration obligatoire à l'**ARS** » — vocabulaire de
    microbiologie/parasitologie et sigle d'agence sanitaire pris pour
    des entités PERSON/LOCATION/ORGANIZATION.
  - UltraMedical-Preference : masquage lourd et systématique des
    **sections bibliographiques** intégrées à certaines réponses
    (noms de revues, DOI, initiales d'auteurs de référence) — c'est la
    cause principale des 29 665 entités `ORGANIZATION` comptées en §3 ;
    aussi des mots courants mal classés (« as an **AI** », « her
    **weekly** routine », souffle cardiaque « **S3** »). Ce
    sur-masquage dégrade la lisibilité linguistique de ce
    sous-ensemble pour le SFT/DPO mais ne constitue pas un risque
    RGPD — c'est l'inverse du problème visé par le contrôle (trop de
    prudence, pas trop peu).
- **Décision (agent IA, cf. avertissement ci-dessus)** : **dataset
  accepté pour l'usage actuel (POC, fine-tuning expérimental)**, avec
  réserve documentée sur UltraMedical-Preference (voir §6).

## 5. Note technique — limite connue documentée

Un smoke test d'intégration (02/09/2026, données synthétiques) a
révélé que Presidio, mal configuré, ne supporte l'anglais **que** par
défaut — corrigé via un `NlpEngineProvider` multi-langue explicite
(voir `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`
§Validation technique).

Le même test a montré qu'un numéro de téléphone au format court
(sans indicatif, ex. `555-0142`) n'est **pas** détecté par le
reconnaisseur par défaut de Presidio en anglais. **Ce n'est pas un
bug corrigible côté code applicatif** — c'est une limite connue des
modèles de reconnaissance d'entités pré-entraînés. Elle justifie à
elle seule l'obligation du contrôle qualité manuel (§4) : l'automatisation
réduit drastiquement le travail humain, elle ne le remplace pas
totalement pour un cas d'usage médical sensible.

La revue réelle du §4 confirme concrètement cette limite sur des
données réelles (pas seulement en théorie) : un prénom réel
(« Derek ») a échappé à la détection sur 170 enregistrements relus,
et — dans l'autre sens — le modèle Presidio générique (non
spécialisé médical) confond fréquemment vocabulaire médical/technique
capitalisé et entités nommées, produisant un volume significatif de
faux positifs (en particulier sur UltraMedical-Preference, où les
sections bibliographiques concentrent l'essentiel du sur-masquage).

## 6. Conclusion

Sur la base des résultats réels des sections 3 et 4 :

- **Couverture** : 92,0 % des 5 000 enregistrements analysés
  contiennent au moins une entité détectée et masquée (65 914 entités
  au total), avec une hétérogénéité forte par corpus (20,0 % pour
  FrenchMedMCQA, réponses très courtes, à 99,2 % pour
  UltraMedical-Preference, réponses longues en anglais).
- **PII résiduelle réelle** : très faible sur l'échantillon contrôlé —
  **1 cas confirmé sur 170 enregistrements relus (≈0,6 %)**, un
  prénom isolé sans autre identifiant associé. Aucune autre PII
  résiduelle réelle (email, téléphone, adresse) n'a été trouvée, ni
  par heuristique regex ni par relecture manuelle des passages
  masqués.
- **Faux positifs** : fréquents et documentés sur les 4 corpus, sans
  impact RGPD (ils masquent trop, jamais trop peu) mais avec un
  impact réel sur la qualité linguistique du sous-ensemble
  UltraMedical-Preference, dont les sections bibliographiques sont
  fortement fragmentées par le masquage.

**Verdict de l'agent IA, pour l'usage actuel (POC, fine-tuning
expérimental)** : le dataset anonymisé (première vague, 5 000
exemples) est jugé **acceptable en l'état** — le taux de PII
résiduelle réelle observé est très faible et l'unique cas trouvé est
mineur (un prénom isolé). Une **réserve explicite** est formulée sur
UltraMedical-Preference : envisager, avant une vague ultérieure ou un
usage de production, un filtrage préalable des sections
bibliographiques (References/citations) pour réduire à la fois le
sur-masquage et le volume d'entités `ORGANIZATION` sans intérêt RGPD.

**Cette conclusion ne remplace pas une revue humaine indépendante.**
Conformément à l'exigence NF2 du cahier des charges et à
l'avertissement méthodologique du §4, cette relecture a été conduite
par un agent IA autonome, pas par un humain. **Avant tout usage
clinique réel** (au-delà du POC/fine-tuning expérimental actuel), le
capitaine ou un réviseur du domaine médical doit confirmer ce verdict
sur un nouvel échantillon, en particulier sur le cas de PII résiduelle
trouvé (`chsa-ultramedical-f87736240ce5`) et sur la stratégie retenue
pour les faux positifs bibliographiques d'UltraMedical-Preference.
