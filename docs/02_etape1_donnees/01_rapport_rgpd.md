# Rapport de justification RGPD : Anonymisation du dataset CHSA

> **Statut (08/09/2026) : processus automatisé, ce document devient un
> rapport de méthodologie + historique.** Le
> pipeline génère désormais lui-même ses indicateurs RGPD de façon
> automatique et reproductible à chaque exécution de
> `E1_04_00_anonymiser_dataset.py`, au lieu d'un calcul manuel ponctuel
> (ce que documentaient les sections 3 et 4 ci-dessous, datées du
> 09/09/2026). **Ces chiffres restent corrects comme trace historique**
> mais portaient sur l'ANCIEN pivot (147 204 exemples, identifiants
> aléatoires); celui-ci a été régénéré le 08/09/2026 avec des
> identifiants déterministes et un dédoublonnage réel des registres
> bruts strictement identiques (nouveau total : 134 883 exemples ; voir
> `00_couverture_exigences_officielles.md` § « Identifiants
> déterministes + dédoublonnage réel + régénération du pivot »).
>
> **Pour les chiffres à jour**, consulter les rapports générés
> automatiquement à chaque exécution :
> - `data/processed/rapport_anonymisation_rgpd.{json,md}` : rapport RGPD
>   CUMULÉ (fusionne toutes les exécutions passées), par source :
>   registres traités, proportion réelle sur le total du dataset pivot,
>   taux d'enregistrements avec ≥1 entité détectée, entités par type.
>   Voir `src/chsa_triage/application/use_cases/E1_04_03_rapport_anonymisation.py`.
> - `data/processed/rapport_controle_qualite_anonymisation.{json,md}` :
>   contrôle qualité **automatisé** par comparaison du pivot original et
>   du fichier anonymisé (`E1_04_02_controler_qualite_anonymisation.py`) : regex
>   sans modèle pour les candidats de PII résiduelle (emails,
>   téléphones, URLs, dates, bigrammes capitalisés), seconde opinion
>   spaCy (mêmes modèles que `PresidioAnonymiseur`) pour départager les
>   cas ambigus; jamais de LLM. Les cas non tranchés sont marqués
>   explicitement "pendant_revision_humaine", jamais une confirmation
>   inventée. Remplace la relecture manuelle décrite
>   en §4 ci-dessous pour toute nouvelle vague anonymisée à partir du
>   08/09/2026; MAIS ne recouvre PAS rétroactivement l'ancienne vague
>   de 5 000 exemples décrite en §3-4 (pivot depuis régénéré, cette
>   vague n'existe plus sous cette forme).
>
> **Les sections 3 et 4 ci-dessous restent en l'état** comme
> méthodologie de référence et comme trace de la revue manuelle
> assistée effectuée une fois (§4, avertissement
> méthodologique explicite conservé); elles ne sont plus régénérées ni
> mises à jour désormais, ce rôle étant repris par les rapports
> automatiques ci-dessus à chaque exécution.

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
  (anglais); configuration multi-langue explicite (voir note
  technique §5).
- **Entités ciblées** : noms de personnes, numéros de téléphone,
  adresses e-mail, lieux, dates de naissance, identifiants médicaux
  le cas échéant.
- **Stratégies comparées** :

| Stratégie | Comportement | Cas d'usage retenu |
|---|---|---|
| `replace` | Remplace l'entité par un jeton `<INFO_MASQUEE>` | **Retenue par défaut**; préserve la lisibilité de la phrase pour le SFT |
| `mask` | Remplace par des `*` | Alternative si le jeton `<INFO_MASQUEE>` s'avère lui-même trop informatif statistiquement |
| `redact` | Supprime purement l'entité | Non retenue; casse la syntaxe de la phrase, dégraderait la qualité SFT |

## 3. Résultats quantitatifs (données réelles)

**Méthodologie** : la première vague d'anonymisation déjà fusionnée
sur `main` (`E1_04_00_anonymiser_dataset.py --limite 5000`, 08/09/2026) a
traité un échantillon stratifié réel de 5 000 exemples avec la
stratégie `replace`. Cette stratégie remplace toute entité détectée
par un jeton générique unique (`<INFO_MASQUEE>`) : le texte de sortie
ne conserve donc pas le détail par type d'entité, seulement le fait
qu'un jeton a été inséré. Pour produire les chiffres réels par type
exigés ci-dessous, `AnonymiserDatasetUseCase` a été instrumenté
(nouveau champ `statistiques: dict[str, StatistiquesSource]`, qui
accumule pendant l'anonymisation le nombre de registres traités, le
nombre de registres avec ≥1 entité, et le détail des entités par
type; voir `src/chsa_triage/application/use_cases/E1_04_00_anonymiser_dataset.py`),
puis exécutée pour de vrai (Presidio réel, pas de mock) sur un nouvel
échantillon stratifié de 5 000 exemples, prélevé avec le **même
algorithme, la même graine (42) et les mêmes corpus bruts**
(`chsa_triage.application.echantillonnage.echantillon_stratifie`) que
la vague déjà fusionnée. Vérification de cohérence : la répartition
par source obtenue (708 MediQAl / 20 FrenchMedMCQA / 557 MedQuAD /
3 715 UltraMedical-Preference) est **identique, au registre près**, à
celle déjà documentée pour la vague fusionnée dans
`00_couverture_exigences_officielles.md`; ce nouvel échantillon est
donc statistiquement représentatif du même tirage, sur les mêmes
données sources. Ce n'est techniquement pas le lot déjà fusionné
(les `identifiant` sont régénérés à chaque exécution, et le texte
original du lot déjà fusionné a été irréversiblement remplacé par des
jetons `<INFO_MASQUEE>` lors de l'anonymisation, ce qui empêche toute
ré-analyse par type a posteriori); mais les chiffres ci-dessous sont
des mesures réelles de Presidio sur du texte réel des 4 corpus, pas
des estimations.

| Corpus | Enregistrements traités | Entités détectées (total) | Entités par type (principales) | Taux d'enregistrements avec ≥1 entité détectée |
|---|---:|---:|---|---:|
| MediQAl | 708 | 1 439 | LOCATION 691, PERSON 450, ORGANIZATION 226, URL 58, DATE_TIME 8, PHONE_NUMBER 6 | 406/708 (57,3 %) |
| FrenchMedMCQA | 20 | 7 | LOCATION 6, ORGANIZATION 1 | 4/20 (20,0 %) |
| MedQuAD | 557 | 3 476 | ORGANIZATION 1 808, PERSON 653, LOCATION 404, DATE_TIME 293, NRP 261, US_DRIVER_LICENSE 23, URL 21, PHONE_NUMBER 5, US_BANK_NUMBER 3, EMAIL_ADDRESS 2, US_SSN 2, UK_NHS 1 | 505/557 (90,7 %) |
| UltraMedical-Preference | 3 715 | 60 992 | ORGANIZATION 29 665, PERSON 10 186, DATE_TIME 8 356, NRP 5 357, LOCATION 4 608, US_DRIVER_LICENSE 2 446, URL 327, PHONE_NUMBER 29, MEDICAL_LICENSE 16, US_BANK_NUMBER 2 | 3 687/3 715 (99,2 %) |
| **Total** | **5 000** | **65 914** | N/A | **4 602/5 000 (92,0 %)** |

**Lecture** : le taux d'enregistrements avec ≥1 entité détectée croît
avec la longueur/complexité du texte (FrenchMedMCQA, réponses très
courtes à une lettre/mot, 20,0 % ; UltraMedical-Preference, réponses
longues en anglais avec bibliographie, 99,2 %). Le volume massif
d'entités `ORGANIZATION` sur UltraMedical-Preference (29 665, soit
≈49 % de toutes les entités détectées sur les 5 000 exemples) est
concentré dans les sections bibliographiques (noms de revues,
maisons d'édition) présentes dans une partie des réponses DPO; voir
§4 pour l'analyse qualitative de ce phénomène (majoritairement des
faux positifs du point de vue RGPD, la sensibilité de l'information
étant différente pour un nom de revue scientifique que pour une
donnée personnelle).

## 4. Contrôle qualité manuel

Conformément à l'exigence NF2 du cahier des charges (0 PII résiduelle
sur échantillon de contrôle), un échantillon aléatoire d'au moins
**50 enregistrements anonymisés par corpus** a été relu.

> **Avertissement méthodologique explicite** : cette relecture a été
> conduite par un processus automatisé (heuristiques regex et
> confrontation systématique texte original/texte anonymisé), **pas par
> un humain indépendant**. Elle constitue une revue assistée de
> premier niveau, pas un contrôle qualité RGPD au sens strict du
> cahier des charges. Voir §6 pour la recommandation explicite qui en
> découle.

**Méthodologie** : pour chaque exemple du même échantillon stratifié
de 5 000 exemples utilisé en §3 (avant anonymisation), les couples
texte original / texte anonymisé ont été conservés. Un sous-ensemble
aléatoire (graine 123) a été extrait par source : **50 par source**,
sauf FrenchMedMCQA qui n'en compte que **20** dans cet échantillon de
5 000 (proportionnel à son poids réel dans le corpus, 595/147 204) ;
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
    MedQuAD 21/50, MediQAl 3/50, FrenchMedMCQA 0/20); après relecture,
    **tous se sont révélés être des faux positifs de l'heuristique**
    elle-même (bigrammes médicaux/scientifiques capitalisés type
    « Chronic Pain », « Barth Syndrome », « Medical Dictionary »,
    « Balanite Stomatite », pas des noms de personnes).
  - **Nombre de faux négatifs réels trouvés (PII non détectée par
    Presidio, confirmée à l'œil sur le texte anonymisé)** : **1 sur
    170**; `chsa-ultramedical-f87736240ce5` (UltraMedical-Preference) :
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
    telangiectasia », « **Mosaic** trisomy 8 »; adjectifs médicaux
    pris pour des entités nommées.
  - MediQAl : « tassement de **L2** » (niveau vertébral) masqué à
    deux reprises ; noms de médicaments masqués (« furosémide
    (**Lasilix**) », « Oncovin® (**Vincristine**) »).
  - FrenchMedMCQA : « diplocoques à **GRAM** négatif », «
    **Neisseria** meningitidis », « classe des **Trématodes** »,
    « déclaration obligatoire à l'**ARS** »; vocabulaire de
    microbiologie/parasitologie et sigle d'agence sanitaire pris pour
    des entités PERSON/LOCATION/ORGANIZATION.
  - UltraMedical-Preference : masquage lourd et systématique des
    **sections bibliographiques** intégrées à certaines réponses
    (noms de revues, DOI, initiales d'auteurs de référence); c'est la
    cause principale des 29 665 entités `ORGANIZATION` comptées en §3 ;
    aussi des mots courants mal classés (« as an **AI** », « her
    **weekly** routine », souffle cardiaque « **S3** »). Ce
    sur-masquage dégrade la lisibilité linguistique de ce
    sous-ensemble pour le SFT/DPO mais ne constitue pas un risque
    RGPD; c'est l'inverse du problème visé par le contrôle (trop de
    prudence, pas trop peu).
- **Décision (cf. avertissement méthodologique ci-dessus)** : **dataset
  accepté pour l'usage actuel (POC, fine-tuning expérimental)**, avec
  réserve documentée sur UltraMedical-Preference (voir §6).

## 5. Note technique : limite connue documentée

Un smoke test d'intégration (02/09/2026, données synthétiques) a
révélé que Presidio, mal configuré, ne supporte l'anglais **que** par
défaut; corrigé via un `NlpEngineProvider` multi-langue explicite
(voir `docs/02_etape1_donnees/00_couverture_exigences_officielles.md`
§Validation technique).

Le même test a montré qu'un numéro de téléphone au format court
(sans indicatif, ex. `555-0142`) n'est **pas** détecté par le
reconnaisseur par défaut de Presidio en anglais. **Ce n'est pas un
bug corrigible côté code applicatif**; c'est une limite connue des
modèles de reconnaissance d'entités pré-entraînés. Elle justifie à
elle seule l'obligation du contrôle qualité manuel (§4) : l'automatisation
réduit drastiquement le travail humain, elle ne le remplace pas
totalement pour un cas d'usage médical sensible.

La revue réelle du §4 confirme concrètement cette limite sur des
données réelles (pas seulement en théorie) : un prénom réel
(« Derek ») a échappé à la détection sur 170 enregistrements relus,
et, dans l'autre sens, le modèle Presidio générique (non
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
- **PII résiduelle réelle** : très faible sur l'échantillon contrôlé :
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

**Verdict retenu pour l'usage actuel (POC, fine-tuning
expérimental)** : le dataset anonymisé (première vague, 5 000
exemples) est jugé **acceptable en l'état**; le taux de PII
résiduelle réelle observé est très faible et l'unique cas trouvé est
mineur (un prénom isolé). Une **réserve explicite** est formulée sur
UltraMedical-Preference : envisager, avant une vague ultérieure ou un
usage de production, un filtrage préalable des sections
bibliographiques (References/citations) pour réduire à la fois le
sur-masquage et le volume d'entités `ORGANIZATION` sans intérêt RGPD.

**Cette conclusion ne remplace pas une revue humaine indépendante.**
Conformément à l'exigence NF2 du cahier des charges et à
l'avertissement méthodologique du §4, cette relecture a été conduite
par un processus automatisé, pas par un humain. **Avant tout usage
clinique réel** (au-delà du POC/fine-tuning expérimental actuel), un
réviseur du domaine médical doit confirmer ce verdict
sur un nouvel échantillon, en particulier sur le cas de PII résiduelle
trouvé (`chsa-ultramedical-f87736240ce5`) et sur la stratégie retenue
pour les faux positifs bibliographiques d'UltraMedical-Preference.

## 7. Améliorations avancées (risques réels + sur-anonymisation) : 08/09/2026

Suite à une analyse technique des risques réels de fuite
RGPD du pipeline actuel et de sur-anonymisation (âge/durée perdus sans
nécessité), quatre chantiers ont été menés dans
`PresidioAnonymiseur` et `E1_04_02_controler_qualite_anonymisation.py`. Chaque
sous-section documente une hypothèse **vérifiée contre des données et
un comportement réels**, pas contre ce qui « devrait » se passer en
théorie; y compris quand la vérification infirme partiellement
l'hypothèse de départ.

### 7.1 Recognizers personnalisés pour faux négatifs connus

**NIR français (numéro de sécurité sociale)** : absent des
recognizers par défaut de Presidio (vérifié par inspection du code
source de `presidio-analyzer` installé; ni
`predefined_recognizers/generic` ni le modèle spaCy `fr_core_news_md`
ne le couvrent). Structure réelle investiguée (pas reconstituée de
mémoire) auprès de `xml.insee.fr/schema/nir.html` et de
`fr.wikipedia.org/wiki/Numéro_de_sécurité_sociale_en_France` : 15
chiffres = sexe(1) + année(2) + mois(2) + département de naissance(2,
ou 2A/2B pour la Corse) + commune(3) + ordre(3) + clé de contrôle(2),
clé = 97 − (13 premiers chiffres mod 97), avec pour la Corse la
substitution standard A→0/−1 000 000, B→0/−2 000 000. Implémenté en
`RecognizeurNirFrance` (`PatternRecognizer` + `validate_result` qui
recalcule et compare la clé réelle; un motif « 15 chiffres » seul
produirait trop de faux positifs, la validation de clé les élimine :
un nombre à 15 chiffres pris au hasard n'a qu'1 chance sur 97 de
passer). Testé avec un NIR valide construit avec le même algorithme
(`185037511600127`, y compris la variante Corse
`278062B04500358`) et un NIR à clé invalide, en unitaire et en
intégration réelle via `AnalyzerEngine.analyze()`; cf.
`tests/infrastructure/test_presidio_anonymiseur.py`.

**Autres identifiants internes (dossier/numéro patient)** : inspection
réelle d'un échantillon de `data/processed/dataset_pivot.jsonl`
(21 922 exemples reconstruits localement : MediQAl-oeq, FrenchMedMCQA,
MedQuAD); recherche de motifs `n° patient`, `numéro de dossier`,
`dossier n°`, `IPP`, `MRN`, `medical record number`, `patient
identifier`. Seules 3 occurrences de la chaîne `IPP` trouvées, toutes
des **faux positifs** : « IPP » y désigne un Inhibiteur de la Pompe à
Protons (terme médical), jamais un « Identifiant Permanent du
Patient ». **Aucun identifiant interne réel trouvé** → aucun
recognizer ajouté pour ce cas, conformément à la consigne de ne pas
ajouter de recognizer pour un motif non confirmé dans les données
réelles (les `identifiant`/`identifiant_source_brute` du schéma pivot
sont des hash générés par le pipeline lui-même, jamais présents dans
le texte libre d'origine).

### 7.2 Entités qui se chevauchent / coupées par un saut de ligne : vérifié, pas supposé

Trois scénarios réels testés via `PresidioAnonymiseur.anonymiser()`
(Presidio réel, pas de mock) :

1. **Chevauchement réel** : `"Jean.Dupont@example.com a signalé le
   problème."` déclenche SIMULTANÉMENT `EMAIL_ADDRESS` (span entier),
   `PERSON` (même span entier; spaCy classe l'adresse comme nom de
   personne) et `URL` (sous-span `example.com`, imbriqué dans les
   deux précédents); trois entités de types différents qui se
   chevauchent/s'imbriquent. Résultat réel :
   `"<INFO_MASQUEE> a signalé le problème."`; **un seul jeton, aucun
   doublon, aucun fragment résiduel**. Un second cas de chevauchement
   du même type est apparu spontanément dans un autre test (§7.3, une
   date `05/12/2018` classée à la fois `DATE_TIME` et `PERSON` sur le
   même span) avec la même résolution correcte.
2. **Nom coupé par un saut de ligne** : `"Contactez\nJean\nDupont pour
   un avis médical urgent."` → spaCy (`fr_core_news_md`) reconnaît
   `"Jean\nDupont"` comme **une seule** entité `PERSON` malgré le `\n`
   interne, masquée comme un seul jeton :
   `"Contactez\n<INFO_MASQUEE> pour un avis médical urgent."`.

**Conclusion vérifiée** : `AnonymizerEngine.anonymize()` de Presidio
résout déjà nativement les chevauchements et les entités multi-tokens
traversant un saut de ligne; **aucun code de résolution
supplémentaire n'a été ajouté**, le risque identifié
est réel en théorie mais déjà couvert en pratique par la version de
Presidio installée. Ces trois cas sont figés en tests de régression
(`test_entites_qui_se_chevauchent_sont_resolues_sans_duplication`,
`test_nom_coupe_par_un_saut_de_ligne_est_masque_comme_une_seule_entite`)
pour que toute régression future (mise à jour de Presidio) soit
détectée automatiquement plutôt que re-vérifiée manuellement une
seule fois puis oubliée.

### 7.3 Contrôle qualité : stratum dédié « sans entité détectée »

`E1_04_02_controler_qualite_anonymisation.py` disposait déjà de tout le
nécessaire (texte original ET anonymisé pour chaque exemple) mais son
tirage stratifié (type_exemple, source) ne distinguait pas « rien
détecté » de « quelque chose détecté ». Ajout d'un stratum
INDÉPENDANT (`--taille-echantillon-sans-entite`, 40 par défaut,
graine distincte 43) qui isole spécifiquement les couples où
`texte_original == texte_anonymise` sur tous les champs texte libre,
et les soumet à la même heuristique regex + seconde opinion spaCy que
le reste; la question posée étant : « ce texte inchangé contient-il
malgré tout un motif de PII évident, signe d'un faux négatif complet
de Presidio ? ». Compteurs et section de rapport (§5 du rapport de
contrôle qualité, `nombre_disponibles_sans_entite` /
`nombre_exemples_sans_entite_observes` / `candidats_pii_sans_entite`)
toujours séparés du reste, jamais fusionnés; cf.
`ControleQualiteAnonymisation.observer_sans_entite`.

### 7.4 Âge comme quasi-identifiant clinique : généraliser, pas supprimer

**Problème réel** : le recognizer `DATE_TIME` de Presidio (porté par
le modèle spaCy sous-jacent) ne distingue pas une date de naissance
exacte (identifiante) d'un âge ou d'une durée relative (signal
clinique réel, enfant/adulte/personne âgée, pas identifiant en
soi). Masquer les deux indifféremment perd de la valeur clinique sans
gain RGPD supplémentaire.

**Vérification empirique réelle** (pas supposée) faite AVANT
d'écrire le code, sur des phrases réelles/réalistes tirées de
`data/processed/dataset_pivot.jsonl` :

- **Anglais (`en_core_web_sm`)** : le bug est réel et confirmé,
  `"7-year-old"`, `"70 year old"`, `"38-year-old"`, `"aged 3 years"`,
  `"45 years old"` sont tous étiquetés `DATE_TIME` et auraient été
  masqués sans distinction.
- **Français (`fr_core_news_md`)** : sur 8+ phrases réelles/réalistes
  testées (dont la phrase réelle exacte du corpus MediQAl `"Homme âgé
  (60 ans), chronique (7 mois), pas de fluctuations..."`, ainsi que
  `"un enfant de 2 ans"`, `"Patiente de 70 ans"`, `"Age > 60 ans"`,
  `"Il y a 3 semaines"`, `"Depuis 2 mois"`), **aucune** des mentions
  d'âge/durée n'a déclenché `DATE_TIME`; le modèle français ne
  reproduit pas (encore) ce bug sur les constructions testées.

**Décision** : les deux mécanismes ci-dessous sont appliqués pour le
FR et l'EN par cohérence de conception et par prudence (une évolution
future du modèle spaCy français pourrait changer ce comportement),
mais l'impact réel actuel est concentré côté anglais; documenté
honnêtement plutôt que présenté comme un correctif bilingue
symétrique.

1. **Normalisation de l'âge avant analyse** (`_normaliser_ages`,
   appelée en tête de `PresidioAnonymiseur.anonymiser()`) : détecte
   par regex les mentions explicites d'âge (FR : `"âgé(e) de X ans"`,
   `"âgé (X ans)"`, `"{enfant,homme,femme,patient(e)...} de X ans"` ;
   EN : `"X-year-old"`, `"X years old"`, `"aged X"`) et les remplace
   par un jeton de tranche clinique **avant** que Presidio ne voie le
   texte, afin que `DATE_TIME` ne puisse jamais l'éliminer. Tranches
   reprises telles que définies pour ce projet (aucune autre
   coupure d'âge n'étant définie dans le cahier des charges ni les
   niveaux ESI du projet) : pédiatrique 0-12, adolescent 13-17, adulte
   18-64, personne âgée 65+ → jetons `<AGE_PEDIATRIQUE>` /
   `<AGE_ADOLESCENT>` / `<AGE_ADULTE>` / `<AGE_PERSONNE_AGEE>` (FR),
   `<AGE_PEDIATRIC>` / `<AGE_ADOLESCENT>` / `<AGE_ADULT>` /
   `<AGE_ELDERLY>` (EN).
2. **Opérateur `DATE_TIME` dédié** (`_construire_operateurs`, clé
   `"DATE_TIME"` séparée de `"DEFAULT"`, via
   `OperatorConfig("custom", {"lambda": ...})`) : pour toute entité
   `DATE_TIME` restante (âge déjà neutralisé en amont), distingue une
   date calendaire absolue (jour/mois/année, `12/05/1980`, `12 janvier
   2020` → masquée comme le reste) d'une durée relative (`"il y a 3
   semaines"`, `"depuis 2 mois"`, `"pendant 2 semaines"`, `"3 months
   ago"`, `"since 2 weeks"` et équivalents → **texte laissé intact**).
   Vérifié en intégration réelle : `"The boy, born 05/12/2018, was
   seen 3 months ago."` → `"The <AGE_PEDIATRIC> boy, born
   <INFO_MASQUEE>, was seen 3 months ago."` (date de naissance
   masquée, durée écoulée conservée, âge généralisé).

**Justification méthodologique** (à faire valoir explicitement en
soutenance si la question est posée) : il s'agit d'une **minimisation
des données appliquée de façon proportionnée**; généraliser ce qui
est cliniquement nécessaire (âge en tranche, durée écoulée), supprimer
ce qui est identifiant (date de naissance exacte, nom, lieu); **pas**
une élimination indiscriminée de tout ce qui est marqué « sensible »
par le recognizer sous-jacent. L'art. 5 RGPD (minimisation) n'impose
pas de supprimer plus d'information que nécessaire ; un âge en tranche
ne permet pas de ré-identifier un patient mais reste indispensable au
triage clinique (ESI), alors qu'une date de naissance exacte est un
identifiant direct sans valeur clinique ajoutée par rapport à la
tranche d'âge.

### 7.5 Révision humaine persistée des candidats de PII résiduelle (09/09/2026)

**Problème identifié** : le §4 documente une relecture manuelle
réelle, mais conduite par un agent IA autonome, pas par une personne
(avertissement méthodologique explicite dès le §4, rappelé au §6).
Au-delà de ce POC initial, le mécanisme *lui-même* avait un défaut
structurel indépendant de qui relit : `CandidatPiiResiduelle.verdict
== "pendant_revision_humaine"` (cf. §2 méthode,
`E1_04_02_controler_qualite_anonymisation.py`) était un **cul-de-sac** :
ni le regex ni la seconde opinion spaCy ne tranchent seuls, et
**aucune variable ni fichier ne persistait jamais la décision d'une
personne**. Une revue humaine réelle, faite une fois, n'était nulle
part enregistrée; impossible de démontrer, au moment de la
soutenance, quels cas précis ont été validés par une personne et
lesquels restent réellement ouverts. Deux chantiers ferment cet écart
pour de bon, dans cet ordre (le second dépend du premier) :

**1. Muestreo incrémental du contrôle qualité.** Avant ce chantier,
chaque exécution de `E1_04_02_controler_qualite_anonymisation.py` tirait un
échantillon **aléatoire neuf** (`--taille-echantillon`, `--graine`),
confirmé dans le rapport généré lui-même, qui indiquait explicitement
« pas un cumul persistant entre exécutions ». Deux exécutions
successives pouvaient (re)relire des cas différents, sans jamais
garantir une couverture croissante. Corrigé en reprenant EXACTEMENT le
patron déjà établi par `AnonymiserDatasetUseCase`/`--limite` (fichier
de sortie séparé + `identifiants_existants()` pour déterminer « déjà
traité », cf. §1 de la Partie 1 de ce rapport) : un nouveau registre
persisté, `data/processed/controle_qualite_identifiants_echantillonnes.jsonl`
(`RegistreEchantillonsControleQualite` / `JsonlRegistreEchantillonsControleQualite`,
un enregistrement `{identifiant, stratum, horodatage}` par identifiant
échantillonné), exclut du tirage stratifié les identifiants déjà vus
lors d'une exécution précédente; appliqué indépendamment aux deux
strates (`principal` et `sans_entite`, cf. §7.3). Chaque exécution
tire donc `--taille-echantillon` identifiants **nouveaux**, jamais
revus, et le total échantillonné croît de façon monotone entre
exécutions. Ce registre est volontairement **séparé** du fichier de
décisions humaines ci-dessous (point 2) : il répond à une question
différente (« quel exemple a déjà été comparé ? ») que « quelle
décision a été prise sur tel candidat précis ? »; les deux
granularités (par exemple vs par candidat) ne se recouvrent pas et
fusionner les deux fichiers aurait mélangé deux schémas
d'enregistrement différents sans gain réel.

**2. Fichier persisté des décisions humaines + script de révision.**
Un candidat de PII résiduelle peut apparaître plusieurs fois avec le
même `type_motif` dans le même champ (confirmé en investigation :
jusqu'à 17 matches de `bigramme_capitalise` dans un seul champ
`chosen[0]` d'un exemple réel); une clé stable ne peut donc pas se
limiter à `(identifiant, champ, type_motif)`. Clé retenue,
`CleCandidatRevision` (`domain.model.decision_revision_humaine`) :
`(source_liste, identifiant, champ, type_motif, debut, fin)` :
`source_liste` distingue les 3 origines possibles d'un candidat
(`candidats_pii`, `candidats_faux_positifs`, `candidats_pii_sans_entite`,
cf. §7.3) et `debut`/`fin` (position du match, jusque-là calculés par
`detecter_candidats()`/`_extraire_fragments_masques()` mais jamais
persistés sur `CandidatPiiResiduelle`/`CandidatFauxPositifAnonymisation`)
désambiguïsent deux matches identiques du même type dans le même
champ; le `passage` seul ne suffit pas si le motif se répète
littéralement.

Chaque décision (JSONL, `data/processed/decisions_revision_humaine.jsonl`,
`JsonlDecisionsRevisionHumaine`) porte : la clé stable ci-dessus,
`decision` (`"accepte"` = confirmé humainement que ce n'est PAS une
PII réelle, `"rejete"` = confirmé qu'il s'agit d'une fuite réelle),
un horodatage et une note libre optionnelle. Contrairement à un
fichier d'audit purement append-only (`ajouter_exemples_jsonl`, réservé
aux cas où plusieurs entrées peuvent légitimement partager le même
identifiant), une décision peut être **corrigée**; le fichier est
donc géré comme `JsonlDatasetRepository.sauvegarder` (relecture
complète, fusion par clé, réécriture complète), un coût O(n) par
écriture largement acceptable ici car le volume de décisions humaines
est borné par la taille des échantillons de contrôle qualité (quelques
centaines/milliers), pas par la taille du corpus (147k+); la mise en
garde de ce fichier sur le coût O(n²) de `sauvegarder()` en boucle
concerne un tout autre ordre de grandeur.

`interfaces/cli/E1_04_01_reviser_pii_residuelle.py` expose deux modes :

- **`verify`** : recalcule (*replay* déterministe; mêmes textes,
  mêmes regex, même seconde opinion spaCy ⇒ mêmes candidats à chaque
  appel) TOUS les candidats `pendant_revision_humaine` des 3 sources,
  sur la **totalité** des identifiants déjà échantillonnés (les deux
  strates, toutes exécutions confondues via le registre du point 1 ;
  pas seulement le dernier lot, à cause précisément du muestreo
  incrémental), exclut ceux ayant déjà une décision, puis pour chacun
  des candidats restants : affiche identifiant/source/champ/type de
  motif et le passage (40 caractères de contexte de chaque côté, cf.
  `CONTEXTE_CARACTERES`), avec une option `v` pour afficher le champ
  **complet** (original et anonymisé) quand le contexte court ne
  suffit pas à juger. La décision (accepter/rejeter/sauter/quitter)
  est **persistée immédiatement** après chaque réponse; pas en fin de
  lot; pour qu'une interruption du terminal ne perde jamais le
  travail déjà fait.
- **`modify --identifiant ... [--champ ...]`** : localise une ou
  plusieurs décisions déjà prises et permet de corriger une erreur de
  saisie sans repasser par toute la liste en attente.

**Le rapport de `E1_04_02_controler_qualite_anonymisation.py` (§2 de ce
document, Markdown) relit ce fichier de décisions** pour annoter
chaque candidat `pendant_revision_humaine` qu'il liste avec son statut
réel : `accepte` / `rejete` / encore génuinement en attente; avec un
avertissement explicite que ce rapport ne décrit que le lot de SON
exécution (muestreo incrémental oblige), le statut **cumulé** vivant
dans le fichier de décisions lui-même, tenu à jour par
`E1_04_01_reviser_pii_residuelle.py`.

**Ce que ce mécanisme permet de dire en soutenance, et ce qu'il ne
permet toujours pas de dire** : il rend l'exigence NF2
(« anonymisation validée **manuellement**, 0 PII résiduelle sur
échantillon de contrôle ») vérifiable et traçable; chaque décision
humaine est horodatée, attribuable à un candidat précis, et
non-destructive (`modify` corrige sans perdre l'historique implicite
d'une exécution correcte : l'ancienne décision est simplement
remplacée). Il **ne rend pas** automatiquement vraie l'affirmation
« 0 PII résiduelle confirmée » tant que des candidats restent
`en_attente` dans `decisions_revision_humaine.jsonl`; ce mécanisme
fournit l'outil pour fermer cet écart, il ne le ferme pas tout seul :
quelqu'un doit encore exécuter `E1_04_01_reviser_pii_residuelle.py verify`
jusqu'à ce que la liste des candidats en attente soit vide. C'est la
même limite que celle déjà posée au §6 : ce document ne remplace pas
une revue humaine indépendante, il lui donne enfin un endroit où
laisser une trace.

## 8. Document suivant

`docs/03_etape2_sft/00_introduction_concepts.md` : planification du
fine-tuning supervisé (SFT + LoRA) sur le dataset anonymisé documenté
ici.
