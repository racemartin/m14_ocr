"""
Genere le rapport technique et recommandations strategiques (Livrable 3
du cahier des charges, docs/00_cadrage/01_cahier_des_charges.md SS7) :
methodologie, metriques de performance, analyse des resultats, roadmap
de passage a l'echelle. <= 20 pages.

Style visuel calque sur le rapport d'un projet precedent
(docs/05_etude/M13_MCP_Etude de faisabilite_V1.1.docx, fourni comme
gabarit) : page de titre + table des matieres en une colonne, corps en
deux colonnes, hierarchie de titres coloree (Heading 1/2 en 00B0F0,
Heading 3 en 333399), police de corps Optima 8pt, titre principal en
2D4F8E 18pt -- valeurs extraites reellement du gabarit via python-docx,
jamais devinees.

Toutes les metriques citees proviennent du README.md racine (journal
des runs DPO, tableau d'evaluation post-DPO, criteres d'acceptation du
cahier des charges SS9) : aucun chiffre invente.

Usage :
    uv run --with python-docx python docs/05_etude/generer_rapport_technique.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

RACINE = Path(__file__).resolve().parent.parent.parent
SORTIE = Path(__file__).resolve().parent / "M14_Rapport_technique_CHSA_Triage.docx"
DIAGRAMMES = RACINE / "docs" / "diagrams"

COULEUR_TITRE = RGBColor(0x2D, 0x4F, 0x8E)
COULEUR_H1 = RGBColor(0x00, 0xB0, 0xF0)
COULEUR_H2 = RGBColor(0x00, 0xB0, 0xF0)
COULEUR_H3 = RGBColor(0x33, 0x33, 0x99)
COULEUR_H4 = RGBColor(0x9A, 0x6B, 0x21)
POLICE_CORPS = "Optima"
POLICE_TITRES = "Arial"


def _definir_police_style(document: Document, nom: str, police: str, taille_pt: float,
                            couleur: RGBColor | None = None, gras: bool | None = None) -> None:
    style = document.styles[nom]
    style.font.name = police
    style.font.size = Pt(taille_pt)
    if couleur is not None:
        style.font.color.rgb = couleur
    if gras is not None:
        style.font.bold = gras
    # Force la police pour le script est-asiatique/complexe aussi (evite
    # un retour silencieux vers Calibri sur certains rendus Word).
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), police)


def _deux_colonnes(section) -> None:
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = sect_pr.makeelement(qn("w:cols"), {})
        sect_pr.append(cols)
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "425")


def _nouvelle_section_deux_colonnes(document: Document) -> None:
    document.add_section(WD_SECTION.CONTINUOUS)
    _deux_colonnes(document.sections[-1])


def _paragraphe(document: Document, texte: str, style: str | None = None,
                 alignement=None, espace_apres: float | None = None) -> "Paragraph":
    p = document.add_paragraph(texte, style=style)
    if alignement is not None:
        p.alignment = alignement
    if espace_apres is not None:
        p.paragraph_format.space_after = Pt(espace_apres)
    return p


def _figure(document: Document, chemin: Path, legende: str, largeur_cm: float = 8.5) -> None:
    if not chemin.exists():
        return
    document.add_picture(str(chemin), width=Cm(largeur_cm))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = document.add_paragraph(legende)
    cap.style = document.styles["Caption"] if "Caption" in [s.name for s in document.styles] else None
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(7)


def _tableau(document: Document, entetes: list[str], lignes: list[list[str]],
             legende: str | None = None) -> None:
    if legende:
        cap = document.add_paragraph(legende)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            run.italic = True
            run.font.size = Pt(7)
    table = document.add_table(rows=1, cols=len(entetes))
    table.style = "Light Grid Accent 1"
    for cell, texte in zip(table.rows[0].cells, entetes):
        cell.text = texte
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(7)
    for ligne in lignes:
        row = table.add_row()
        for cell, texte in zip(row.cells, ligne):
            cell.text = str(texte)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(7)
    document.add_paragraph()


def construire() -> None:
    d = Document()

    # ------------------------------------------------------------------
    # Styles (valeurs extraites du gabarit M13, cf. docstring du module)
    # ------------------------------------------------------------------
    _definir_police_style(d, "Normal", POLICE_CORPS, 8)
    _definir_police_style(d, "Title", POLICE_TITRES, 18, COULEUR_TITRE, True)
    _definir_police_style(d, "Heading 1", POLICE_TITRES, 12, COULEUR_H1, True)
    _definir_police_style(d, "Heading 2", POLICE_TITRES, 11, COULEUR_H2, True)
    _definir_police_style(d, "Heading 3", POLICE_TITRES, 10, COULEUR_H3, True)
    _definir_police_style(d, "Heading 4", POLICE_TITRES, 9, COULEUR_H4, True)

    section = d.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(1.4)
    section.bottom_margin = Cm(1.0)
    section.left_margin = Cm(1.1)
    section.right_margin = Cm(1.1)

    # ------------------------------------------------------------------
    # Page de titre (une colonne)
    # ------------------------------------------------------------------
    d.add_paragraph("CENTRE HOSPITALIER SAINT-AURÉLIEN (CHSA)", style="Normal")
    titre = d.add_paragraph("CHSA Triage", style="Title")
    titre.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _paragraphe(
        d,
        "Agent IA de triage médical — Rapport technique et "
        "recommandations stratégiques",
        style="Subtitle",
    )
    d.add_paragraph()
    _paragraphe(d, "Spécialisation d'un modèle de base par Fine-Tuning "
                   "Supervisé (SFT) et alignement par préférences (DPO) "
                   "pour le déploiement d'un LLM médical d'aide au "
                   "triage aux urgences.")
    d.add_paragraph()
    _paragraphe(d, "par\tRafael Cerezo Martín")
    _paragraphe(d, "AI Engineer — OpenClassrooms")
    _paragraphe(d, "Septembre 2026")
    d.add_paragraph()
    _figure(d, DIAGRAMMES / "00_vue_ensemble" / "vision_generale_etapes_v3_detaille.png",
            "Figure 1 — Vue d'ensemble du pipeline, des données brutes à l'endpoint déployé.",
            largeur_cm=16)

    # ------------------------------------------------------------------
    # Table des matieres (une colonne, champ TOC natif Word)
    # ------------------------------------------------------------------
    d.add_page_break()
    d.add_heading("Table des matières", level=1)
    paragraph = d.add_paragraph()
    run = paragraph.add_run()
    fld_char1 = run._r.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "begin"})
    instr = run._r.makeelement(qn("w:instrText"), {qn("xml:space"): "preserve"})
    instr.text = r'TOC \o "1-3" \h \z \u'
    fld_char2 = run._r.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "separate"})
    fld_char3 = run._r.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "end"})
    for el in (fld_char1, instr, fld_char2, fld_char3):
        run._r.append(el)
    _paragraphe(d, "(Mettre à jour les champs dans Word : Ctrl+A puis F9, "
                   "ou clic droit → « Mettre à jour les champs ».)")

    # ------------------------------------------------------------------
    # Corps du rapport (deux colonnes a partir d'ici)
    # ------------------------------------------------------------------
    d.add_page_break()
    _nouvelle_section_deux_colonnes(d)

    # === Introduction =====================================================
    d.add_heading("Introduction", level=1)
    _paragraphe(d, "Contexte et mission")
    _paragraphe(
        d,
        "Le Centre Hospitalier Saint-Aurélien (CHSA) souhaite évaluer "
        "l'apport d'un agent conversationnel fondé sur un grand modèle "
        "de langage (LLM) pour assister le personnel infirmier lors de "
        "l'accueil aux urgences : recueillir les symptômes d'un patient "
        "par un court entretien guidé, puis proposer un niveau de "
        "priorité de triage (échelle ESI, 1 à 5) à valider par un "
        "soignant humain. Ce rapport documente le travail réalisé dans "
        "le cadre de cette mission de type proof of concept (POC) : "
        "préparation des données, spécialisation du modèle par SFT puis "
        "DPO, et déploiement d'un endpoint d'inférence réel."
    )
    _paragraphe(
        d,
        "L'agent n'a volontairement aucun pouvoir de décision clinique "
        "autonome : il recueille l'information et propose une "
        "classification, mais chaque diagnostic reste soumis à "
        "validation par un infirmier avant toute action. Cette limite "
        "de périmètre, posée dès le départ, structure une grande partie "
        "des choix techniques décrits dans ce rapport (traçabilité "
        "systématique, garde-fous de sécurité, format structuré "
        "vérifiable)."
    )
    _paragraphe(d, "Contraintes techniques imposées")
    _paragraphe(
        d,
        "Le cahier des charges impose un modèle de base "
        "Qwen3-1.7B-Base spécialisé par LoRA (SFT puis DPO), un moteur "
        "d'inférence vLLM (PagedAttention), une API FastAPI "
        "conteneurisée Docker, une anonymisation par Microsoft Presidio "
        "et un pipeline CI/CD GitHub Actions. Ces choix ne sont pas "
        "discutés ici : ce rapport porte sur leur mise en œuvre et sur "
        "les résultats obtenus."
    )
    _paragraphe(d, "Architecture retenue")
    _paragraphe(
        d,
        "Le projet suit une architecture hexagonale (ports/adaptateurs) : "
        "le domaine métier (cas d'usage d'entretien et de diagnostic) "
        "ignore totalement la technologie d'inférence, de persistance ou "
        "d'anonymisation utilisée. Ce choix s'est révélé décisif pendant "
        "le déploiement (§4) : la même logique d'entretien fonctionne "
        "indifféremment contre un moteur local (llama.cpp), un moteur "
        "distant (vLLM) ou un double en mémoire dans les tests, sans "
        "aucune modification du code métier. Concrètement, le domaine "
        "définit un port `MoteurInference` (une interface abstraite : "
        "« générer une réponse à partir de messages ») dont trois "
        "adaptateurs distincts implémentent chacun un moyen technique "
        "différent d'y répondre — c'est ce découplage qui a permis, "
        "pendant l'incident de déploiement (§4.2), de continuer à "
        "développer et tester le reste du système sans dépendre en "
        "permanence d'un GPU réel disponible."
    )
    _figure(d, DIAGRAMMES / "01_environnement" / "paquets" / "architecture_hexagonale.png",
            "Figure 2 — Architecture hexagonale du projet : domaine, cas d'usage et adaptateurs.")

    d.add_heading("Concepts clés", level=2)
    _paragraphe(
        d,
        "Quatre notions techniques structurent ce rapport ; elles sont "
        "introduites brièvement ici pour un lecteur non spécialiste, "
        "puis reprises en contexte dans les sections correspondantes."
    )
    _paragraphe(
        d,
        "LoRA (Low-Rank Adaptation) — au lieu de réentraîner les "
        "milliards de paramètres d'un modèle de langage, LoRA insère de "
        "petites matrices supplémentaires de rang réduit dans certaines "
        "couches (typiquement l'attention) et n'entraîne que celles-ci. "
        "Le modèle de base reste inchangé et gelé ; l'adaptateur LoRA, "
        "lui, ne pèse que quelques dizaines de mégaoctets et peut être "
        "chargé ou déchargé à la volée."
    )
    _paragraphe(
        d,
        "SFT (Supervised Fine-Tuning) — spécialisation supervisée "
        "classique : le modèle apprend, exemple par exemple, à "
        "reproduire une réponse attendue à partir d'un prompt donné. "
        "C'est la première phase d'entraînement de ce projet (§2)."
    )
    _paragraphe(
        d,
        "DPO (Direct Preference Optimization) — seconde phase "
        "d'alignement, qui n'apprend plus à partir d'une seule réponse "
        "« vraie » mais à partir de paires (réponse préférée, réponse "
        "rejetée) : le modèle apprend à préférer la première à la "
        "seconde, ce qui permet d'orienter des qualités plus difficiles "
        "à spécifier par un simple exemple (concision, pertinence "
        "clinique, absence de dérive)."
    )
    _paragraphe(
        d,
        "vLLM et PagedAttention — moteur de service d'inférence "
        "imposé par le cahier des charges. PagedAttention gère la "
        "mémoire GPU du cache d'attention par pages (comme la mémoire "
        "virtuelle d'un système d'exploitation) plutôt que par bloc "
        "contigu réservé à l'avance, ce qui permet de servir plusieurs "
        "requêtes simultanées sans gaspillage de VRAM. vLLM expose une "
        "API compatible OpenAI et sait servir un adaptateur LoRA "
        "nativement, sans jamais le fusionner avec le modèle de base."
    )

    d.add_heading("Environnement de développement", level=1)
    _paragraphe(
        d,
        "Le projet distingue deux environnements de travail, chacun "
        "avec son propre jeu de dépendances Python (gérées par `uv`) : "
        "un Environnement A local (WSL2, sans GPU, 5 Go de RAM), "
        "utilisé pour la préparation des données et le développement "
        "courant, et un Environnement B distant (GPU à la demande via "
        "Hugging Face Jobs et Spaces Dev Mode), utilisé pour "
        "l'entraînement et l'évaluation coûteuse. Un script de "
        "vérification dédié à chacun (`check_env_local.py`, "
        "`check_env_gpu.py`, `check_env_remote_hf.py`) contrôle les "
        "prérequis avant de lancer une étape, plutôt que de découvrir un "
        "problème de configuration en cours de run facturé."
    )
    _paragraphe(
        d,
        "La dépendance `torch` est résolue différemment selon "
        "l'environnement actif (mécanisme `tool.uv.sources` de `uv`) : "
        "une version CPU légère pour l'Environnement A, la version CUDA "
        "complète pour l'Environnement B — un défaut d'origine (`torch` "
        "toujours résolu en version CUDA, même en local) a été corrigé "
        "pendant ce projet et documenté en section Dépannage du "
        "README."
    )
    _paragraphe(
        d,
        "La suite de tests automatisés (448 tests au moment de la "
        "rédaction, tous verts) couvre le domaine métier, les "
        "adaptateurs (avec des doubles en mémoire pour ceux qui "
        "nécessitent normalement un GPU ou un réseau réel) et la couche "
        "API. Elle s'exécute entièrement dans l'Environnement A, sans "
        "GPU, ce qui la rend gratuite à relancer aussi souvent que "
        "nécessaire — c'est cette même suite qui s'exécute dans le "
        "pipeline CI/CD (§4.3)."
    )

    # === 1. Préparation des données ======================================
    d.add_heading("1. Préparation des données", level=1)

    d.add_heading("1.1 Méthodologie", level=2)
    _paragraphe(
        d,
        "Quatre corpus publics ont été collectés et unifiés dans un "
        "schéma pivot commun (JSON, bilingue FR/EN) : MediQAl "
        "(questions/réponses médicales, déclinées en trois sous-formats "
        "OEQ, MCQU, MCQM), FrenchMedMCQA (QCM pharmacie/médecine), "
        "MedQuAD (questions/réponses médicales) et "
        "UltraMedical-Preference (paires choisies/rejetées, seule "
        "source directement exploitable pour le DPO sans reformulation). "
        "Chaque exemple pivot porte un identifiant déterministe "
        "(`chsa-<source>-<uuid>`), une source, une langue, un type "
        "(sft ou dpo) et, lorsqu'ils sont disponibles, des champs "
        "cliniques structurés (symptômes, antécédents, constantes "
        "vitales : pression artérielle, fréquence cardiaque, SpO2, "
        "fréquence respiratoire)."
    )
    _paragraphe(
        d,
        "La fusion des quatre sources en un pivot unique déduplique par "
        "identifiant déterministe : sur l'ensemble des exemples "
        "collectés, 12 321 doublons exacts ont été détectés et écartés "
        "(archivés séparément plutôt que simplement supprimés, pour "
        "traçabilité)."
    )
    _figure(d, DIAGRAMMES / "02_etape1_donnees" / "sequence" / "construction_dataset_pivot.png",
            "Figure 3 — Construction du dataset pivot : fusion et déduplication des quatre sources.")

    d.add_heading("1.2 Anonymisation et contrôle qualité", level=2)
    _paragraphe(
        d,
        "L'anonymisation (Microsoft Presidio, imposée au cahier des "
        "charges) retire les identifiants directs (noms, dates, "
        "identifiants de dossier, coordonnées) avant toute publication "
        "sur le Hub Hugging Face. Presidio fonctionne par détection "
        "d'entités (reconnaisseurs dédiés par type de PII, combinables "
        "avec des règles spécifiques au domaine médical) suivie d'une "
        "stratégie de remplacement configurable ; une révision manuelle "
        "de la PII résiduelle et un contrôle qualité systématique sur un "
        "échantillon stratifié complètent la détection automatique, "
        "qui n'est jamais supposée parfaite à elle seule sur un texte "
        "clinique libre."
    )
    _figure(d, DIAGRAMMES / "02_etape1_donnees" / "sequence" / "anonymiser_dataset.png",
            "Figure 4 — Séquence d'anonymisation : détection Presidio, remplacement, contrôle qualité.")
    _paragraphe(
        d,
        "L'anonymisation complète a porté sur 134 883 exemples (pivot "
        "dédupliqué). Le découpage final, stratifié par source et "
        "vérifié automatiquement (écart maximal toléré entre strates), "
        "répartit ces exemples en 37 802 exemples éligibles SFT et "
        "97 081 exemples éligibles DPO."
    )
    _figure(d, DIAGRAMMES / "02_etape1_donnees" / "activite" / "pipeline_donnees.png",
            "Figure 5 — Pipeline complet de préparation des données : collecte, pivot, anonymisation, splits.")

    d.add_heading("1.3 Résultat", level=2)
    _tableau(
        d,
        ["Étape", "Volume"],
        [
            ["Exemples collectés (4 corpus bruts)", "134 883 + 12 321 doublons écartés"],
            ["Pivot anonymisé (après déduplication)", "134 883 exemples"],
            ["Split éligible SFT", "37 802 exemples"],
            ["Split éligible DPO", "97 081 exemples"],
            ["Sous-ensemble publié SFT (Livrable 1)", "5 000 exemples"],
            ["Sous-ensemble publié DPO (Livrable 1)", "5 000 exemples"],
            ["Jeu de test clinique (évaluation)", "278 exemples"],
        ],
        legende="Tableau 1 — Volumes réels du jeu de données à chaque étape (README §1).",
    )
    _paragraphe(
        d,
        "Le cahier des charges (Livrable 1) demandant un jeu "
        "d'entraînement de l'ordre de 5000 paires, un sous-ensemble "
        "filtré et stratifié de cette taille est extrait de chaque "
        "split (SFT et DPO) pour publication et entraînement réel — le "
        "pivot complet (134 883 exemples) reste disponible pour "
        "d'éventuels travaux futurs à plus grande échelle. L'ensemble "
        "est publié et versionné sur le Hub Hugging Face (dépôts privés "
        "dédiés à chaque étape), bilingue, anonymisé et prêt pour "
        "l'entraînement."
    )

    # === 2. SFT + LoRA ====================================================
    d.add_heading("2. SFT + LoRA", level=1)

    d.add_heading("2.1 Méthodologie", level=2)
    _paragraphe(
        d,
        "Le modèle de base Qwen3-1.7B-Base est spécialisé par "
        "fine-tuning supervisé (SFT) avec un adaptateur LoRA plutôt "
        "qu'un fine-tuning complet : seule une fraction des paramètres "
        "est entraînée, ce qui réduit drastiquement le coût mémoire et "
        "de calcul tout en conservant les capacités générales du modèle "
        "de base — un compromis adapté à un POC avec budget de calcul "
        "contraint. Le chargement s'effectue en quantification 4-bit "
        "(`BitsAndBytesConfig`) pour l'entraînement sur une seule GPU "
        "distante (HF Jobs)."
    )
    _tableau(
        d,
        ["Paramètre", "Valeur", "Rôle"],
        [
            ["Quantification", "nf4, double, bf16", "Divise par 4 la VRAM du modèle base"],
            ["Rang LoRA (r)", "16", "Capacité de l'adaptateur"],
            ["Alpha LoRA", "32 (2r)", "Échelle de la mise à jour, règle standard"],
            ["Dropout LoRA", "0,05", "Régularisation anti-surapprentissage"],
            ["Modules cibles", "q, k, v, o_proj", "Attention seulement, adaptateur léger"],
            ["Taux d'apprentissage", "2·10⁻⁴", "Valeur typique LoRA"],
            ["Époques", "3", "Compromis apprentissage/mémorisation"],
            ["Taille de lot", "4", "Limité par la VRAM disponible"],
            ["Packing", "activé", "GPU utilisé à ~100% (vs 40-60% en padding classique)"],
        ],
        legende="Tableau 2 — Configuration réelle de l'entraînement SFT-LoRA "
                "(recipes/sft_qwen3_lora.yaml, README « Explication du contenu de la recette »).",
    )
    _paragraphe(
        d,
        "Le type de quantification `nf4` (« NormalFloat 4-bit ») est un "
        "format numérique optimisé pour des poids à distribution "
        "approximativement gaussienne, plus précis à taille égale que "
        "le format `fp4` générique — à ne pas confondre avec le "
        "garde-fou de sécurité clinique NF4 discuté en §5, homonyme "
        "sans lien technique."
    )
    _paragraphe(
        d,
        "L'évaluation baseline zero-shot (modèle non spécialisé) sert "
        "de point de comparaison avant tout entraînement, sur le même "
        "jeu de test clinique (278 exemples) que les évaluations "
        "post-SFT et post-DPO."
    )
    _figure(d, DIAGRAMMES / "03_etape2_sft" / "activite" / "pipeline_sft_lora.png",
            "Figure 6 — Pipeline SFT+LoRA : baseline, entraînement, évaluation post-SFT.")

    d.add_heading("2.2 Résultat", level=2)
    _paragraphe(
        d,
        "Le premier entraînement réel a été mené à son terme sur GPU "
        "L4 en environ 20 minutes (342 pas, 3 époques), verdict de "
        "convergence SAINE. Le checkpoint SFT-LoRA atteint un F1 "
        "(token) d'environ 0,112 sur le jeu de test clinique — la "
        "valeur de référence à laquelle le DPO est ensuite comparé "
        "(§3). C'est ce checkpoint qui sert de politique de référence "
        "(π_ref) pour l'alignement par préférences, et ses poids sont "
        "publiés durablement sur le Hub (dépôt privé dédié)."
    )

    # === 3. DPO ============================================================
    d.add_heading("3. DPO (Direct Preference Optimization)", level=1)

    d.add_heading("3.1 Méthodologie", level=2)
    _paragraphe(
        d,
        "Le DPO aligne le modèle sur des paires de préférence "
        "(réponse choisie / réponse rejetée) sans modèle de récompense "
        "séparé : la perte compare directement, sous forme de "
        "log-vraisemblance, la politique en cours d'entraînement à la "
        "politique de référence figée (le checkpoint SFT). "
        "L'hyperparamètre β contrôle la force de cet ancrage à la "
        "référence : plus β est faible, plus le modèle est autorisé à "
        "s'écarter de π_ref, au risque d'une dérive de génération non "
        "capturée par la seule perte d'entraînement (§3.2)."
    )
    _figure(d, DIAGRAMMES / "04_etape3_dpo" / "activite" / "dpo_double_fonction_entrainement.png",
            "Figure 7 — Boucle d'entraînement DPO : double passage (politique + référence figée).")
    _figure(d, DIAGRAMMES / "04_etape3_dpo" / "sequence" / "entrainer_dpo.png",
            "Figure 8 — Séquence d'entraînement DPO, du dataset de préférence au checkpoint publié.")

    d.add_heading("3.2 Métriques et itérations réelles", level=2)
    _paragraphe(
        d,
        "La mise au point de β a nécessité plusieurs itérations "
        "réelles, chacune diagnostiquée sur des métriques concrètes "
        "(rewards/accuracies, rewards/margins) plutôt que sur la seule "
        "perte d'entraînement brute :"
    )
    _tableau(
        d,
        ["Run", "β", "Taille", "Verdict", "rewards/accuracies", "rewards/margins"],
        [
            ["1 (batch=1)", "0,1", "100", "sous-apprentissage", "0,30", "-1,25"],
            ["2 (lr relevé)", "0,1", "100", "sous-apprentissage*", "0,625 / 0,75", "+0,72 / +0,60"],
            ["3", "0,1", "5000", "saine", "0,725", "+3,00"],
            ["4", "0,3", "100", "sous-apprentissage*", "0,60", "1,93"],
            ["5 (final)", "0,3", "5000", "saine", "0,75", "4,84"],
        ],
        legende="Tableau 3 — Journal des runs DPO (extrait, README §3.2). "
                "*Verdict basé sur un seuil de perte hérité du SFT, non "
                "recalibré pour DPO ; rewards/accuracies et margins "
                "indiquent un apprentissage réel malgré ce verdict.",
    )
    _paragraphe(
        d,
        "Le run à β=0,1 sur 5000 exemples atteint un rewards/margins "
        "anormalement élevé (+3,00) : signe d'un ancrage trop faible à "
        "π_ref. L'évaluation post-DPO correspondante révèle une "
        "dégénérescence de génération réelle (changements de langue "
        "aléatoires FR/EN vers JA/ZH/AR, effondrements en répétitions) "
        "non visible dans la perte d'entraînement seule. Relever β à "
        "0,3 corrige ce défaut, confirmé sur les deux échelles (100 "
        "puis 5000 exemples, aucune régression)."
    )

    d.add_heading("3.3 Évaluation post-DPO", level=2)
    _tableau(
        d,
        ["β (échelle)", "repetition_penalty", "F1 (token)", "Latence"],
        [
            ["0,1", "—", "0,049", "~12,0 s"],
            ["0,1", "1,2", "0,063", "~11,9 s"],
            ["0,3 (100)", "1,2", "0,112", "~10,7 s"],
            ["0,3 (5000, final)", "1,2", "0,110", "~11,2 s"],
        ],
        legende="Tableau 4 — Évaluation post-DPO (checkpoint "
                "mombasstic/chsa-triage-dpo-lora, README §3.3). β=0,3 + "
                "repetition_penalty=1,2 ramène le F1 au niveau du "
                "post-SFT (0,112), sans régression au passage à "
                "l'échelle (100 → 5000 exemples).",
    )
    _paragraphe(
        d,
        "Hyperparamètres finaux retenus : β=0,3, taux d'apprentissage "
        "5·10⁻⁵, taille de lot 4, rang LoRA 16 "
        "(recipes/dpo_qwen3_lora.yaml). Le checkpoint DPO n'est jamais "
        "fusionné avec la base : il est servi nativement par vLLM comme "
        "adaptateur LoRA (§4), ce qui permet de revenir au modèle de "
        "base à tout moment sans réentraînement, et de comparer "
        "facilement plusieurs versions du garde-fou d'alignement sans "
        "dupliquer les 3,2 Go de poids du modèle de base."
    )

    d.add_heading("3.4 Portée de la reformulation des préférences", level=2)
    _paragraphe(
        d,
        "Les paires de préférence brutes d'UltraMedical-Preference ne "
        "portent pas nativement le format de sortie structuré ciblé par "
        "ce projet (niveau ESI, catégorie, ressources) : un cas d'usage "
        "dédié reformule chaque paire choisie/rejetée dans ce format "
        "avant l'entraînement DPO proprement dit. Ce découplage "
        "reformulation/DPO — les réponses choisies ne sont pas "
        "reformulées, seul le prompt système change — reste une cause "
        "probable, distincte de la dégénérescence de fluidité (§3.2), "
        "de la perte partielle du format JSON constatée à "
        "l'évaluation : la mission ne demande que l'alignement SFT+DPO "
        "sur les préférences, le format JSON strict étant un ajout du "
        "cahier des charges local visé plutôt au niveau du prompting de "
        "l'endpoint déployé (§5, §6.1)."
    )

    # === 4. Déploiement ====================================================
    d.add_heading("4. Déploiement", level=1)

    d.add_heading("4.1 Architecture retenue", level=2)
    _paragraphe(
        d,
        "Décision produit prise pendant le déploiement : une "
        "architecture à deux pièces plutôt que trois. Un unique Space "
        "Hugging Face Docker/GPU héberge, dans le même conteneur, "
        "l'API FastAPI et le serveur vLLM (un seul coût GPU, un seul "
        "on/off) ; le frontend Streamlit de test tourne en local sur le "
        "poste de l'opérateur, car il ne fait qu'appeler l'API en HTTP "
        "sans aucun calcul GPU — un Space dédié uniquement pour cette "
        "interface aurait exigé un abonnement payant sans bénéfice "
        "réel. Le frontend ne sait jamais à l'avance si le modèle est "
        "chargé : il sonde en boucle un endpoint de santé applicatif "
        "qui répond toujours HTTP 200, y compris pendant les "
        "plusieurs minutes de chargement du modèle, avec un corps "
        "structuré distinguant explicitement « serveur d'inférence "
        "indisponible » d'une véritable erreur — un simple health-check "
        "Docker qui redémarrerait le conteneur sur ce délai normal "
        "aurait été contre-productif."
    )
    _figure(d, DIAGRAMMES / "05_etape4_deploiement" / "deploiement" / "deploiement_etape4.png",
            "Figure 9 — Architecture de déploiement réelle : Space Docker/GPU (API+vLLM), "
            "frontend local, journal d'audit persistant.")
    _figure(d, DIAGRAMMES / "05_etape4_deploiement" / "sequence" / "poursuivre_entretien.png",
            "Figure 10 — Séquence d'un tour d'entretien : de la requête infirmier à la "
            "réponse du modèle, en passant par le journal d'audit.")

    d.add_heading("4.2 Incident technique majeur et résolution", level=2)
    _paragraphe(
        d,
        "Le premier déploiement réel s'est heurté à un segfault natif "
        "(crash silencieux du processus, sans message Python "
        "exploitable) systématique de vLLM, survenant à chaque fois "
        "exactement au même point : juste après le chargement des "
        "poids du modèle, pendant le profilage interne que vLLM "
        "effectue pour dimensionner son cache de mémoire GPU, sur "
        "l'instance GPU réelle (NVIDIA L4). Huit pistes ciblées ont été "
        "testées methodiquement en conditions réelles, chacune "
        "produisant le même échec identique malgré le changement — la "
        "liste complète, dans l'ordre où elles ont été écartées :"
    )
    _tableau(
        d,
        ["#", "Piste testée", "Résultat"],
        [
            ["1", "Désactivation du multiprocessing interne de vLLM "
                   "(variable d'environnement dédiée)", "Sans effet — "
             "cette variable ne s'applique qu'à l'usage programmatique "
             "de vLLM, jamais au serveur lancé en ligne de commande"],
            ["2", "Exécution du conteneur sous un utilisateur non "
                   "root explicite (au lieu de root par défaut)",
             "Correction légitime conservée, mais sans effet sur le crash"],
            ["3", "Adaptateur LoRA désactivé entièrement", "Même crash "
             "identique, sans le module de chargement LoRA en cause"],
            ["4", "Moteur d'attention alternatif (FlashInfer au lieu "
                   "du moteur par défaut)", "Même crash identique"],
            ["5", "Ordonnancement asynchrone désactivé", "Même crash identique"],
            ["6", "Délai fixe de démarrage avant le lancement de vLLM",
             "Même crash identique — écarte un GPU/pilote pas encore stabilisé"],
            ["7", "Version de vLLM différente (deux versions testées)",
             "Même crash identique sur les deux versions"],
            ["8", "Désaccord pilote NVIDIA / version CUDA (vérifié "
                   "directement via l'outil de diagnostic du pilote)",
             "Pilote et CUDA confirmés sains, écarté"],
        ],
        legende="Tableau 5 — Les huit pistes de diagnostic testées et écartées pour le "
                "segfault de démarrage vLLM (issue publique vllm-project/vllm#58616).",
    )
    _paragraphe(
        d,
        "Un test décisif a permis de caractériser la nature du "
        "problème sans le résoudre directement : activer une trace "
        "complète de chaque appel de fonction Python (diagnostic "
        "officiel de vLLM, qui ralentit l'exécution de plus de 100 "
        "fois) a permis, à lui seul, un démarrage complet et sain. Ce "
        "ralentissement massif faisant disparaître le crash est la "
        "signature caractéristique d'une condition de course "
        "dépendante du temps réel d'exécution — un défaut jamais isolé "
        "à une seule cause identifiable dans le code du projet ou de "
        "vLLM lui-même."
    )
    _paragraphe(
        d,
        "La résolution effective est venue d'un changement de "
        "stratégie plutôt que d'un neuvième correctif ciblé : "
        "reconstruire l'image du Space à partir de l'image Docker "
        "officielle vllm/vllm-openai (construite et testée par "
        "l'équipe vLLM elle-même, avec son propre environnement Python "
        "cohérent) plutôt que d'installer vLLM manuellement dans une "
        "image CUDA générique. Le problème a disparu entièrement, sans "
        "qu'une cause racine précise n'ait pu être établie côté vLLM. "
        "Cet incident, sa reproduction complète et sa résolution ont "
        "été signalés en amont au projet vLLM (issue publique "
        "vllm-project/vllm#58616) pour bénéficier à d'autres équipes "
        "confrontées au même symptôme."
    )
    _paragraphe(
        d,
        "Deux bugs applicatifs supplémentaires ont ensuite été trouvés "
        "et corrigés lors de la toute première conversation réelle "
        "contre ce vLLM enfin fonctionnel — un rappel concret que la "
        "disponibilité du serveur d'inférence n'implique pas à elle "
        "seule la correction du comportement applicatif :"
    )
    _paragraphe(
        d,
        "(1) Le paramètre de nombre maximal de tokens générés utilisait "
        "en interne le vocabulaire du moteur local llama.cpp "
        "(`n_predict`), jamais traduit vers le nom réellement attendu "
        "par l'API compatible OpenAI de vLLM (`max_tokens`) — un "
        "adaptateur voisin du projet effectuait déjà cette traduction "
        "pour un troisième moteur, mais elle avait été omise pour "
        "l'adaptateur vLLM, resté non testé en conditions réelles "
        "jusqu'à la résolution du segfault. Le champ inconnu était "
        "ignoré silencieusement par vLLM, laissant la génération partir "
        "jusqu'à la limite par défaut du modèle (2048 tokens) sans "
        "aucune limite de tour de conversation."
    )
    _paragraphe(
        d,
        "(2) Aucune température de génération n'était fixée "
        "explicitement dans les cas d'usage d'entretien et de "
        "diagnostic, laissant vLLM retomber sur un échantillonnage "
        "aléatoire non validé — alors que l'évaluation post-DPO qui "
        "avait validé la pénalité de répétition (§3.3) utilisait "
        "systématiquement une génération déterministe. Sans "
        "température explicite, le symptôme observé (réponses "
        "incohérentes, mélange spontané de langues au sein d'une même "
        "réponse) s'est reproduit à l'identique sur deux générations de "
        "GPU différentes (L4 en bfloat16, T4 en float16 par repli "
        "automatique), ce qui a permis d'écarter une cause matérielle "
        "avant de trouver la cause réelle. Les deux corrections ont "
        "rétabli des réponses cohérentes en conditions réelles, "
        "vérifiées par une conversation complète de bout en bout."
    )

    d.add_heading("4.3 CI/CD et endpoint réel", level=2)
    _paragraphe(
        d,
        "Le pipeline GitHub Actions (imposé au cahier des charges) "
        "comprend deux jobs : l'exécution de la suite de tests "
        "automatisés (448 tests, 0 échec, aucune dépendance GPU) et la "
        "vérification que l'image Docker de l'API se construit "
        "correctement, à chaque push sur la branche principale. "
        "L'endpoint réel (mombasstic/chsa-triage-api) répond de façon "
        "opérationnelle : chargement du modèle et de l'adaptateur LoRA "
        "DPO, healthcheck applicatif ne renvoyant jamais d'erreur "
        "brute, latence de réponse observée de l'ordre de 5 à 30 "
        "secondes par tour d'entretien selon le matériel GPU alloué (L4 "
        "ou T4, en fonction de la disponibilité côté fournisseur) — "
        "cette latence n'a pas encore fait l'objet d'une mesure "
        "statistique systématique sur un échantillon représentatif "
        "(recommandation §6.1)."
    )
    _figure(d, DIAGRAMMES / "05_etape4_deploiement" / "activite" / "pipeline_ci_cd.png",
            "Figure 11 — Pipeline CI/CD : tests automatisés + vérification du build Docker.")

    d.add_heading("4.4 Traçabilité et sécurité applicative", level=2)
    _paragraphe(
        d,
        "Chaque tour d'entretien et chaque demande de diagnostic est "
        "consigné dans un journal d'audit horodaté (conversation, "
        "entrée, sortie, version du modèle), garde-fou de traçabilité "
        "imposé au projet — persisté gratuitement dans un dataset privé "
        "Hugging Face via un mécanisme de publication périodique en "
        "arrière-plan, déjà utilisé ailleurs dans le projet pour le "
        "suivi d'expérimentation d'entraînement. Ce choix évite un "
        "abonnement de stockage persistant payant tout en garantissant "
        "que le journal survit aux redémarrages du conteneur, "
        "contrairement à une simple écriture sur le disque éphémère du "
        "Space, qui aurait perdu tout l'historique à chaque "
        "reconstruction de l'image (redémarrages nombreux pendant "
        "l'incident décrit en §4.2)."
    )
    _figure(d, DIAGRAMMES / "05_etape4_deploiement" / "sequence" / "obtenir_diagnostic.png",
            "Figure 12 — Séquence de demande de diagnostic : de l'historique d'entretien "
            "à la classification ESI proposée.")
    _paragraphe(
        d,
        "L'accès à l'API est protégé par une clé (en-tête `X-API-Key`), "
        "vérifiée par une dépendance FastAPI sur chaque route sensible, "
        "distincte du jeton d'accès Hugging Face lui-même utilisé pour "
        "les opérations internes (téléchargement de modèle, publication "
        "du journal d'audit). Le stockage du jeton Hugging Face a "
        "lui-même donné lieu à un incident réel pendant ce déploiement "
        "(formulaire web du Space rejetant un caractère du jeton, "
        "contournable en le déposant directement via l'API "
        "programmatique plutôt que le formulaire), documenté en détail "
        "dans le README du projet."
    )

    # === 5. Sécurité, conformité et vie privée ============================
    d.add_heading("5. Sécurité, conformité et vie privée", level=1)
    _paragraphe(
        d,
        "Le projet distingue explicitement deux notions homonymes qui "
        "portent toutes deux le sigle « NF4 » dans la documentation du "
        "projet, à ne pas confondre : le format de quantification "
        "`nf4` (NormalFloat 4-bit, §2.1, un détail d'implémentation "
        "de l'entraînement, déjà en place) et le garde-fou de sécurité "
        "clinique NF4 du cahier des charges (un juge automatique "
        "censé rejeter toute réponse dont le score de sécurité serait "
        "inférieur à 4 sur 7, non encore conçu ni implémenté — voir "
        "statut réel en §6)."
    )
    _paragraphe(
        d,
        "La protection des données personnelles s'appuie sur trois "
        "mécanismes complémentaires, à des étapes différentes du "
        "cycle de vie de la donnée : l'anonymisation Presidio en amont "
        "de tout entraînement (§1.2, retire les identifiants directs "
        "avant publication) ; le contrôle d'accès applicatif par clé "
        "d'API sur l'endpoint déployé (§4.4) ; et le journal d'audit, "
        "qui ne consigne que les échanges cliniques de l'entretien "
        "(jamais d'identité de patient, celle-ci restant hors du "
        "périmètre technique de l'agent, dont le rôle se limite à "
        "l'entretien symptomatique)."
    )
    _paragraphe(
        d,
        "Le magasin de conversations actif (état d'une conversation en "
        "cours) est délibérément gardé hors de l'architecture "
        "hexagonale ports/adaptateurs : c'est un état de session "
        "éphémère du processus, en mémoire, jamais persisté ni "
        "consigné dans le journal d'audit lui-même — seuls les tours "
        "d'entretien et les diagnostics le sont. Cette conception a une "
        "conséquence directe pour le passage à l'échelle, discutée "
        "en §6.3 : elle est incompatible avec plusieurs instances de "
        "l'API sans état partagé externe."
    )

    # === 6. Analyse des resultats vs criteres d'acceptation ==============
    d.add_heading("6. Analyse des résultats", level=1)
    _paragraphe(
        d,
        "Statut réel de chaque critère d'acceptation du POC "
        "(cahier des charges §9), à date :"
    )
    _tableau(
        d,
        ["Critère", "Statut"],
        [
            ["Endpoint vLLM opérationnel, latence documentée", "Atteint (§4.3)"],
            ["Pipeline CI/CD exécute les tests à chaque push", "Atteint (448 tests, 0 échec)"],
            ["JSON de diagnostic valide sur ≥95% des requêtes", "Non atteint — format non "
             "toujours respecté en sortie réelle de vLLM (voir §7)"],
            ["Accuracy ESI > baseline zéro-shot, mesurable", "Non mesurable tant que le "
             "format JSON n'est pas fiable"],
            ["Aucune réponse < 4/7 en sécurité (garde-fou NF4)", "Non implémenté — juge de "
             "sécurité clinique non encore conçu"],
        ],
        legende="Tableau 6 — Statut réel des critères d'acceptation du POC (cahier des charges §9).",
    )
    _paragraphe(
        d,
        "Le F1 (0,110–0,112) reste modeste en valeur absolue, mais "
        "l'analyse comparative (Tableau 2) montre que la variable "
        "décisive n'était pas la seule qualité de l'alignement DPO : "
        "un paramétrage d'inférence incomplet (§4.2) suffisait à lui "
        "seul à produire des réponses inexploitables, indépendamment de "
        "la qualité réelle du modèle. Ce constat — qu'un modèle "
        "correctement aligné peut sembler défaillant à cause d'un "
        "maillon d'infrastructure non validé — est probablement le "
        "résultat méthodologique le plus transférable de ce POC pour la "
        "suite du projet : évaluer un modèle uniquement hors ligne, "
        "sans jamais l'exercer contre le serveur d'inférence réellement "
        "visé pour la production, laisse une classe entière de défauts "
        "invisible jusqu'au déploiement."
    )
    _paragraphe(
        d,
        "Sur le plan de l'ingénierie du projet, deux disciplines se "
        "sont révélées particulièrement rentables pendant cette "
        "mission : ne jamais accepter une affirmation sans preuve "
        "directement vérifiable (logs réels, code source lu plutôt que "
        "supposé, tests exécutés plutôt qu'estimés), et traiter chaque "
        "correctif de déploiement comme une hypothèse à valider "
        "explicitement plutôt qu'une certitude — la moitié des huit "
        "pistes du Tableau 5, par exemple, semblaient plausibles a "
        "priori et ont dû être écartées une à une par des tests réels, "
        "pas par intuition."
    )

    # === 7. Roadmap ========================================================
    d.add_heading("7. Roadmap et recommandations pour le passage à l'échelle", level=1)
    _paragraphe(
        d,
        "Les recommandations suivantes sont organisées par horizon, "
        "et découlent directement des écarts réels constatés pendant "
        "cette mission (§6) plutôt que de bonnes pratiques génériques "
        "de mise en production de LLM."
    )

    d.add_heading("7.1 Court terme (avant toute extension de périmètre)", level=2)
    _paragraphe(
        d,
        "Contraindre la sortie du diagnostic à un schéma JSON strict "
        "au niveau du serveur d'inférence (grammaire de génération "
        "contrainte, disponible nativement côté vLLM et déjà exposée "
        "par sa configuration de sorties structurées) plutôt que par le "
        "seul prompt — c'est l'écart le plus bloquant identifié (§6) et "
        "le plus directement actionnable : il conditionne à la fois le "
        "critère d'acceptation JSON et la mesure d'accuracy ESI, "
        "aujourd'hui non mesurable pour cette seule raison."
    )
    _paragraphe(
        d,
        "Concevoir et implémenter le juge de sécurité clinique "
        "(garde-fou NF4), aujourd'hui explicitement en attente d'une "
        "décision produit : quel modèle ou quelle règle sert de juge, "
        "sur quel échantillon il est validé lui-même, et comment son "
        "propre taux d'erreur est mesuré avant d'en faire un filtre de "
        "sécurité sur lequel s'appuyer."
    )
    _paragraphe(
        d,
        "Mesurer la latence de façon statistique (percentiles p50/p95 "
        "sur un échantillon représentatif de conversations réelles, "
        "par type de matériel GPU alloué), pas seulement de façon "
        "ponctuelle comme actuellement — le journal d'audit déjà "
        "persisté (§4.4) porte l'horodatage nécessaire pour cette "
        "mesure sans instrumentation supplémentaire."
    )

    d.add_heading("7.2 Moyen terme (fiabilisation avant extension d'usage)", level=2)
    _paragraphe(
        d,
        "Ajouter une supervision applicative minimale : alertes sur "
        "les réponses hors format JSON ou en échec, tableau de bord de "
        "latence réelle construit directement à partir du journal "
        "d'audit déjà persisté (aucune nouvelle télémétrie à mettre en "
        "place, seulement à exploiter ce qui existe déjà). Étendre le "
        "jeu de test clinique (278 exemples aujourd'hui) au-delà du "
        "périmètre actuel pour fiabiliser statistiquement la mesure "
        "d'accuracy ESI une fois le format JSON corrigé."
    )
    _paragraphe(
        d,
        "Documenter et tester une procédure de restauration du "
        "service en cas d'indisponibilité prolongée de l'infrastructure "
        "GPU sous-jacente — un point de fragilité concret rencontré à "
        "plusieurs reprises pendant ce déploiement (échecs "
        "d'initialisation de conteneur imputables à la plateforme "
        "d'hébergement elle-même, résolus par un changement de "
        "configuration matérielle plutôt que par une action côté "
        "projet), donc indépendant du code du projet mais bien réel "
        "en exploitation."
    )
    _paragraphe(
        d,
        "Ajouter une interface de reprise explicite (« nouveau "
        "patient ») dans le frontend de test, déjà identifiée et "
        "corrigée pendant ce projet — sans elle, un opérateur "
        "poursuivant la saisie après un diagnostic mélangeait "
        "silencieusement le patient suivant avec l'historique du "
        "précédent, un risque d'erreur clinique par confusion "
        "d'identité d'entretien qui mérite une vigilance équivalente "
        "dans toute interface future."
    )

    d.add_heading("7.3 Long terme (déploiement à l'échelle du CHSA)", level=2)
    _paragraphe(
        d,
        "Avant toute mise en production clinique réelle : validation "
        "médicale formelle des diagnostics proposés par un comité "
        "clinique du CHSA sur un échantillon représentatif, étude "
        "d'impact réglementaire (le logiciel relève potentiellement du "
        "cadre des dispositifs médicaux logiciels selon l'usage final "
        "retenu), et mécanisme de retour d'expérience structuré depuis "
        "le terrain vers une boucle de réentraînement périodique du "
        "modèle plutôt qu'un modèle figé une fois pour toutes."
    )
    _paragraphe(
        d,
        "Sur le plan technique, prévoir une architecture à plusieurs "
        "instances de l'API avant toute charge réelle : le magasin de "
        "conversations actuel est volontairement en mémoire (§5), donc "
        "incompatible avec plusieurs instances sans état partagé "
        "externe (base de données ou cache partagé). Dimensionner le "
        "budget GPU sur la charge réelle observée en usage clinique "
        "plutôt que sur l'usage ponctuel de ce POC — le choix du "
        "matériel (L4 ou équivalent) devra être revalidé une fois un "
        "volume de requêtes concurrentes réaliste connu, et non plus "
        "supposé."
    )
    _paragraphe(
        d,
        "Enfin, conserver la discipline de traçabilité déjà en place "
        "(garde-fou F6) comme fondation de toute extension future plutôt "
        "que comme une case à cocher : c'est ce même journal d'audit "
        "qui permettra, à l'échelle, de mesurer objectivement l'impact "
        "clinique réel de l'agent plutôt que ses seules métriques "
        "d'entraînement hors ligne."
    )

    # === Conclusion ========================================================
    d.add_heading("Conclusion", level=1)
    _paragraphe(
        d,
        "Ce POC démontre la faisabilité technique d'un agent de "
        "triage médical spécialisé par SFT+DPO et servi par un endpoint "
        "vLLM réel, avec une architecture hexagonale qui a concrètement "
        "facilité le développement et le déploiement (bascule entre "
        "trois moteurs d'inférence — local, distant, double de test — "
        "sans changement du domaine métier). Chaque étape du pipeline "
        "(données, SFT, DPO, déploiement) a produit des artefacts "
        "réels et vérifiables : jeux de données publiés et versionnés, "
        "checkpoints LoRA hébergés sur le Hub, endpoint opérationnel "
        "avec pipeline CI/CD vert, journal d'audit persistant."
    )
    _paragraphe(
        d,
        "Le principal apprentissage de la phase de déploiement est que "
        "la validation d'un modèle aligné ne peut pas se limiter à son "
        "évaluation hors ligne : deux défauts d'intégration invisibles "
        "en évaluation (traduction de paramètres vers le vocabulaire "
        "réel du serveur d'inférence, température de génération jamais "
        "fixée explicitement) ont suffi à masquer un alignement DPO par "
        "ailleurs correctement validé et mesuré. Un incident "
        "d'infrastructure majeur (segfault systématique de vLLM, huit "
        "hypothèses testées et écartées avant résolution) a par "
        "ailleurs occupé une part significative du temps de "
        "déploiement — un rappel que l'intégration d'un composant "
        "d'inférence tiers récent, même imposé par le cahier des "
        "charges pour de bonnes raisons de performance, reste un risque "
        "projet à budgéter explicitement, pas un détail d'exécution."
    )
    _paragraphe(
        d,
        "Les prochaines étapes (§7) sont concrètes, priorisées, et "
        "directement issues des écarts réels constatés pendant cette "
        "mission plutôt que de recommandations génériques : contraindre "
        "le format de sortie, concevoir le garde-fou de sécurité "
        "clinique encore manquant, et mesurer statistiquement ce qui "
        "n'a pour l'instant été observé que ponctuellement. Le POC "
        "pose une base technique solide ; le chemin vers un déploiement "
        "clinique réel au CHSA reste conditionné à une validation "
        "médicale et réglementaire qui dépasse le périmètre de ce "
        "rapport."
    )

    # === Glossaire ==========================================================
    d.add_heading("Glossaire", level=1)
    _tableau(
        d,
        ["Terme", "Définition"],
        [
            ["DPO", "Direct Preference Optimization — alignement à "
                    "partir de paires (réponse préférée, réponse rejetée)."],
            ["ESI", "Emergency Severity Index — échelle de triage aux "
                    "urgences, niveaux 1 (critique) à 5 (non urgent)."],
            ["LoRA", "Low-Rank Adaptation — fine-tuning par matrices "
                     "de rang réduit, modèle de base gelé."],
            ["NF4 (quantification)", "NormalFloat 4-bit — format de "
                                      "quantification des poids, distinct du garde-fou "
                                      "de sécurité clinique homonyme."],
            ["PagedAttention", "Gestion par pages du cache d'attention "
                               "GPU dans vLLM, comme une mémoire virtuelle."],
            ["POC", "Proof of Concept — démonstrateur de faisabilité, "
                    "pas un produit de production."],
            ["QLoRA", "LoRA appliqué sur un modèle chargé en "
                      "quantification 4-bit."],
            ["SFT", "Supervised Fine-Tuning — spécialisation "
                    "supervisée classique, exemple par exemple."],
            ["vLLM", "Moteur de service d'inférence LLM haute "
                     "performance, imposé par le cahier des charges."],
        ],
        legende="Tableau 7 — Glossaire des sigles et termes techniques utilisés dans ce rapport.",
    )

    d.add_heading("Références", level=1)
    _paragraphe(d, "README.md (racine du dépôt) — journal complet des runs, métriques, "
                   "dépannage et guide de déploiement.")
    _paragraphe(d, "docs/00_cadrage/01_cahier_des_charges.md — cahier des charges de la mission.")
    _paragraphe(d, "docs/00_cadrage/05_presentation_soutenance.pptx — support de soutenance détaillé.")
    _paragraphe(d, "vllm-project/vllm, issue #58616 — signalement du segfault et de sa résolution.")
    _paragraphe(d, "docs/01_environnement/00_guide_installation_environnement.md — guide d'installation.")
    _paragraphe(d, "docs/references/M14 Support de presentation_V3.pptx.pdf — support de "
                   "présentation détaillé, référence complémentaire.")

    d.save(str(SORTIE))
    print(f"Rapport genere : {SORTIE}")


if __name__ == "__main__":
    construire()
