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
            "Figure 0 — Vue d'ensemble du pipeline, des données brutes à l'endpoint déployé.",
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
        "aucune modification du code métier."
    )
    _figure(d, DIAGRAMMES / "01_environnement" / "paquets" / "architecture_hexagonale.png",
            "Figure 1 — Architecture hexagonale du projet : domaine, cas d'usage et adaptateurs.")

    # === 1. Préparation des données ======================================
    d.add_heading("1. Préparation des données", level=1)

    d.add_heading("1.1 Méthodologie", level=2)
    _paragraphe(
        d,
        "Quatre corpus publics ont été collectés et unifiés dans un "
        "schéma pivot commun (JSON, bilingue FR/EN) : MediQAl et "
        "MedQuAD (questions/réponses médicales), FrenchMedMCQA (QCM "
        "pharmacie/médecine) et UltraMedical-Preference (paires "
        "choisies/rejetées, seule source directement exploitable pour "
        "le DPO). Chaque exemple pivot porte une source, une langue, un "
        "type (sft ou dpo) et, lorsqu'ils sont disponibles, des champs "
        "cliniques structurés (symptômes, antécédents, constantes "
        "vitales)."
    )
    _paragraphe(
        d,
        "L'anonymisation (Microsoft Presidio, imposée au cahier des "
        "charges) retire les identifiants directs (noms, dates, "
        "identifiants de dossier) avant toute publication sur le Hub "
        "Hugging Face, avec une révision manuelle de la PII résiduelle "
        "et un contrôle qualité systématique sur un échantillon "
        "stratifié. Le découpage final respecte une répartition "
        "train/validation/test stratifiée par source, vérifiée "
        "automatiquement (écart maximal toléré entre strates)."
    )
    _figure(d, DIAGRAMMES / "02_etape1_donnees" / "activite" / "pipeline_donnees.png",
            "Figure 2 — Pipeline de préparation des données : collecte, pivot, anonymisation, splits.")

    d.add_heading("1.2 Résultat", level=2)
    _paragraphe(
        d,
        "Le jeu de données final atteint l'ordre de grandeur visé par "
        "le cahier des charges (Livrable 1 : environ 5000 paires SFT "
        "plus les paires de préférence DPO), publié et versionné sur le "
        "Hub Hugging Face (dépôts privés dédiés à chaque étape), "
        "bilingue, anonymisé et prêt pour l'entraînement."
    )

    # === 2. SFT + LoRA ====================================================
    d.add_heading("2. SFT + LoRA", level=1)

    d.add_heading("2.1 Méthodologie", level=2)
    _paragraphe(
        d,
        "Le modèle de base Qwen3-1.7B-Base est spécialisé par "
        "fine-tuning supervisé (SFT) avec un adaptateur LoRA (Low-Rank "
        "Adaptation) plutôt qu'un fine-tuning complet : seule une "
        "fraction des paramètres est entraînée (matrices de rang "
        "réduit insérées dans les couches d'attention), ce qui réduit "
        "drastiquement le coût mémoire et de calcul tout en conservant "
        "les capacités générales du modèle de base — un compromis "
        "adapté à un POC avec budget de calcul contraint. Le chargement "
        "s'effectue en quantification 4-bit (BitsAndBytesConfig) pour "
        "l'entraînement sur une seule GPU distante (HF Jobs)."
    )
    _paragraphe(
        d,
        "L'évaluation baseline zero-shot (modèle non spécialisé) sert "
        "de point de comparaison avant tout entraînement, sur le même "
        "jeu de test clinique que les évaluations post-SFT et post-DPO."
    )
    _figure(d, DIAGRAMMES / "03_etape2_sft" / "activite" / "pipeline_sft_lora.png",
            "Figure 3 — Pipeline SFT+LoRA : baseline, entraînement, évaluation post-SFT.")

    d.add_heading("2.2 Résultat", level=2)
    _paragraphe(
        d,
        "Le checkpoint SFT-LoRA atteint un F1 (token) d'environ 0,112 "
        "sur le jeu de test clinique — la valeur de référence à "
        "laquelle le DPO est ensuite comparé (§3). C'est ce checkpoint "
        "qui sert de politique de référence (π_ref) pour l'alignement "
        "par préférences."
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
        "s'écarter de π_ref."
    )
    _figure(d, DIAGRAMMES / "04_etape3_dpo" / "activite" / "dpo_double_fonction_entrainement.png",
            "Figure 4 — Boucle d'entraînement DPO : double passage (politique + référence figée).")

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
        legende="Tableau 1 — Journal des runs DPO (extrait, README §3.2). "
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
        legende="Tableau 2 — Évaluation post-DPO (checkpoint "
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
        "base à tout moment sans réentraînement."
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
        "réel."
    )
    _figure(d, DIAGRAMMES / "05_etape4_deploiement" / "deploiement" / "deploiement_etape4.png",
            "Figure 5 — Architecture de déploiement réelle : Space Docker/GPU (API+vLLM), "
            "frontend local, journal d'audit persistant.")

    d.add_heading("4.2 Incident technique majeur et résolution", level=2)
    _paragraphe(
        d,
        "Le premier déploiement réel s'est heurté à un segfault natif "
        "systématique de vLLM, survenant juste après le chargement des "
        "poids du modèle, sur l'instance GPU réelle (NVIDIA L4). Huit "
        "pistes ciblées ont été testées et exclues méthodiquement, "
        "chacune avec le même échec identique malgré le changement : "
        "configuration du multiprocessing interne de vLLM, permissions "
        "utilisateur du conteneur, adaptateur LoRA, moteur d'attention, "
        "ordonnancement asynchrone, délai de démarrage, version de "
        "vLLM, et désaccord pilote/CUDA. Un test décisif "
        "(ralentissement extrême de l'exécution via une trace complète "
        "des appels Python) a permis un démarrage sain, signature "
        "caractéristique d'une condition de course dépendante du temps "
        "réel d'exécution."
    )
    _paragraphe(
        d,
        "La résolution est venue d'un changement de stratégie plutôt "
        "que d'un correctif ciblé supplémentaire : reconstruire l'image "
        "du Space à partir de l'image Docker officielle "
        "vllm/vllm-openai (construite et testée par l'équipe vLLM "
        "elle-même) plutôt que d'installer vLLM manuellement dans une "
        "image CUDA générique. Le problème a disparu entièrement. Cet "
        "incident, sa reproduction complète et sa résolution ont été "
        "signalés en amont au projet vLLM (issue publique "
        "vllm-project/vllm#58616) pour bénéficier à d'autres équipes "
        "confrontées au même symptôme."
    )
    _paragraphe(
        d,
        "Deux bugs applicatifs supplémentaires ont ensuite été trouvés "
        "et corrigés lors de la toute première conversation réelle "
        "contre ce vLLM enfin fonctionnel : (1) le paramètre de nombre "
        "de tokens n'était jamais traduit vers le vocabulaire réel de "
        "l'API compatible OpenAI de vLLM, laissant la génération partir "
        "sans limite de tour de conversation ; (2) aucune température "
        "de génération n'était fixée explicitement, laissant vLLM "
        "retomber sur un échantillonnage aléatoire non validé, "
        "provoquant des réponses incohérentes (mélange de langues) "
        "indépendamment du matériel GPU utilisé (confirmé identique sur "
        "L4 et T4). Les deux corrections ont rétabli des réponses "
        "cohérentes en conditions réelles."
    )

    d.add_heading("4.3 CI/CD et endpoint réel", level=2)
    _paragraphe(
        d,
        "Le pipeline GitHub Actions (imposé au cahier des charges) "
        "exécute la suite de tests automatisés (448 tests, 0 échec) "
        "et vérifie la construction de l'image Docker de l'API à "
        "chaque push, sans dépendance GPU. L'endpoint réel "
        "(mombasstic/chsa-triage-api) répond de façon opérationnelle : "
        "chargement du modèle et de l'adaptateur LoRA DPO, healthcheck "
        "applicatif ne renvoyant jamais d'erreur brute, latence de "
        "réponse observée de l'ordre de 5 à 30 secondes par tour "
        "d'entretien selon le matériel GPU alloué (L4 ou T4) — cette "
        "latence n'a pas encore fait l'objet d'une mesure statistique "
        "systématique (§6)."
    )
    _figure(d, DIAGRAMMES / "05_etape4_deploiement" / "activite" / "pipeline_ci_cd.png",
            "Figure 6 — Pipeline CI/CD : tests automatisés + vérification du build Docker.")

    d.add_heading("4.4 Traçabilité (garde-fou F6)", level=2)
    _paragraphe(
        d,
        "Chaque tour d'entretien et chaque demande de diagnostic est "
        "consigné dans un journal d'audit horodaté (conversation, "
        "entrée, sortie, version du modèle), persisté gratuitement dans "
        "un dataset privé Hugging Face via un mécanisme de publication "
        "périodique déjà utilisé ailleurs dans le projet pour le suivi "
        "d'expérimentation — le journal survit ainsi aux redémarrages "
        "du conteneur, contrairement à une simple écriture sur le "
        "disque éphémère du Space."
    )

    # === 5. Analyse des resultats vs criteres d'acceptation ==============
    d.add_heading("5. Analyse des résultats", level=1)
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
             "toujours respecté en sortie réelle de vLLM (voir §6)"],
            ["Accuracy ESI > baseline zéro-shot, mesurable", "Non mesurable tant que le "
             "format JSON n'est pas fiable"],
            ["Aucune réponse < 4/7 en sécurité (garde-fou NF4)", "Non implémenté — juge de "
             "sécurité clinique non encore conçu"],
        ],
        legende="Tableau 3 — Statut réel des critères d'acceptation du POC.",
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
        "suite du projet."
    )

    # === 6. Roadmap ========================================================
    d.add_heading("6. Roadmap et recommandations pour le passage à l'échelle", level=1)

    d.add_heading("6.1 Court terme (avant toute extension de périmètre)", level=2)
    _paragraphe(
        d,
        "Contraindre la sortie du diagnostic à un schéma JSON strict "
        "au niveau du serveur d'inférence (grammaire de génération "
        "contrainte, disponible nativement côté vLLM) plutôt que par le "
        "seul prompt — c'est l'écart le plus bloquant identifié (§5) et "
        "le plus directement actionnable. Concevoir et implémenter le "
        "juge de sécurité clinique (garde-fou NF4), aujourd'hui "
        "explicitement en attente d'une décision produit. Mesurer la "
        "latence de façon statistique (p50/p95 sur un échantillon "
        "représentatif de conversations), pas seulement de façon "
        "ponctuelle comme actuellement."
    )

    d.add_heading("6.2 Moyen terme (fiabilisation avant extension d'usage)", level=2)
    _paragraphe(
        d,
        "Ajouter une supervision applicative (alertes sur les "
        "réponses hors format, tableau de bord de latence réelle à "
        "partir du journal d'audit déjà persisté). Étendre le jeu de "
        "test clinique au-delà du périmètre actuel pour fiabiliser la "
        "mesure d'accuracy ESI. Documenter et tester une procédure de "
        "restauration du service en cas d'indisponibilité prolongée de "
        "l'infrastructure GPU sous-jacente — un point de fragilité "
        "concret rencontré plusieurs fois pendant ce déploiement, "
        "indépendant du code du projet."
    )

    d.add_heading("6.3 Long terme (déploiement à l'échelle du CHSA)", level=2)
    _paragraphe(
        d,
        "Avant toute mise en production clinique réelle : validation "
        "médicale formelle des diagnostics proposés par un comité "
        "clinique du CHSA, étude d'impact réglementaire (dispositif "
        "médical logiciel), et mécanisme de retour d'expérience "
        "structuré depuis le terrain vers une boucle de réentraînement "
        "périodique du modèle. Sur le plan technique, prévoir une "
        "architecture à plusieurs instances (le magasin de conversations "
        "actuel est volontairement en mémoire, donc incompatible avec "
        "plusieurs instances sans état partagé externe) et un budget "
        "GPU dimensionné sur la charge réelle observée plutôt que sur "
        "l'usage ponctuel de ce POC."
    )

    # === Conclusion ========================================================
    d.add_heading("Conclusion", level=1)
    _paragraphe(
        d,
        "Ce POC démontre la faisabilité technique d'un agent de "
        "triage médical spécialisé par SFT+DPO et servi par un endpoint "
        "vLLM réel, avec une architecture hexagonale qui a concrètement "
        "facilité le déploiement (bascule entre moteurs d'inférence "
        "sans changement du domaine métier). Le principal apprentissage "
        "de la phase de déploiement est que la validation d'un modèle "
        "aligné ne peut pas se limiter à son évaluation hors ligne : "
        "deux défauts d'intégration invisibles en évaluation "
        "(traduction de paramètres, température de génération) ont "
        "suffi à masquer un alignement DPO par ailleurs correctement "
        "validé. Les prochaines étapes (§6) sont concrètes, priorisées, "
        "et directement issues des écarts réels constatés pendant cette "
        "mission plutôt que de recommandations génériques."
    )

    d.add_heading("Références", level=1)
    _paragraphe(d, "README.md (racine du dépôt) — journal complet des runs, métriques, "
                   "dépannage et guide de déploiement.")
    _paragraphe(d, "docs/00_cadrage/01_cahier_des_charges.md — cahier des charges de la mission.")
    _paragraphe(d, "docs/00_cadrage/05_presentation_soutenance.pptx — support de soutenance détaillé.")
    _paragraphe(d, "vllm-project/vllm, issue #58616 — signalement du segfault et de sa résolution.")
    _paragraphe(d, "docs/01_environnement/00_guide_installation_environnement.md — guide d'installation.")

    d.save(str(SORTIE))
    print(f"Rapport genere : {SORTIE}")


if __name__ == "__main__":
    construire()
