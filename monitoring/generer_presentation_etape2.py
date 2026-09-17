"""Genere la presentation PowerPoint de synthese de l'Etape 2 (SFT+LoRA).

Usage (python-pptx n'est PAS une dependance permanente du projet, installee
de facon isolee pour cette seule execution) :

    uv run --with python-pptx python monitoring/generer_presentation_etape2.py

Regenere docs/03_etape2_sft/04_presentation_synthese_sft_lora.pptx a partir
des chiffres reels mesures et deja documentes dans le README (sections 2.2,
2.3, 2.4). Si ces chiffres changent (nouveau run, nouvelle mesure), mettre a
jour les constantes ci-dessous puis relancer ce script plutot que d'editer
le .pptx a la main.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

CHEMIN_SORTIE = Path(__file__).resolve().parent.parent / "docs" / "03_etape2_sft" / "04_presentation_synthese_sft_lora.pptx"

COULEUR_TITRE = RGBColor(0x1F, 0x3A, 0x5F)
COULEUR_ACCENT = RGBColor(0x2E, 0x7D, 0x32)
COULEUR_TEXTE = RGBColor(0x33, 0x33, 0x33)
COULEUR_FOND_ENTETE = RGBColor(0xF2, 0xF5, 0xF8)

LARGEUR = Inches(13.333)
HAUTEUR = Inches(7.5)


def ajouter_diapositive_vide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def ajouter_bandeau_titre(diapo, titre: str, sous_titre: str | None = None) -> None:
    bandeau = diapo.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, LARGEUR, Inches(1.15))
    bandeau.fill.solid()
    bandeau.fill.fore_color.rgb = COULEUR_FOND_ENTETE
    bandeau.line.fill.background()
    bandeau.shadow.inherit = False

    zone = diapo.shapes.add_textbox(Inches(0.5), Inches(0.15), LARGEUR - Inches(1.0), Inches(0.9))
    cadre = zone.text_frame
    cadre.word_wrap = True
    p = cadre.paragraphs[0]
    p.text = titre
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COULEUR_TITRE

    if sous_titre:
        p2 = cadre.add_paragraph()
        p2.text = sous_titre
        p2.font.size = Pt(14)
        p2.font.color.rgb = COULEUR_TEXTE


def ajouter_liste(diapo, puces: list[str], top: float = 1.5, taille_police: int = 18, gras_premier_mot: bool = False) -> None:
    zone = diapo.shapes.add_textbox(Inches(0.7), Inches(top), LARGEUR - Inches(1.4), HAUTEUR - Inches(top) - Inches(0.4))
    cadre = zone.text_frame
    cadre.word_wrap = True
    for i, puce in enumerate(puces):
        p = cadre.paragraphs[0] if i == 0 else cadre.add_paragraph()
        p.text = f"• {puce}"
        p.font.size = Pt(taille_police)
        p.font.color.rgb = COULEUR_TEXTE
        p.space_after = Pt(12)


def ajouter_chiffre_cle(diapo, chiffre: str, legende: str, left: float, top: float, largeur: float = 3.6) -> None:
    zone = diapo.shapes.add_textbox(Inches(left), Inches(top), Inches(largeur), Inches(1.8))
    cadre = zone.text_frame
    cadre.word_wrap = True
    p1 = cadre.paragraphs[0]
    p1.text = chiffre
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = COULEUR_ACCENT
    p1.alignment = PP_ALIGN.CENTER
    p2 = cadre.add_paragraph()
    p2.text = legende
    p2.font.size = Pt(13)
    p2.font.color.rgb = COULEUR_TEXTE
    p2.alignment = PP_ALIGN.CENTER


def ajouter_tableau(diapo, entetes: list[str], lignes: list[list[str]], top: float = 1.6, largeur_colonnes: list[float] | None = None) -> None:
    nb_lignes = len(lignes) + 1
    nb_colonnes = len(entetes)
    largeur_tableau = Inches(11.5)
    hauteur_tableau = Inches(0.5 * nb_lignes)
    graphique = diapo.shapes.add_table(
        nb_lignes, nb_colonnes, Inches(0.9), Inches(top), largeur_tableau, hauteur_tableau
    )
    table = graphique.table

    if largeur_colonnes:
        for idx, l in enumerate(largeur_colonnes):
            table.columns[idx].width = Inches(l)

    for c, entete in enumerate(entetes):
        cellule = table.cell(0, c)
        cellule.text = entete
        for p in cellule.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(15)
            p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cellule.fill.solid()
        cellule.fill.fore_color.rgb = COULEUR_TITRE

    for r, ligne in enumerate(lignes, start=1):
        for c, valeur in enumerate(ligne):
            cellule = table.cell(r, c)
            cellule.text = valeur
            for p in cellule.text_frame.paragraphs:
                p.font.size = Pt(15)
                p.font.color.rgb = COULEUR_TEXTE
                if r == len(lignes) and c > 0:
                    p.font.bold = True
                    p.font.color.rgb = COULEUR_ACCENT


def ajouter_pied_de_page(diapo, texte: str) -> None:
    zone = diapo.shapes.add_textbox(Inches(0.5), HAUTEUR - Inches(0.4), LARGEUR - Inches(1.0), Inches(0.3))
    p = zone.text_frame.paragraphs[0]
    p.text = texte
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(0x88, 0x88, 0x88)


def construire_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = LARGEUR
    prs.slide_height = HAUTEUR

    # 1. Page de titre
    diapo = ajouter_diapositive_vide(prs)
    zone = diapo.shapes.add_textbox(Inches(0.8), Inches(2.5), LARGEUR - Inches(1.6), Inches(2.0))
    cadre = zone.text_frame
    cadre.word_wrap = True
    p1 = cadre.paragraphs[0]
    p1.text = "Etape 2 - SFT + LoRA"
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = COULEUR_TITRE
    p2 = cadre.add_paragraph()
    p2.text = "Du baseline zero-shot a l'evaluation post-entrainement"
    p2.font.size = Pt(22)
    p2.font.color.rgb = COULEUR_TEXTE
    p3 = cadre.add_paragraph()
    p3.text = "Projet CHSA-Triage - Qwen3-1.7B-Base"
    p3.font.size = Pt(16)
    p3.font.color.rgb = COULEUR_TEXTE
    p3.space_before = Pt(20)
    ajouter_pied_de_page(diapo, "17/09/2026")

    # 2. Objectif
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Objectif")
    ajouter_liste(
        diapo,
        [
            "Mesurer un point de depart (baseline, modele SANS entrainement) avant le SFT.",
            "Entrainer reellement le modele (SFT + LoRA) sur les donnees du projet.",
            "Re-mesurer apres entrainement pour confirmer une amelioration chiffree.",
            "Objectif issu du cahier des charges, section 9.",
        ],
        top=1.6,
        taille_police=20,
    )

    # 3. Baseline CPU
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Baseline CPU", "llama.cpp / GGUF Q4_K_M - Qwen/Qwen3-1.7B-Base quantifie")
    ajouter_chiffre_cle(diapo, "0.037", "F1 token (278 exemples, split test)", 0.8, 1.9)
    ajouter_chiffre_cle(diapo, "~21.6 s", "Latence moyenne / generation", 4.9, 1.9)
    ajouter_chiffre_cle(diapo, "36 / 278", "Echecs d'inference isoles\n(reessais automatiques, run non interrompu)", 9.0, 1.9)
    ajouter_liste(
        diapo,
        ["Exact match : 0.000 (attendu, cf. diapositive Limitations)."],
        top=4.3,
        taille_police=16,
    )

    # 4. Baseline GPU
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Baseline GPU", "transformers / bf16 - meme modele, SANS quantification")
    ajouter_chiffre_cle(diapo, "0.043", "F1 token (memes 278 exemples)", 0.8, 1.9)
    ajouter_chiffre_cle(diapo, "~7.3 s", "Latence moyenne / generation", 4.9, 1.9)
    ajouter_chiffre_cle(diapo, "0 / 278", "Echecs d'inference", 9.0, 1.9)
    ajouter_liste(
        diapo,
        [
            "Exact match : 0.000 (idem baseline CPU).",
            "Pourquoi deux baselines ? Isoler l'effet de la quantification de l'effet du "
            "futur entrainement, pour une comparaison finale propre.",
        ],
        top=4.3,
        taille_police=16,
    )

    # 5. Entrainement SFT-LoRA
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Entrainement SFT-LoRA reel", "GPU L4 (HF Jobs)")
    ajouter_liste(
        diapo,
        [
            "Rang LoRA 16 sur 4 modules d'attention (q_proj, k_proj, v_proj, o_proj).",
            "3 epoques, 342 pas, environ 20 minutes.",
            "Premier entrainement reel du projet mene a bien avec succes.",
            "Verdict de convergence : SAINE.",
            "Poids publies de facon persistante : mombasstic/chsa-triage-sft-lora.",
        ],
        top=1.6,
        taille_police=19,
    )

    # 6. Evaluation post-SFT
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Evaluation post-SFT", "Meme modele entraine, memes 278 exemples, memes metriques")
    ajouter_chiffre_cle(diapo, "0.112", "F1 token", 0.8, 1.9)
    ajouter_chiffre_cle(diapo, "~11.6 s", "Latence moyenne / generation", 4.9, 1.9)
    ajouter_chiffre_cle(diapo, "0 / 278", "Echecs d'inference", 9.0, 1.9)
    ajouter_liste(diapo, ["Exact match : 0.000 (idem les deux baselines)."], top=4.3, taille_police=16)

    # 7. Tableau comparatif
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Comparaison des trois runs", "Memes 278 exemples, memes metriques")
    ajouter_tableau(
        diapo,
        entetes=["", "Exact match", "F1 (token)", "Latence moyenne"],
        lignes=[
            ["Baseline CPU (Q4_K_M)", "0.000", "0.037", "~21.6 s"],
            ["Baseline GPU (bf16)", "0.000", "0.043", "~7.3 s"],
            ["Post-SFT (bf16 + LoRA)", "0.000", "0.112", "~11.6 s"],
        ],
        top=1.8,
        largeur_colonnes=[4.5, 2.33, 2.33, 2.34],
    )

    # 7bis. Graphique F1
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "F1 token par run")
    donnees = CategoryChartData()
    donnees.categories = ["Baseline CPU", "Baseline GPU", "Post-SFT"]
    donnees.add_series("F1 token", (0.037, 0.043, 0.112))
    diapo.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(2.0), Inches(1.7), Inches(9.3), Inches(5.0), donnees
    )

    # 8. Conclusion principale
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Conclusion principale")
    ajouter_liste(
        diapo,
        [
            "Le F1 token quasi triple par rapport au meilleur baseline : 0.043 -> 0.112.",
            "Premiere preuve chiffree que le SFT-LoRA a un effet mesurable.",
            "Coherent avec le verdict de convergence SAINE obtenu pendant l'entrainement.",
            "L'exact match reste a 0.000 sur les trois runs : ce n'est PAS un echec de "
            "l'entrainement, c'est un trait structurel de la metrique (correspondance "
            "caractere-a-caractere avec une reponse de reference en langage libre, "
            "quasi inatteignable meme avec une amelioration reelle).",
            "Le F1 token est la metrique qui porte reellement le signal ici.",
        ],
        top=1.6,
        taille_police=18,
    )

    # 9. Limitations honnetes
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Limitations honnetes")
    ajouter_liste(
        diapo,
        [
            "Classification du niveau ESI (objectif F3 du cahier des charges) : non "
            "calculable sur aucun des trois runs.",
            "Cause : le dataset actuel n'a pas encore de reponses de reference au format "
            "JSON structure qu'exige cette metrique.",
            "C'est un travail de donnees en attente, pas un defaut du pipeline "
            "d'evaluation.",
        ],
        top=1.8,
        taille_police=19,
    )

    # 10. Prochaines etapes
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Prochaines etapes")
    ajouter_liste(
        diapo,
        [
            "Etape 3 (DPO) : pas encore commencee.",
            "assistant_only_loss : correctif connu en attente. Aujourd'hui la perte est "
            "calculee sur la sequence complete, pas seulement sur la reponse de "
            "l'assistant, en raison d'une limitation technique deja identifiee et "
            "consignee.",
        ],
        top=1.8,
        taille_police=19,
    )

    # 11. Merci
    diapo = ajouter_diapositive_vide(prs)
    zone = diapo.shapes.add_textbox(Inches(0.8), Inches(3.0), LARGEUR - Inches(1.6), Inches(1.5))
    p = zone.text_frame.paragraphs[0]
    p.text = "Merci - Questions"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = COULEUR_TITRE

    return prs


def main() -> None:
    prs = construire_presentation()
    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(CHEMIN_SORTIE))
    print(f"Presentation ecrite : {CHEMIN_SORTIE} ({len(prs.slides)} diapositives)")


if __name__ == "__main__":
    main()
