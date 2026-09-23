"""Genere le support PowerPoint de soutenance couvrant toutes les etapes.

Usage (python-pptx n'est PAS une dependance permanente du projet, installee
de facon isolee pour cette seule execution) :

    uv run --with python-pptx python monitoring/generer_presentation_soutenance.py

Regenere docs/00_cadrage/05_presentation_soutenance.pptx a partir des
chiffres reels mesures et deja documentes dans README.md (sections 1 a 4).
Reutilise les fonctions utilitaires de generer_presentation_etape2.py
(meme palette/style) plutot que de les redupliquer. Si les chiffres cites
ici changent (nouveau run, nouvelle mesure), mettre a jour README.md
d'abord puis les constantes ci-dessous, et relancer ce script plutot que
d'editer le .pptx a la main.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

from generer_presentation_etape2 import (
    COULEUR_TEXTE,
    COULEUR_TITRE,
    HAUTEUR,
    LARGEUR,
    ajouter_bandeau_titre,
    ajouter_chiffre_cle,
    ajouter_diapositive_vide,
    ajouter_liste,
    ajouter_pied_de_page,
    ajouter_tableau,
)

CHEMIN_SORTIE = Path(__file__).resolve().parent.parent / "docs" / "00_cadrage" / "05_presentation_soutenance.pptx"


def ajouter_notes(diapo, texte: str) -> None:
    """Ajoute des notes du presentateur (job IDs reels, non affiches a l'ecran)."""
    diapo.notes_slide.notes_text_frame.text = texte


def construire_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = LARGEUR
    prs.slide_height = HAUTEUR

    # 1. Page de titre
    diapo = ajouter_diapositive_vide(prs)
    zone = diapo.shapes.add_textbox(Inches(0.8), Inches(2.3), LARGEUR - Inches(1.6), Inches(2.4))
    cadre = zone.text_frame
    cadre.word_wrap = True
    p1 = cadre.paragraphs[0]
    p1.text = "CHSA Triage - Soutenance"
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = COULEUR_TITRE
    p2 = cadre.add_paragraph()
    p2.text = "Preparation des donnees, SFT+LoRA, DPO, Deploiement"
    p2.font.size = Pt(22)
    p2.font.color.rgb = COULEUR_TEXTE
    p3 = cadre.add_paragraph()
    p3.text = "Projet CHSA-Triage - Qwen3-1.7B-Base"
    p3.font.size = Pt(16)
    p3.font.color.rgb = COULEUR_TEXTE
    p3.space_before = Pt(20)
    ajouter_pied_de_page(diapo, "23/09/2026")

    # 2. Introduction - objet et architecture
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Objet du POC & architecture", "Centre Hospitalier Saint-Aurelien - agent IA de triage medical")
    ajouter_liste(
        diapo,
        [
            "POC : aider au triage medical par entretien clinique conversationnel puis diagnostic structure.",
            "Architecture hexagonale : domaine / ports / adaptateurs, testable sans GPU a chaque etape.",
            "Un seul port MoteurInference reutilise sans modification a travers 7 usages reels "
            "(baselines, evaluation post-SFT, evaluation post-DPO, entretien clinique en production).",
            "4 etapes reelles : preparation des donnees -> SFT+LoRA -> DPO -> deploiement.",
        ],
        top=1.7,
        taille_police=18,
    )

    # 3. Introduction - resultat phare
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Resultat phare", "Vue d'ensemble, detail dans les sections suivantes")
    ajouter_chiffre_cle(diapo, "0,112", "F1 token post-SFT\n(vs 0,043 baseline GPU)", 0.7, 2.0)
    ajouter_chiffre_cle(diapo, "75 %", "rewards/accuracies DPO retenu\n(beta=0,3, 5000 exemples, verdict saine)", 4.9, 2.0)
    ajouter_chiffre_cle(diapo, "3", "pieces de deploiement\n(vLLM+LoRA, API, Streamlit)", 9.0, 2.0)
    ajouter_liste(
        diapo,
        [
            "Le SFT triple le F1 token par rapport a la meilleure baseline : premiere preuve chiffree d'un effet mesurable.",
            "Le DPO a d'abord revele un probleme reel a l'evaluation (degenerescence de generation), diagnostique, "
            "corrige (beta=0,3) et reconfirme a l'echelle reelle : le fil conducteur de la section 3.",
        ],
        top=4.3,
        taille_police=16,
    )

    # 4. Section 1 - Preparation de donnees
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "1. Preparation de donnees", "6 corpus sources fusionnes, anonymises, repartis en splits")
    ajouter_chiffre_cle(diapo, "134 883", "exemples pivot\n(147 204 bruts, 12 321 doublons ecartes)", 0.7, 2.0)
    ajouter_chiffre_cle(diapo, "RGPD", "Anonymisation Presidio + spaCy\npar vagues incrementales", 4.9, 2.0, largeur=4.0)
    ajouter_chiffre_cle(diapo, "train/val/test", "Splits stratifies,\njamais reassignes d'une execution a l'autre", 9.3, 2.0, largeur=3.6)
    ajouter_liste(
        diapo,
        [
            "Identifiant deterministe (hash de cle naturelle) : a mis au jour de vrais doublons invisibles avec des ids aleatoires.",
            "Pivot original jamais mute : l'anonymisation ecrit dans un fichier separe, texte source toujours disponible pour controle.",
            "Revision humaine persistee des candidats PII residuelle avant toute publication.",
        ],
        top=4.3,
        taille_police=16,
    )

    # 5. Section 2.1 - Architecture SFT
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "2. SFT + LoRA - Architecture", "Deux baselines zero-shot mesurees avant tout entrainement")
    ajouter_liste(
        diapo,
        [
            "MoteurInference : 3 adaptateurs reels (llama.cpp/GGUF, transformers/bf16, transformers+LoRA).",
            "EntraineurSupervise (TrlSftEntraineurAdapter), FormateurConversation (ChatMLFormateurAdapter), "
            "SuiviExperimentation (MLflow / TensorBoard / dataset HF).",
            "Deux baselines independantes AVANT tout entrainement (CPU quantifie Q4_K_M et GPU bf16), pour ne "
            "jamais melanger l'effet de la quantification avec l'effet reel de l'entrainement.",
        ],
        top=1.7,
        taille_police=19,
    )

    # 6. Section 2.2 - Baselines
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "2.2 Baselines zero-shot", "Qwen/Qwen3-1.7B-Base SANS entrainement, memes 278 exemples")
    ajouter_tableau(
        diapo,
        entetes=["", "Exact match", "F1 (token)", "Latence moyenne", "Echecs"],
        lignes=[
            ["CPU (Q4_K_M, llama.cpp)", "0,000", "0,037", "~21,6 s", "36/278"],
            ["GPU (bf16, transformers)", "0,000", "0,043", "~7,3 s", "0/278"],
        ],
        top=1.9,
        largeur_colonnes=[4.5, 1.9, 1.9, 2.1, 1.1],
    )
    ajouter_liste(
        diapo,
        ["La baseline GPU est plus rapide, plus fiable (zero echec) et legerement meilleure en F1 : reference retenue."],
        top=3.4,
        taille_police=16,
    )

    # 7. Section 2.3-2.4 - Entrainement SFT-LoRA reel
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "2.3-2.4 Entrainement SFT-LoRA reel", "GPU L4 (HF Jobs)")
    ajouter_liste(
        diapo,
        [
            "QLoRA 4-bit, rang LoRA 16 sur 4 modules d'attention (q_proj, k_proj, v_proj, o_proj).",
            "3 epoques, 342 pas, environ 20 minutes - premier entrainement reel du projet mene a bien.",
            "Verdict de convergence : SAINE.",
            "Poids LoRA publies de facon persistante : mombasstic/chsa-triage-sft-lora.",
        ],
        top=1.7,
        taille_police=19,
    )
    ajouter_notes(diapo, "Job HF Jobs 6aaab9a95527934177eeaac8. Courbe de perte reconstruite a posteriori depuis le log brut (backend de suivi mal configure a l'origine, corrige depuis, cf. README 2.6).")

    # 8. Section 2.5 - Evaluation post-SFT
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "2.5 Evaluation post-SFT", "Resultat le plus important du projet a ce jour")
    ajouter_tableau(
        diapo,
        entetes=["", "Exact match", "F1 (token)", "Latence moyenne"],
        lignes=[
            ["Baseline CPU (Q4_K_M)", "0,000", "0,037", "~21,6 s"],
            ["Baseline GPU (bf16)", "0,000", "0,043", "~7,3 s"],
            ["Post-SFT (bf16+LoRA)", "0,000", "0,112", "~11,6 s"],
        ],
        top=1.8,
        largeur_colonnes=[4.5, 2.33, 2.33, 2.34],
    )
    ajouter_liste(
        diapo,
        [
            "Le F1 token quasi triple par rapport a la meilleure baseline (0,043 -> 0,112), coherent avec le verdict SAINE.",
            "L'exact match reste a 0,000 sur les trois runs : attendu, la metrique exige une correspondance "
            "caractere-a-caractere avec des reponses de reference en langage libre.",
        ],
        top=4.1,
        taille_police=16,
    )

    # 9. Section 3.1 - Architecture DPO
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "3. DPO - Architecture & verification", "Continue le checkpoint SFT-LoRA deja entraine")
    ajouter_liste(
        diapo,
        [
            "13 etapes implementees (domaine -> ports -> application -> TrlDpoEntraineurAdapter -> "
            "training/E3_03_dpo_train.py), 428 tests passent, 4 ignores (GPU absent).",
            "Tout verifie sans GPU avant le premier run reel (installation temporaire de trl/peft "
            "pour confirmer les signatures), meme methode que pour le SFT.",
            "Deux bugs reels trouves et corriges au passage : le chat template natif de Qwen3 retirait "
            "le bloc <think> d'un chosen reformule ; la deserialisation d'un checkpoint DPO levait une TypeError.",
            "Continue mombasstic/chsa-triage-sft-lora ; poids DPO destines a mombasstic/chsa-triage-dpo-lora.",
        ],
        top=1.7,
        taille_police=17,
    )

    # 10. Section 3.2a - Entrainement DPO : iterations a 100 exemples
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "3.2 Entrainement DPO - diagnostiquer, corriger, reverifier", "Trois lancements reels sur 100 exemples (22/09/2026)")
    ajouter_liste(
        diapo,
        [
            "1. taux=5e-6, lot=4 : boucle complete avec succes, mais l'evaluation plante en OutOfMemoryError. "
            "Cause verifiee : per_device_eval_batch_size jamais fixe (defaut reel = 8, double du lot d'entrainement). "
            "Corrige : per_device_eval_batch_size = taille_lot.",
            "2. taux=5e-6, lot=1 : premier run complet bout-en-bout. Verdict sous_apprentissage ; "
            "rewards/accuracies=0,30 (pire que le hasard), rewards/margins negatif.",
            "3. taux=5e-5, lot=4 (recette corrigee) : rewards/accuracies 0,30 -> 0,625 (train) / 0,75 (eval), "
            "margins redevient positif. Verdict encore sous_apprentissage, mais probable faux negatif "
            "(le seuil de convergence herite du SFT ignore rewards/accuracies et rewards/margins).",
        ],
        top=1.7,
        taille_police=15,
    )
    ajouter_notes(
        diapo,
        "Jobs reels : 6ab2c3e552d0dbd7f1d7fcd1 (OOM eval, commit 250ce41 pour le fix), "
        "6ab2d7ee51992417dfcd40cd (sous_apprentissage lot=1). "
        "Le run 3 (taux=5e-5) n'a pas de job ID distinct documente dans le README au-dela de ces deux-la.",
    )

    # 11. Section 3.2b - Run complet a 5000 exemples
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "3.2 Entrainement DPO - run complet, 5000 exemples", "Cahier des charges §7, Livrable 1 - premier run a l'echelle reelle")
    ajouter_chiffre_cle(diapo, "SAINE", "Verdict de convergence\n(premier verdict sain DPO du projet)", 0.7, 2.0, largeur=3.8)
    ajouter_chiffre_cle(diapo, "72,5 %", "rewards/accuracies\n(rewards/margins = +3,00)", 5.0, 2.0, largeur=3.6)
    ajouter_chiffre_cle(diapo, "1h44", "Duree reelle\n(estimation initiale : 30-60 min)", 9.1, 2.0, largeur=3.6)
    ajouter_liste(
        diapo,
        [
            "taux_apprentissage=5e-5, taille_lot=4, beta=0,1 - recette corrigee d'apres les 3 runs precedents.",
            "rewards/chosen=-1,72, rewards/rejected=-4,72 : le modele prefere bien 'chosen' a 'rejected', comme attendu.",
            "Confirme la lecture faite a 100 exemples : le verdict sous_apprentissage etait un faux negatif du "
            "seuil herite du SFT, pas un echec d'apprentissage. Poids publies : mombasstic/chsa-triage-dpo-lora.",
            "Ce premier run a l'echelle reelle utilise encore beta=0,1 : l'evaluation qui suit (3.3) y "
            "decouvre un probleme reel, corrige puis reconfirme a cette meme echelle.",
        ],
        top=4.3,
        taille_police=15,
    )
    ajouter_notes(diapo, "Job HF Jobs 6ab2ff0c52d0dbd7f1d80b0b, checkpoint 165d4040855abb6a (outputs/dpo-lora/run-20260922T222105Z). 3992 exemples train reels, ~998 pas.")

    # 12. Section 3.3 - Evaluation post-DPO : un resultat inattendu
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "3.3 Evaluation post-DPO - un resultat inattendu", "Memes 278 exemples, meme metrique que les baselines et le post-SFT")
    ajouter_tableau(
        diapo,
        entetes=["", "Exact match", "F1 (token)", "Latence moyenne"],
        lignes=[
            ["Baseline CPU (Q4_K_M)", "0,000", "0,037", "~21,6 s"],
            ["Baseline GPU (bf16)", "0,000", "0,043", "~7,3 s"],
            ["Post-SFT (bf16+LoRA)", "0,000", "0,112", "~11,6 s"],
            ["Post-DPO (bf16+LoRA)", "0,000", "0,049", "~12,0 s"],
        ],
        top=1.8,
        largeur_colonnes=[4.5, 2.33, 2.33, 2.34],
    )
    ajouter_liste(
        diapo,
        [
            "Le F1 regresse nettement apres DPO (0,112 -> 0,049), quasiment au niveau de la baseline GPU seule.",
            "Cause reelle confirmee par inspection directe des generations : degenerescence reelle du modele "
            "(changements de langue aleatoires EN/FR -> JA/ZH/AR, effondrements en repetitions), pas seulement "
            "une perte du format JSON attendu.",
        ],
        top=4.4,
        taille_police=16,
    )

    # 13. Section 3.3 - Cause racine et correction
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "3.3 Cause racine, correction, confirmation a l'echelle", "Ancrage trop faible au modele de reference (pi_ref)")
    ajouter_liste(
        diapo,
        [
            "rewards/margins=+3,00 est tres eleve pour beta=0,1 : signe coherent d'un ancrage trop faible a pi_ref, "
            "pas seulement d'un parametre de decodage a l'evaluation.",
            "beta releve a 0,3 : gain confirme d'abord sur le checkpoint pas cher a 100 exemples, PUIS sur le "
            "run complet a 5000 (verdict saine, rewards/accuracies=0,75) - aucune regression en passant a l'echelle.",
        ],
        top=1.7,
        taille_police=17,
    )
    ajouter_tableau(
        diapo,
        entetes=["Checkpoint", "repetition_penalty", "F1 (token)"],
        lignes=[
            ["beta=0,1 (5000)", "-", "0,049"],
            ["beta=0,1 (5000)", "1,2", "0,063"],
            ["beta=0,3 (100, verification)", "1,2", "0,112 (= post-SFT)"],
            ["beta=0,3 (5000, echelle reelle)", "1,2", "0,110 (= post-SFT)"],
        ],
        top=4.1,
        largeur_colonnes=[5.5, 3.5, 3.0],
    )
    ajouter_notes(
        diapo,
        "Jobs d'evaluation reels : 6ab3933152d0dbd7f1d83a16 beta=0,1 sans repetition_penalty (F1=0,049) ; "
        "6ab3adaa52d0dbd7f1d8445b beta=0,1 + repetition_penalty=1,2 (F1=0,063) ; "
        "6ab3c38a52d0dbd7f1d84c5c beta=0,3 (checkpoint 100) + repetition_penalty=1,2 (F1=0,112) ; "
        "6ab40ee752d0dbd7f1d86485 beta=0,3 (checkpoint 5000, job d'entrainement 6ab3f52a51992417dfcd7fe5) "
        "+ repetition_penalty=1,2 (F1=0,110). beta=0,3 est desormais l'hyperparametre retenu pour ce projet.",
    )

    # 14. Diapositive de synthese
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "Synthese SFT + DPO", "Parametres retenus, resultats, problemes rencontres et corriges")
    ajouter_liste(
        diapo,
        [
            "Parametres DPO retenus (finaux) : beta=0,3, taux_apprentissage=5e-5, taille_lot=4, "
            "per_device_eval_batch_size=4, repetition_penalty=1,2 a l'inference - confirmes a l'echelle "
            "reelle (5000 exemples), aucune regression par rapport au run pas cher a 100.",
            "1. OOM a l'evaluation -> per_device_eval_batch_size jamais fixe (defaut = 8) -> fixe explicitement.",
            "2. Sous-apprentissage initial -> taux_apprentissage trop faible (5e-6) -> releve a 5e-5.",
            "3. Degenerescence de generation post-DPO -> ancrage trop faible a pi_ref (beta=0,1) -> "
            "beta releve a 0,3 + repetition_penalty=1,2 a l'inference.",
        ],
        top=1.7,
        taille_police=16,
    )
    ajouter_tableau(
        diapo,
        entetes=["", "Exact match", "F1 (token)", "Latence moyenne"],
        lignes=[
            ["Baseline CPU (Q4_K_M)", "0,000", "0,037", "~21,6 s"],
            ["Baseline GPU (bf16)", "0,000", "0,043", "~7,3 s"],
            ["Post-SFT (bf16+LoRA)", "0,000", "0,112", "~11,6 s"],
            ["Post-DPO (beta=0,3, retenu)", "0,000", "0,110", "~11,2 s"],
        ],
        top=4.7,
        largeur_colonnes=[4.5, 2.33, 2.33, 2.34],
    )

    # 15. Section 4 - Deploiement, decisions actees
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "4. Deploiement - decisions actees", "Code ecrit et teste, jamais deploye reellement a ce jour")
    ajouter_liste(
        diapo,
        [
            "Le LoRA DPO n'est jamais fusionne avec la base : vLLM le sert nativement (--enable-lora).",
            "L'entretien adaptatif et le format JSON strict + <think> sont geres au niveau du prompting a "
            "l'inference, jamais re-appris pendant l'entrainement.",
            "Le garde-fou de securite clinique NF4 (juge LLM, safety < 4/7) reste un point d'extension "
            "documente (TODO explicite dans le code) : decision produit encore ouverte.",
        ],
        top=1.7,
        taille_police=18,
    )

    # 16. Section 4 - Architecture en 3 pieces
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "4.1-4.2 Architecture en trois pieces", "Aucun Space HF cree ni pousse a ce jour")
    ajouter_liste(
        diapo,
        [
            "1. Space Docker/GPU (couteux, allume seulement pour les tests) : serveur vLLM + LoRA DPO non "
            "fusionne (deploy/space_gpu_api_vllm/).",
            "2. Space Docker/CPU : API FastAPI seule (Dockerfile racine), entretien multi-tours + bouton "
            "explicite 'obtenir le diagnostic', cle API (X-API-Key), journal d'audit JSONL append-only.",
            "3. Space Streamlit/CPU (leger, laisse allume en permanence) : interface de test de l'entretien "
            "clinique (interfaces/web/), sonde GET /sante en boucle sans synchronisation manuelle des demarrages.",
            "/sante retourne toujours HTTP 200 (corps structure disponible/detail) : un redemarrage automatique "
            "du conteneur API pendant le chargement du modele vLLM n'aiderait en rien.",
        ],
        top=1.7,
        taille_police=15,
    )

    # 17. Section 4.4 - CI/CD
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "4.4 CI/CD", ".github/workflows/ci.yml - aucun deploiement automatique")
    ajouter_liste(
        diapo,
        [
            "Job tests : installe dev+local+api (+pyyaml), lance uv run pytest tests/ -q. "
            "433 passed / 5 skipped au moment de l'ecriture.",
            "Job docker-build (apres tests) : reconstruit l'image de l'API pour verifier que le Dockerfile "
            "build, sans jamais la publier ni la deployer.",
            "Le deploiement des Spaces HF reste manuel par decision : jamais automatise dans ce pipeline.",
            "Aucun job GPU/vLLM reel lance dans le cadre du deploiement (le build Docker n'a pas pu etre "
            "verifie non plus dans ce bac a sable, WSL2 sans integration Docker Desktop).",
        ],
        top=1.7,
        taille_police=17,
    )

    # 18. Section 5 - Conclusions
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "5. Conclusions - statut reel")
    ajouter_liste(
        diapo,
        [
            "Etape 1 (donnees) : terminee - 134 883 exemples pivot, anonymises et repartis en splits.",
            "Etape 2 (SFT) : reussie et validee - F1 token triple par rapport a la meilleure baseline.",
            "Etape 3 (DPO) : entrainee avec succes a l'echelle reelle ; degenerescence a l'evaluation "
            "diagnostiquee et corrigee (beta=0,3 + repetition_penalty=1,2), gain confirme a l'echelle reelle "
            "(F1=0,110, quasi identique au post-SFT, aucune regression en passant de 100 a 5000 exemples).",
            "Etape 4 (deploiement) : code ecrit et teste (433 passed / 5 skipped), jamais deploye reellement.",
        ],
        top=1.6,
        taille_police=17,
    )

    # 19. Section 5 - Ce qui reste ouvert
    diapo = ajouter_diapositive_vide(prs)
    ajouter_bandeau_titre(diapo, "5. Conclusions - ce qui reste ouvert")
    ajouter_liste(
        diapo,
        [
            "Garde-fou de securite clinique NF4 (juge LLM, safety < 4/7) : decision produit encore ouverte, non implemente.",
            "Aucun Space HF (GPU vLLM, API, Streamlit) cree ni pousse a ce jour.",
            "Tests de latence et de robustesse en conditions reelles (charge, concurrence) restant a faire.",
            "Classification du niveau ESI (F3) non calculable : le dataset n'a pas encore de reponses de "
            "reference au format JSON structure qu'exige cette metrique.",
        ],
        top=1.7,
        taille_police=17,
    )

    # 20. Merci
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
