#!/usr/bin/env bash
#
# Relance interfaces/cli/E1_04_00_anonymiser_dataset.py en boucle par vagues
# successives jusqu'a couverture complete du dataset pivot, sans avoir
# a relancer la commande a la main a chaque vague.
#
# Le CLI (--dataset/--sortie/--strategie/--limite) ne modifie JAMAIS
# le pivot source : chaque appel ajoute au fichier --sortie (separe)
# un echantillon stratifie (type_exemple, source) d'au plus --limite
# nouveaux exemples parmi ceux du pivot pas encore presents dans
# --sortie. Ce script rappelle la meme commande en boucle ; un
# exemple deja present dans --sortie n'est jamais retraite ; jusqu'a
# ce que --sortie contienne autant d'exemples que le pivot source, ou
# qu'une vague ne fasse plus progresser --sortie (protection
# anti-boucle infinie ci-dessous).
#
# Usage :
#   scripts/anonymiser_par_lots.sh [pivot] [sortie] [strategie] [taille_lot]
#   scripts/anonymiser_par_lots.sh data/processed/dataset_pivot.jsonl data/processed/dataset_pivot_anonymise.jsonl replace 5000

set -euo pipefail

PIVOT="${1:-data/processed/dataset_pivot.jsonl}"
SORTIE="${2:-data/processed/dataset_pivot_anonymise.jsonl}"
STRATEGIE="${3:-replace}"
TAILLE_LOT="${4:-5000}"
LOG_DIR="logs/anonymisation"
mkdir -p "${LOG_DIR}"

if [ ! -f "${PIVOT}" ]; then
    echo "Erreur : fichier pivot introuvable : ${PIVOT}" >&2
    exit 1
fi

compter_lignes() {
    if [ -f "$1" ]; then
        wc -l < "$1"
    else
        echo 0
    fi
}

TOTAL_PIVOT=$(wc -l < "${PIVOT}")
ITERATION=1
DEBUT=$(date +%s)
DERNIER_LOG=""

while true; do
    DEJA_TRAITES=$(compter_lignes "${SORTIE}")
    if [ "${DEJA_TRAITES}" -ge "${TOTAL_PIVOT}" ]; then
        break
    fi

    HORODATAGE=$(date +%Y%m%d_%H%M%S)
    LOG_FICHIER="${LOG_DIR}/anonymisation_${ITERATION}_${HORODATAGE}.log"
    DERNIER_LOG="${LOG_FICHIER}"
    echo "=== Iteration ${ITERATION} : ${DEJA_TRAITES}/${TOTAL_PIVOT} traites avant ce lot ==="

    set +e
    uv run python interfaces/cli/E1_04_00_anonymiser_dataset.py \
        --dataset "${PIVOT}" --sortie "${SORTIE}" \
        --strategie "${STRATEGIE}" --limite "${TAILLE_LOT}" \
        2>&1 | tee "${LOG_FICHIER}"
    CODE_SORTIE=${PIPESTATUS[0]}
    set -e

    if [ "${CODE_SORTIE}" -ne 0 ]; then
        echo "Erreur : E1_04_00_anonymiser_dataset.py a echoue (code ${CODE_SORTIE}) a l'iteration ${ITERATION}. Voir ${LOG_FICHIER}." >&2
        exit "${CODE_SORTIE}"
    fi

    APRES=$(compter_lignes "${SORTIE}")
    if [ "${APRES}" -le "${DEJA_TRAITES}" ]; then
        echo "Erreur : aucune progression a l'iteration ${ITERATION} (${DEJA_TRAITES} -> ${APRES} traites). Arret pour eviter une boucle infinie. Voir ${LOG_FICHIER}." >&2
        exit 1
    fi

    ITERATION=$((ITERATION + 1))
done

FIN=$(date +%s)
DUREE=$((FIN - DEBUT))
NB_ITERATIONS=$((ITERATION - 1))

echo "=== Anonymisation complete : ${TOTAL_PIVOT}/${TOTAL_PIVOT} exemples ==="
echo "Iterations : ${NB_ITERATIONS}"
echo "Duree totale approx. : ${DUREE}s"
if [ -n "${DERNIER_LOG}" ]; then
    echo "Dernier log : ${DERNIER_LOG}"
fi
