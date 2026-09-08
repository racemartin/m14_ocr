#!/usr/bin/env bash
#
# Relance interfaces/cli/anonymiser_dataset.py en boucle jusqu'a ce que tout
# le dataset pivot soit anonymise, sans avoir a relancer la commande a la
# main a chaque plantage.
#
# Important : le CLI anonymise IN PLACE (pas de --sortie) et n'a pas de
# --limite -- une seule invocation tente de traiter en une passe TOUS les
# exemples encore anonymise=false du fichier --dataset, en persistant sa
# progression exemple par exemple. Sur un gros corpus, cette passe peut
# planter avant la fin (OOM, erreur Presidio). Ce script relance alors
# automatiquement la meme commande : les exemples deja anonymises ne sont
# pas retraites, donc chaque nouvelle iteration ne fait avancer que le
# reliquat, jusqu'a couverture complete.
#
# Usage :
#   scripts/anonymiser_par_lots.sh [chemin_pivot] [strategie]
#   scripts/anonymiser_par_lots.sh data/processed/dataset_pivot.jsonl replace

set -euo pipefail

PIVOT="${1:-data/processed/dataset_pivot.jsonl}"
STRATEGIE="${2:-replace}"
LOG_DIR="logs/anonymisation"
mkdir -p "${LOG_DIR}"

if [ ! -f "${PIVOT}" ]; then
    echo "Erreur : fichier pivot introuvable : ${PIVOT}" >&2
    exit 1
fi

compter_anonymises() {
    grep -c '"anonymise":[[:space:]]*true' "${PIVOT}" || true
}

TOTAL_PIVOT=$(wc -l < "${PIVOT}")
ITERATION=1
DEBUT=$(date +%s)
DERNIER_LOG=""

while true; do
    AVANT=$(compter_anonymises)
    if [ "${AVANT}" -ge "${TOTAL_PIVOT}" ]; then
        break
    fi

    HORODATAGE=$(date +%Y%m%d_%H%M%S)
    LOG_FICHIER="${LOG_DIR}/anonymisation_${ITERATION}_${HORODATAGE}.log"
    DERNIER_LOG="${LOG_FICHIER}"
    echo "=== Iteration ${ITERATION} : ${AVANT}/${TOTAL_PIVOT} anonymises avant cette passe ==="

    set +e
    uv run python interfaces/cli/anonymiser_dataset.py \
        --dataset "${PIVOT}" --strategie "${STRATEGIE}" \
        2>&1 | tee "${LOG_FICHIER}"
    CODE_SORTIE=${PIPESTATUS[0]}
    set -e

    if [ "${CODE_SORTIE}" -ne 0 ]; then
        echo "Erreur : anonymiser_dataset.py a echoue (code ${CODE_SORTIE}) a l'iteration ${ITERATION}. Voir ${LOG_FICHIER}." >&2
        exit "${CODE_SORTIE}"
    fi

    APRES=$(compter_anonymises)
    if [ "${APRES}" -le "${AVANT}" ]; then
        echo "Erreur : aucune progression a l'iteration ${ITERATION} (${AVANT} -> ${APRES} anonymises). Arret pour eviter une boucle infinie. Voir ${LOG_FICHIER}." >&2
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
