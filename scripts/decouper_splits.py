"""Script CLI pour découper le dataset pivot en fichiers de splits individuels (train, val, test).

Usage:
    uv run python scripts/decouper_splits.py \
        --input data/processed/dataset_pivot_anonymise.jsonl \
        --output-dir data/splits/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Découpe un fichier JSONL pivot en fichiers de splits."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Chemin du fichier pivot JSONL d'entrée",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        "-o",
        default="data/splits",
        help="Dossier de sortie pour les fichiers de splits (défaut: data/splits/)",
    )

    args = parser.parse_args()

    fichier_input = Path(args.input)
    dossier_sortie = Path(args.output_dir)

    if not fichier_input.exists():
        raise FileNotFoundError(f"Le fichier d'entrée n'existe pas : {fichier_input}")

    # Créer le dossier de sortie s'il n'existe pas
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    f_train_path = dossier_sortie / "train.jsonl"
    f_val_path = dossier_sortie / "val.jsonl"
    f_test_path = dossier_sortie / "test_baseline.jsonl"

    compteurs = {"train": 0, "val": 0, "test": 0, "ignores": 0}

    with (
        open(fichier_input, "r", encoding="utf-8") as f_in,
        open(f_train_path, "w", encoding="utf-8") as f_train,
        open(f_val_path, "w", encoding="utf-8") as f_val,
        open(f_test_path, "w", encoding="utf-8") as f_test,
    ):
        for line in f_in:
            if not line.strip():
                continue

            item = json.loads(line)
            split = str(item.get("split", "")).lower()

            ligne_json = json.dumps(item, ensure_ascii=False) + "\n"

            if split == "train":
                f_train.write(ligne_json)
                compteurs["train"] += 1
            elif split in ("validation", "val"):
                f_val.write(ligne_json)
                compteurs["val"] += 1
            elif split == "test":
                f_test.write(ligne_json)
                compteurs["test"] += 1
            else:
                compteurs["ignores"] += 1

    print(f"✅ Découpage terminé dans : {dossier_sortie.resolve()}")
    print(f"   • Train : {compteurs['train']} exemples -> {f_train_path.name}")
    print(f"   • Val   : {compteurs['val']} exemples -> {f_val_path.name}")
    print(f"   • Test  : {compteurs['test']} exemples -> {f_test_path.name}")
    if compteurs["ignores"] > 0:
        print(f"   ⚠️ Exemples ignorés (sans split valide) : {compteurs['ignores']}")


if __name__ == "__main__":
    main()