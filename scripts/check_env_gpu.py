"""
Verifie que l'environnement cloud (Environnement B - GPU) est pret
avant de lancer un run SFTTrainer ou DPOTrainer couteux.

Reprend la checklist du manuel "Fine-Tuning Supervise (SFT) et
Alignement de LLMs", chapitre 1 :
  1. Le tokenizer expose bien le chat template natif du modele cible.
  2. Les tokens speciaux ChatML ne se fragmentent pas en sous-tokens.
  3. `assistant_only_loss` est disponible dans la version de `trl`
     installee.
  4. Le chargement 4-bit (QLoRA) fonctionne sans erreur CUDA.

Usage (sur l'environnement cloud, apres requirements-cloud.txt) :
    python scripts/check_env_gpu.py --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import sys


def verifier_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print("  torch.................: MANQUANT              [ECHEC]")
        return False

    disponible = torch.cuda.is_available()
    statut = "OK" if disponible else "ECHEC"
    print(f"  CUDA disponible.......: {disponible!s:<20} [{statut}]")

    if disponible:
        nom_gpu = torch.cuda.get_device_name(0)
        vram_go = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU detecte...........: {nom_gpu:<20} [OK]")
        print(f"  VRAM totale...........: {vram_go:.1f} Go{'':<15} [OK]")
    return disponible


def verifier_trl() -> bool:
    try:
        import trl
        from trl import SFTConfig, DPOConfig  # noqa: F401
    except ImportError as exc:
        print(f"  trl (SFT/DPOConfig)...: MANQUANT ({exc})   [ECHEC]")
        return False

    print(f"  Version trl...........: {trl.__version__:<20} [OK]")

    # `assistant_only_loss` doit exister sur SFTConfig
    a_le_flag = "assistant_only_loss" in SFTConfig.__dataclass_fields__
    statut = "OK" if a_le_flag else "ECHEC"
    print(f"  assistant_only_loss...: {'disponible' if a_le_flag else 'absent':<20} [{statut}]")
    return a_le_flag


def verifier_chat_template(nom_modele: str) -> bool:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  transformers..........: MANQUANT              [ECHEC]")
        return False

    tokenizer = AutoTokenizer.from_pretrained(nom_modele, trust_remote_code=True)

    if tokenizer.chat_template is None:
        print(f"  Chat template ({nom_modele})...: absent [ECHEC]")
        return False
    print(f"  Chat template ({nom_modele})...: present [OK]")

    # Verifie que les tokens de controle ChatML ne se fragmentent pas
    tokens_controle = ["<|im_start|>", "<|im_end|>"]
    tous_ok = True
    for token in tokens_controle:
        ids = tokenizer.encode(token, add_special_tokens=False)
        fragmente = len(ids) > 1
        statut = "ECHEC" if fragmente else "OK"
        print(f"  Token {token:<15}: {'fragmente' if fragmente else 'atomique':<15} [{statut}]")
        tous_ok = tous_ok and not fragmente

    if tokenizer.eos_token is None:
        print("  Token EOS.............: absent                [ECHEC]")
        tous_ok = False
    else:
        print(f"  Token EOS.............: {tokenizer.eos_token:<20} [OK]")

    return tous_ok


def verifier_chargement_4bit(nom_modele: str) -> bool:
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        print("  bitsandbytes/transformers...: MANQUANT        [ECHEC]")
        return False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        AutoModelForCausalLM.from_pretrained(
            nom_modele,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"  Chargement 4-bit ({nom_modele})...: reussi [OK]")
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic volontairement large
        print(f"  Chargement 4-bit ({nom_modele})...: echec ({exc}) [ECHEC]")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="Identifiant Hugging Face du modele cible",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Ne pas tester le chargement 4-bit complet (plus rapide)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("VERIFICATION DE L'ENVIRONNEMENT GPU (Environnement B - Cloud)")
    print("=" * 80)

    resultats = [
        verifier_cuda(),
        verifier_trl(),
        verifier_chat_template(args.model),
    ]

    if not args.skip_load:
        resultats.append(verifier_chargement_4bit(args.model))

    print("=" * 80)
    if all(resultats):
        print("Environnement GPU pret pour lancer SFTTrainer / DPOTrainer.")
    else:
        print("Des elements sont manquants ou incorrects : voir le detail ci-dessus.")
        sys.exit(1)


if __name__ == "__main__":
    main()
