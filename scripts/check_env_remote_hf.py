"""
Verifie que l'acces a l'environnement distant Hugging Face (Jobs +
Spaces Dev Mode) est correctement configure, AVANT de lancer un
entrainement couteux. S'execute depuis la machine locale (pas besoin
de GPU pour ce script).

Usage :
    uv run python scripts/check_env_remote_hf.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def verifier_cli_hf_installe() -> bool:
    present = shutil.which("hf") is not None
    statut = "OK" if present else "ECHEC"
    print(f"  CLI 'hf' installee.........: {'oui' if present else 'non':<10} [{statut}]")
    if not present:
        print("    -> uv tool install huggingface_hub[cli]")
    return present


def verifier_authentification() -> bool:
    try:
        resultat = subprocess.run(
            ["hf", "auth", "whoami"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        connecte = resultat.returncode == 0 and resultat.stdout.strip()
        statut = "OK" if connecte else "ECHEC"
        nom = resultat.stdout.strip() if connecte else "non connecte"
        print(f"  Authentification HF.........: {nom:<10} [{statut}]")
        return bool(connecte)
    except FileNotFoundError:
        print("  Authentification HF.........: CLI absente [ECHEC]")
        return False
    except subprocess.TimeoutExpired:
        print("  Authentification HF.........: timeout    [ECHEC]")
        return False


def verifier_hf_jobs_disponible() -> bool:
    try:
        resultat = subprocess.run(
            ["hf", "jobs", "ps"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        ok = resultat.returncode == 0
        statut = "OK" if ok else "ECHEC"
        print(f"  HF Jobs accessible..........: {'oui' if ok else 'non':<10} [{statut}]")
        if not ok:
            print(f"    -> {resultat.stderr.strip()[:200]}")
            print("    -> Verifier qu'un solde de credits positif est disponible.")
        return ok
    except FileNotFoundError:
        print("  HF Jobs accessible..........: CLI absente [ECHEC]")
        return False


def verifier_connectivite_gpu_minimale() -> bool:
    """Lance un job minuscule pour verifier bout-en-bout (facture quelques secondes)."""
    reponse = input(
        "  Lancer un job de test GPU (facture quelques secondes) ? [o/N] "
    ).strip().lower()
    if reponse != "o":
        print("  Test GPU distant............: ignore     [SKIP]")
        return True

    try:
        resultat = subprocess.run(
            [
                "hf", "jobs", "uv", "run",
                "--flavor", "t4-small",
                "python", "-c",
                "import torch; print(torch.cuda.get_device_name())",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        ok = resultat.returncode == 0
        statut = "OK" if ok else "ECHEC"
        print(f"  Test GPU distant.............: {'reussi' if ok else 'echec':<10} [{statut}]")
        if not ok:
            print(f"    -> {resultat.stderr.strip()[:300]}")
        return ok
    except subprocess.TimeoutExpired:
        print("  Test GPU distant.............: timeout    [ECHEC]")
        return False


def main() -> None:
    print("=" * 80)
    print("VERIFICATION DE L'ENVIRONNEMENT DISTANT (HF Jobs / Spaces Dev Mode)")
    print("=" * 80)

    resultats = [
        verifier_cli_hf_installe(),
        verifier_authentification(),
        verifier_hf_jobs_disponible(),
    ]

    if all(resultats):
        resultats.append(verifier_connectivite_gpu_minimale())

    print("=" * 80)
    if all(resultats):
        print("Environnement distant pret. Pour le developpement interactif,")
        print("suivre la section 3.3 de docs/01_environnement/00_guide_installation_environnement.md")
        print("(creation du Space, activation de Dev Mode, connexion SSH/VSCode).")
    else:
        print("Des elements sont manquants ou incorrects : voir le detail ci-dessus.")
        sys.exit(1)


if __name__ == "__main__":
    main()
