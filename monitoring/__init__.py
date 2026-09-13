"""
Paquet du dashboard Streamlit de suivi d'entrainement EN VIVO.

Vit hors `training/` (pas un pas execute par le job d'entrainement,
c'est un visualiseur passif deploye a part sur un Space Streamlit) et
hors `interfaces/cli/` (pas une CLI de la sequence de cas d'usage
E1/E2) : meme critere que `domain/`/`ports/`/`infrastructure/adapters/`,
d'ou l'absence de prefixe `E1_`/`E2_` sur les fichiers de ce paquet.
"""

from __future__ import annotations
