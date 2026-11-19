# Choix du backbone — vitesse CPU (ONNX Runtime, 2 cœurs, poids aléatoires)

| Backbone | 256×448 | 288×512 | Paramètres |
|---|---|---|---|
| mit_b2 (SegFormer) | 3,0 img/s | 2,3 img/s | 27 M |
| convnext_tiny | 3,0 img/s | 2,3 img/s | 31 M |
| efficientnet-b2 | **7,8 img/s** | 5,6 img/s | 10 M |
| mit_b0 (SegFormer) | **7,3 img/s** | 5,7 img/s | 5,5 M |

Mesuré sur une machine de 2 cœurs ; un PC portable à 4-8 cœurs est environ 2 à 3 fois plus rapide.
Seuls efficientnet-b2 et mit_b0 respectent l'objectif de ≥ 5 img/s sur 2 cœurs à 256×448.
Le choix final se fait sur la précision en validation (Kaggle) entre ces deux candidats.
