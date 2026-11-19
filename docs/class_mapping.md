# Correspondance des classes

| id | Classe CholeGuard | Endoscapes2023 (id semseg brut) | CholecSeg8k |
|---|---|---|---|
| 0 | fond | 0 background (255 = non annoté, ignoré) | Black Background, Abdominal Wall, Liver, GI Tract, Fat, Connective Tissue, Blood, Hepatic Vein, Liver Ligament |
| 1 | outil | 6 tool | Grasper, L-hook Electrocautery |
| 2 | vésicule | 5 gallbladder | Gallbladder |
| 3 | canal cystique | 4 cystic_duct | Cystic Duct |
| 4 | artère cystique | 3 cystic_artery | — |
| 5 | triangle hépatocystique | 2 calot_triangle | — |
| 6 | plaque cystique | 1 cystic_plate | — |

Le foie n'est pas annoté dans Endoscapes : il est classé en fond pour éviter des étiquettes contradictoires entre les deux jeux de données.
CholecSeg8k ne sert qu'au pré-entraînement (outils, vésicule, canal) ; le modèle final est affiné sur Endoscapes, seule source des classes 4 à 6.
