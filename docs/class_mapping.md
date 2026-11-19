# Correspondance des classes

| id | Classe CholeGuard | Endoscapes2023 | CholecSeg8k |
|---|---|---|---|
| 0 | fond | (non annoté) | Black Background, Abdominal Wall, Liver, GI Tract, Fat, Connective Tissue, Blood, Hepatic Vein, Liver Ligament |
| 1 | outil | Tool | Grasper, L-hook Electrocautery |
| 2 | vésicule | Gallbladder | Gallbladder |
| 3 | canal cystique | Cystic Duct | Cystic Duct |
| 4 | artère cystique | Cystic Artery | — |
| 5 | triangle hépatocystique | Hepatocystic Triangle Dissection | — |
| 6 | plaque cystique | Cystic Plate | — |

Le foie n'est pas annoté dans Endoscapes : il est classé en fond pour éviter des étiquettes contradictoires entre les deux jeux de données.
CholecSeg8k ne sert qu'au pré-entraînement (outils, vésicule, canal) ; le modèle final est affiné sur Endoscapes, seule source des classes 4 à 6.
