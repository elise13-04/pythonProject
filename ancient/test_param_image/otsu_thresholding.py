import cv2
import matplotlib.pyplot as plt

# 1. Charger l'image originale
image_originale = cv2.imread('goutte1.png')

# Convertir en RGB pour l'affichage visuel d'origine
image_rgb = cv2.cvtColor(image_originale, cv2.COLOR_BGR2RGB)

# Convertir en niveaux de gris (Obligatoire pour Otsu)
image_grise = cv2.cvtColor(image_originale, cv2.COLOR_BGR2GRAY)

# 2. Appliquer le Thresholding d'Otsu
# Paramètres :
# - image_grise : L'image source en niveaux de gris
# - 0 : La valeur de seuil manuelle (ici ignorée par OpenCV grâce à l'option OTSU)
# - 255 : La valeur à donner aux pixels qui dépassent le seuil (ici, blanc pur)
# - cv2.THRESH_BINARY + cv2.THRESH_OTSU : Demande à OpenCV de calculer le seuil tout seul
valeur_seuil_optimale, image_otsu = cv2.threshold(
    image_grise,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 3. Préparer l'affichage comparatif
plt.figure(figsize=(12, 6))

# Fenêtre de gauche : Image d'origine
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Image Original\n(Pixels with complex nuances)")
plt.axis('off')

# Fenêtre de droite : Binarisation d'Otsu
plt.subplot(1, 2, 2)
# On utilise cmap='gray' car l'image est maintenant purement noire et blanche
plt.imshow(image_otsu, cmap='gray')
plt.title(f"Otsu binarisation\n(Threshold automatically calculated : {valeur_seuil_optimale})")
plt.axis('off')

# 4. Afficher le résultat
plt.tight_layout()
plt.show()

print(f"value for optimal threshold : {valeur_seuil_optimale}")