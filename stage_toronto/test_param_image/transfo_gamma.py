import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger l'image
image_originale = cv2.imread('goutte1.png')
image_rgb = cv2.cvtColor(image_originale, cv2.COLOR_BGR2RGB)


# Fonction pour appliquer la correction Gamma via une LUT
def appliquer_gamma(image, gamma):
    # Création de la Look-Up Table (table de correspondance)
    # Pour chaque valeur de 0 à 255, on applique la formule Power-Law
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # cv2.LUT applique le tableau à toute l'image instantanément
    return cv2.LUT(image, table)


# 2. Appliquer deux transformations différentes
# Gamma < 1 : Étire les teintes sombres (éclaircit l'image et révèle les ombres)
gamma_clair = 0.4
image_eclaircie = appliquer_gamma(image_rgb, gamma_clair)

# Gamma > 1 : Compresse les teintes sombres (assombrit l'image et booste le contraste des hautes lumières)
gamma_sombre = 2.5
image_assombrie = appliquer_gamma(image_rgb, gamma_sombre)

# 3. Préparer l'affichage comparatif (3 fenêtres)
plt.figure(figsize=(15, 5))

# Image d'origine
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Image Original\n(Gamma = 1.0)")
plt.axis('off')

# Image éclaircie
plt.subplot(1, 3, 2)
plt.imshow(image_eclaircie)
plt.title(f"Transformation Power-Law\n(Gamma = {gamma_clair} -> Revels shadows)")
plt.axis('off')

# Image assombrie
plt.subplot(1, 3, 3)
plt.imshow(image_assombrie)
plt.title(f"Transformation Power-Law\n(Gamma = {gamma_sombre} -> Isolates reflects)")
plt.axis('off')

# 4. Afficher
plt.tight_layout()
plt.show()