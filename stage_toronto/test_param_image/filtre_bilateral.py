import cv2
import matplotlib.pyplot as plt

# 1. Charger l'image originale
image_originale = cv2.imread('goutte1.png')

# Convertir BGR (OpenCV) en RGB (Matplotlib)
image_rgb = cv2.cvtColor(image_originale, cv2.COLOR_BGR2RGB)

# 2. Appliquer le Filtre Bilatéral
# Paramètres du filtre bilatéral :
# - image_rgb : l'image source
# - d (Diamètre) : Taille du voisinage (ex: 9). Plus c'est grand, plus c'est lent.
# - sigmaColor : Force du lissage des couleurs. Plus c'est grand, plus des couleurs lointaines seront mélangées.
# - sigmaSpace : Force du lissage spatial. Plus c'est grand, plus les pixels éloignés s'influencent (si leurs couleurs sont proches).
image_bilaterale = cv2.bilateralFilter(image_rgb, d=9, sigmaColor=75, sigmaSpace=75)

# 3. Préparer l'affichage comparatif
plt.figure(figsize=(12, 6))

# Fenêtre de gauche
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Image Original")
plt.axis('off')

# Fenêtre de droite
plt.subplot(1, 2, 2)
plt.imshow(image_bilaterale)
plt.title("Bilateral Filter\n(Internal smoothening, edges conserved)")
plt.axis('off')

# 4. Afficher
plt.tight_layout()
plt.show()