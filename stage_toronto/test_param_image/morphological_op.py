import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger l'image directement en niveaux de gris
image_originale = cv2.imread('goutte1.png', cv2.IMREAD_GRAYSCALE)

# Binarisation avec Otsu (les gouttes/objets deviennent blancs, le fond noir)
_, image_binaire = cv2.threshold(image_originale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2. Définir le "Noyau" (Kernel)
# C'est la forme géométrique qui va parcourir l'image.
# Ici, on crée un carré plein de 5x5 pixels.
noyau = np.ones((5, 5), np.uint8)

# 3. Appliquer l'Érosion
# "Grignote" les bords extérieurs des zones blanches.
image_erodee = cv2.erode(image_binaire, noyau, iterations=1)

# 4. Appliquer la Dilatation
# "Étend" les bords extérieurs des zones blanches.
image_dilatee = cv2.dilate(image_binaire, noyau, iterations=1)

# 5. Préparer l'affichage comparatif (3 fenêtres)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_binaire, cmap='gray')
plt.title("Binarized image (Otsu)\n(Some holes and noises are presents)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_erodee, cmap='gray')
plt.title("Opening (Iterations=1)\n(Noise disappears, drops are shrinking)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_dilatee, cmap='gray')
plt.title("Dilation (Itérations=1)\n(Holes are filing, drops are growing)")
plt.axis('off')

plt.tight_layout()
plt.show()