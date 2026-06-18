import cv2
import matplotlib.pyplot as plt

# 1. Charger l'image originale
# OpenCV la lit par défaut en couleur (format BGR : Bleu, Vert, Rouge)
image_originale = cv2.imread('goutte1.png')

# Convertir BGR en RGB pour que les couleurs s'affichent correctement dans Matplotlib
image_rgb = cv2.cvtColor(image_originale, cv2.COLOR_BGR2RGB)

# 2. Appliquer le Grayscaling (Niveaux de gris)
# On convertit l'image BGR originale en niveaux de gris
image_grise = cv2.cvtColor(image_originale, cv2.COLOR_BGR2GRAY)

# 3. Préparer l'affichage comparatif
plt.figure(figsize=(12, 6))

# Fenêtre de gauche : Image d'origine
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Image Originale (3 canaux : RGB)\nDonnées lourdes")
plt.axis('off')

# Fenêtre de droite : Image en niveaux de gris
plt.subplot(1, 2, 2)
# ATTENTION : Avec Matplotlib, il faut préciser cmap='gray'
# sinon il affichera l'image avec un filtre jaune/bleu par défaut
plt.imshow(image_grise, cmap='gray')
plt.title("Grayscaling (1 canal : Intensité lumineuse)\nCalculs 3x plus rapides")
plt.axis('off')

# 4. Afficher le résultat
plt.tight_layout()
plt.show()