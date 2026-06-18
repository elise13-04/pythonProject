from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# 1. Charger l'image originale avec Pillow
# Remplacez 'gouttes.jpg' par le nom de votre fichier
img_original = Image.open('goutte1.png')

# 2. Appliquer le Flou Gaussien
# Le paramètre 'radius' définit la force du flou.
# Plus le chiffre est grand, plus l'image sera floue.
img_blur = img_original.filter(ImageFilter.GaussianBlur(radius=1))

# 3. Préparer l'affichage comparatif
plt.figure(figsize=(12, 6))

# Fenêtre de gauche : Image d'origine
plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.title("Image Original")
plt.axis('off')

# Fenêtre de droite : Image floutée
plt.subplot(1, 2, 2)
plt.imshow(img_blur)
plt.title("Blur Gaussien (Pillow, radius=1)")
plt.axis('off')

# 4. Afficher le résultat
plt.tight_layout()
plt.show()