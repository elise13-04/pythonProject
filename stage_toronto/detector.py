import cv2
import numpy as np
import time


# --- 1. LE MOTEUR DE TRACKING ---
class SuiviGouttes:
    # ATTENTION : La distance max passe de 20 à 150 pour autoriser les grands sauts d'une seconde à l'autre
    def __init__(self, distance_max=150):
        self.prochain_id = 0
        self.objets = {}
        self.couleurs = {}
        self.distance_max = distance_max

    def enregistrer(self, centre):
        self.objets[self.prochain_id] = centre
        self.couleurs[self.prochain_id] = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255))
        )
        self.prochain_id += 1

    def mettre_a_jour(self, centres_detectes):
        if len(centres_detectes) == 0:
            self.objets.clear()
            self.couleurs.clear()
            return self.objets

        if len(self.objets) == 0:
            for i in range(len(centres_detectes)):
                self.enregistrer(centres_detectes[i])
        else:
            ids_objets = list(self.objets.keys())
            centres_objets = list(self.objets.values())

            D = np.linalg.norm(np.array(centres_objets)[:, np.newaxis] - np.array(centres_detectes), axis=2)

            lignes = D.min(axis=1).argsort()
            colonnes = D.argmin(axis=1)[lignes]

            lignes_utilisees = set()
            colonnes_utilisees = set()

            for ligne, colonne in zip(lignes, colonnes):
                if ligne in lignes_utilisees or colonne in colonnes_utilisees:
                    continue

                if D[ligne, colonne] > self.distance_max:
                    continue

                id_objet = ids_objets[ligne]
                self.objets[id_objet] = centres_detectes[colonne]

                lignes_utilisees.add(ligne)
                colonnes_utilisees.add(colonne)

            lignes_non_utilisees = set(range(D.shape[0])).difference(lignes_utilisees)
            for ligne in lignes_non_utilisees:
                id_objet = ids_objets[ligne]
                del self.objets[id_objet]
                del self.couleurs[id_objet]

            colonnes_non_utilisees = set(range(D.shape[1])).difference(colonnes_utilisees)
            for colonne in colonnes_non_utilisees:
                self.enregistrer(centres_detectes[colonne])

        return self.objets


# --- 2. LE MOTEUR DE DÉTECTION ULTIME ---
def extraire_centres(frame, w):
    image_grise = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img_clahe = clahe.apply(image_grise)
    blur = cv2.medianBlur(img_clahe, 5)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        55, 8
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 5)

    kernel_nms = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dist_dilated = cv2.dilate(dist_transform, kernel_nms)

    seuil = 0.15 * dist_transform.max()
    pics = (dist_transform == dist_dilated) & (dist_transform > seuil)

    y_coords, x_coords = np.where(pics)

    limite_gauche = int(w * 0.38)
    limite_droite = int(w * 0.85)

    centres_valides = []

    for y, x in zip(y_coords, x_coords):
        if x < limite_gauche or x > limite_droite:
            rayon_estime = dist_transform[y, x]
            if 15 < rayon_estime < 40:
                centres_valides.append((int(x), int(y)))

    return centres_valides


# --- 3. LE GESTIONNAIRE DE VIDÉO (MODIFIÉ 1 FPS) ---
def analyser_video_1fps(chemin_entree, chemin_sortie):
    cap = cv2.VideoCapture(chemin_entree)

    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo {chemin_entree}")
        return

    largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = int(cap.get(cv2.CAP_PROP_FPS))
    nombre_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"--- DÉBUT DU TRACKING (MODE 1 FPS) ---")
    print(f"Vidéo source : {fps_source} FPS")
    print(f"La vidéo finale sera un diaporama à 1 FPS.")

    # La vidéo de sortie est forcée à 1 image par seconde
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(chemin_sortie, fourcc, 1, (largeur, hauteur))

    tracker = SuiviGouttes(distance_max=150)

    temps_debut = time.time()

    # On commence à la frame 0
    frame_index = 0
    images_generees = 0

    while frame_index < nombre_total_frames:
        # L'astuce magique : On ordonne au lecteur de "sauter" directement à la frame ciblée
        # Cela évite à votre processeur de lire inutilement les 59 frames intermédiaires
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            break

        # Traitement
        centres_actuels = extraire_centres(frame, largeur)
        objets_suivis = tracker.mettre_a_jour(centres_actuels)

        # Dessin
        for id_objet, centre in objets_suivis.items():
            couleur_specifique = tracker.couleurs[id_objet]
            cv2.drawMarker(
                frame,
                centre,
                color=couleur_specifique,
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=3
            )

        # Écriture dans le fichier final
        out.write(frame)

        images_generees += 1
        print(f"Seconde {images_generees} analysée...")

        # On avance le "curseur" de l'équivalent du framerate (ex: on passe de la frame 0 à 60, puis 120...)
        frame_index += fps_source

    cap.release()
    out.release()

    temps_total = time.time() - temps_debut
    print(f"--- TRACKING TERMINÉ ! ---")
    print(f"Vidéo sauvegardée sous : {chemin_sortie}")
    print(f"Temps de calcul : {temps_total:.2f} secondes.")

#Lancement du code (remplacez par vos noms de fichiers)
analyser_video_1fps('Video1_1.mp4', 'Video1_1_cross.mp4')