import cv2
import numpy as np
import time


# --- 1. LE MOTEUR DE DÉTECTION (Notre version Ultime) ---
def traiter_frame(frame):
    # La frame arrive en couleur (BGR), on la passe en niveaux de gris pour l'analyse
    image_grise = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = image_grise.shape

    # Prétraitement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    img_clahe = clahe.apply(image_grise)
    blur = cv2.medianBlur(img_clahe, 5)

    # Binarisation Sévère
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        55, 8
    )

    # Nettoyage Morphologique
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Transformée de Distance
    dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 5)

    # Filtre Anti-Doublons (Cercle 17x17)
    kernel_nms = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dist_dilated = cv2.dilate(dist_transform, kernel_nms)

    seuil = 0.15 * dist_transform.max()
    pics = (dist_transform == dist_dilated) & (dist_transform > seuil)

    y_coords, x_coords = np.where(pics)

    # Limites (à ajuster si la caméra bouge dans la vidéo)
    limite_gauche = int(w * 0.38)
    limite_droite = int(w * 0.85)

    # On dessine directement sur la frame originale en couleur
    for y, x in zip(y_coords, x_coords):
        if x < limite_gauche or x > limite_droite:
            rayon_estime = dist_transform[y, x]

            if 15 < rayon_estime < 40:
                cv2.drawMarker(
                    frame,
                    (x, y),
                    color=(0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=3
                )

    return frame


# --- 2. LE GESTIONNAIRE DE VIDÉO ---
def analyser_video(chemin_entree, chemin_sortie):
    # Ouvrir la vidéo source
    cap = cv2.VideoCapture(chemin_entree)

    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo {chemin_entree}")
        return

    # Récupérer les caractéristiques de la vidéo source
    largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    nombre_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"--- DÉBUT DE L'ANALYSE ---")
    print(f"Résolution : {largeur}x{hauteur}")
    print(f"Framerate : {fps} FPS")
    print(f"Total des frames à traiter : {nombre_total_frames}")

    # Préparer le fichier de sortie (Format MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(chemin_sortie, fourcc, fps, (largeur, hauteur))

    temps_debut = time.time()
    frames_traitees = 0

    # Boucle de lecture frame par frame
    while True:
        ret, frame = cap.read()

        # Si 'ret' est False, la vidéo est terminée
        if not ret:
            break

        # 1. Analyser la frame et dessiner les croix
        frame_annotee = traiter_frame(frame)

        # 2. Écrire la frame dans la nouvelle vidéo
        out.write(frame_annotee)

        # 3. Afficher l'avancement dans la console
        frames_traitees += 1
        if frames_traitees % 60 == 0:  # Mise à jour toutes les 60 frames (1 seconde de vidéo)
            print(f"Progression : {frames_traitees}/{nombre_total_frames} frames traitées...")

    # Libérer la mémoire et fermer les fichiers
    cap.release()
    out.release()

    temps_total = time.time() - temps_debut
    print(f"--- TERMINÉ ! ---")
    print(f"Vidéo sauvegardée sous : {chemin_sortie}")
    print(f"Temps de traitement : {temps_total:.2f} secondes.")

# Lancement du code (remplacez par vos noms de fichiers)
analyser_video('Video1_1.mp4', 'Video1_1_cross.mp4')