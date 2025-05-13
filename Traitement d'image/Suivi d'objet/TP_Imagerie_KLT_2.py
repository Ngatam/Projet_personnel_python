# -*- coding: utf-8 -*-
"""
Created on Fr Mar 25 16:20:22 2025

@author: Ngatam
"""
################################# Import ######################################

import os
import cv2
import numpy as np



################################# Script ######################################

#### Passage coordonées 3D vers coordonnées repère camera
# Coordonnées 3D des coins de la boîte
objPoints = np.array([[0, 90, 0],
                      [125, 90, 0],
                      [125, 90, 70],
                      [0, 90, 70],
                      [0, 0, 0],
                      [125, 0, 0],
                      [125, 0, 70],
                      [0, 0, 70]], np.float32)


# Calibration caméra
cameraMatrix = np.asarray([[606.209, 0, 320.046],
                           [0, 606.719, 238.926],
                           [0, 0, 1]], np.float32)
distCoeffs = np.zeros((5, 1))


# Vecteurs de rotation et translation initiaux
rvec = np.zeros((3, 1))
tvec = np.zeros((3, 1))

# Projection des coins 3D en 2D
imagePoints, _ = cv2.projectPoints(objPoints, rvec, tvec, cameraMatrix, distCoeffs) # Calcul de la matrice de projection
imagePoints = imagePoints.reshape(-1, 1, 2)
indices = [2, 3, 6, 7]  # Indices des coins choisis
p0 = imagePoints[indices] # Recherches des coordonnées dans le répère caméra des points choisis
print("p0 =  ", p0)


#### Lecture de la video
# Chemin vidéo
video_path = 'C:/Users/Administrator/Desktop/Cours 3ème année/Mécatronique/Traitement_d_image/TP_KLT_PNP/box_video_data.avi'
cap = cv2.VideoCapture(video_path)

# Message d'erreur sur on ne peut pas ouvrir la vidéo
if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo.")
    exit()

# Créer un dossier pour les frames sauvegardées
output_folder = 'frames_video'
os.makedirs(output_folder, exist_ok=True)

# Lire la première frame
ret, old_frame = cap.read()
if not ret:
    print("Erreur : impossible de lire la première frame.")
    cap.release()
    exit()

# Passage sous N&B
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


####  Passage coordonées 3D vers coordonnées repère camera
# Paramètres KLT
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

frame_count = 0
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
    print("\np1 =  ", p1)

    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for (new, old) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        
    else:
        img = frame
    
    # Affichage de l'erreur
    print("\nerreur = ", err)

    # Mise à jour pour prochaine frame
    p0 = p1
    old_gray = frame_gray.copy()

    # Affichage dans une fenêtre
    cv2.imshow('Tracking des coins', img)

    # Sauvegarder l'image
    filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(filename, img)

    # Quitter si on appuie sur 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"{frame_count} frames traitées et enregistrées dans '{output_folder}'.")
