# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:12:16 2025

@author: Ngatam
"""
################################# Import ######################################

import os
import cv2
import numpy as np


################################# Script #######################################


# Création du modèle 3D
## Coordonnées 3D réels des 8 points de la boite
objPoints = np.array([[0, 90, 0],
                      [125, 90, 0],
                      [125, 90, 70],
                      [0, 90, 70],
                      [0, 0, 0],
                      [125, 0, 0],
                      [125, 0, 70],
                      [0, 0, 70]], np.float32)
objPoints.reshape((-1, 1, 3))*1e-3

# Calibration de la camera
cameraMatrix = np.asarray([[606.209, 0,         320.046],
                           [0,       0606.719,  238.926],
                           [0,       0,         1]], np.float32)


distCoeffs = np.asarray([[0, 0, 0, 0, 0]], np.float32)

rvec = np.eye(3)

# Calcul des coordonnées des points 3 dans le repère camera
imagePoints = cv2.projectPoints(objPoints, rvec, distCoeffs, cameraMatrix)

# Définition des 4 points de la boite pour le tracking -> Points 3, 4, 7 et 8
point_choisis = [objPoints[2], objPoints[3], objPoints[6], objPoints[7]]


# Chemin vers la vidéo (remplace ce chemin par le tien)
video_path = 'C:/Users/Administrator/Desktop/Cours 3ème année/Mécatronique/Traitement_d_image/TP_KLT_PNP/box_video_data.avi'

# Dossier de sortie pour les frames
output_folder = 'frames_video'

# Créer le dossier s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    new_points, statut, erreur=  cv2.calcOpticalFlowPyrLK(point_choisis) 
    
    if not ret:
        break  # Fin de la vidéo

    # Sauvegarder la frame
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

print(f"{frame_count} frames ont été extraites et sauvegardées dans le dossier '{output_folder}'.")

cap.release()


print('Chaque image a été sauvegardée dans le dossier frame')

print('Vidéo lue avec succès!')

## Libérez l'objet VideoCapture
cap.release()