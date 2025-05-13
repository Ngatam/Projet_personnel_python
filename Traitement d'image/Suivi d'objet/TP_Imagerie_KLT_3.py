# -*- coding: utf-8 -*-
"""
Created on Fr Apr 25 16:20:22 2025

@author: Ngatam
"""

######################## Import ###############################################

import cv2
import os
import numpy as np

######################## Paramètres vidéos et sorties #########################

video_path = 'C:/Users/Administrator/Desktop/Cours 3ème année/Mécatronique/Traitement_d_image/TP_KLT_PNP/box_video_data.avi'
output_folder = 'frames_video'
os.makedirs(output_folder, exist_ok=True) # Creer le document de dépose des fichiers

######################## Clic de la souris ####################################

points_choisis = []
drawing = True

def click_event(event, x, y, flags, param):
    
    # Ajoute un point lors du clic gauche
    global points_choisis, drawing
    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        points_choisis.append((x, y)) # Ajouts des points choisis
        print(f"Point sélectionné : ({x}, {y})") 

######################## Chargement de la vidéo ###############################

cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    print("Erreur : impossible de lire la vidéo.")
    cap.release()
    exit()

first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) # Lit à la 1ère frame en N&B

################## Sélection des points par la souris #########################

cv2.namedWindow('Sélectionne les points (clic gauche)') # Fenêtre de sélection des points de suivis
cv2.setMouseCallback('Sélectionne les points (clic gauche)', click_event)

temp_frame = first_frame.copy()

print("Clique sur les points à suivre, puis appuie sur 's' pour démarrer.")
while True:
    for pt in points_choisis:
        cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
    cv2.imshow('Sélectionne les points (clic gauche)', temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # Si on appuie sur s on arrête la sélection
        drawing = False
        break

cv2.destroyWindow('Sélectionne les points (clic gauche)')

################## Initialisation #############################################

p0 = np.array(points_choisis, dtype=np.float32).reshape(-1, 1, 2) # Vecteurs des points choisis

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

old_gray = first_frame_gray.copy()
mask = np.zeros_like(first_frame) # Création d'un masque 
frame_idx = 0

############################### Script ########################################

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Frame active

    # Optical Flow LK
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) # Suivi des points

    if p1 is not None and st is not None: # Si pas de nouveaux points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Tracer les lignes de mouvement
        for (new, old) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Relier les points entre eux
        if len(good_new) >= 2:
            for i in range(len(good_new)):
                pt1 = tuple(np.int32(good_new[i % len(good_new)].ravel()))
                pt2 = tuple(np.int32(good_new[(i + 1) % len(good_new)].ravel()))
                frame = cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Ligne bleue entre les points

        img = cv2.add(frame, mask)

        # Mise à jour pour l’itération suivante
        p0 = good_new.reshape(-1, 1, 2)
        old_gray = frame_gray.copy()
   
    else:
        img = frame  # Aucun point suivi

    # Afficher et sauvegarder
    cv2.imshow('Tracking', img)
    frame_name = os.path.join(output_folder, f'frame_{frame_idx:04d}.jpg')
    cv2.imwrite(frame_name, img)
    frame_idx += 1

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
print(f"{frame_idx} images traitées et sauvegardées.")
