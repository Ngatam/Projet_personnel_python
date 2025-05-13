# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:12:16 2025

@author: Ngatam
"""
################################# Import ######################################


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
cameraMatrix = np.asarray([[606.209, 0, 320.046],
                           [0, 0606.719, 238.926],
                           [0, 0, 1]], np.float32)

distCoeffs = np.asarray([[0, 0, 0, 0, 0]], np.float32)

# Définition des 4 points de la boite pour le tracking -> Points 3, 4, 7 et 8
point_choisis = [objPoints[2], objPoints[3], objPoints[6], objPoints[7]]

# Charger la vidéo dans la variable video
video = cv2.VideoCapture('box_video_data.mp4')
 
# Boucle infinie pour lecture
while (True):
	# Stoquer l'image issue de la vidéo à l'instant t dans la variable "frame"
	ret, frame = video.read()
    
	# Afficher l'image contenue dans "frame"
	cv2.imshow('output', frame)
    
    #cv2.calcOpticalFlowPyrLK() 
    
	# Quiter la boucle infinie lorqu'on appuie sur la touche 'q'
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break
 
# Quiter le programme et fermer toutes les fenêtres ouvertes
video.release()
cv2.destroyAllWindows()