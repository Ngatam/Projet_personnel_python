# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:44:00 2025

@author: Administrator
"""

################################# Import ######################################

import numpy as np  # Manipulation de tableaux numériques
import matplotlib.pyplot as plt  # Affichage de courbes et d’images
import random  # Tirage d’échantillons aléatoires
from sklearn.model_selection import train_test_split  # Séparation train/test
from tensorflow.keras.models import Sequential  # Modèle séquentiel (empilement de couches)
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     Flatten, Dense, Dropout)  # Couches CNN
from tensorflow.keras.optimizers import RMSprop  # Optimiseur pour la compilation



################################# Chargement des données ######################################

images = np.load("data_2_Patches_RGB_list.npy")  # Images RGB de taille 50x50
coords = np.load("data_2_Patches_3D_list.npy").reshape(-1, 3)  # Coordonnées (X, Y, Z)

# Affichage de quelques infos sur les données
print("Taille des images :", images.shape)  # Dimensions (nb_images, 50, 50, 3)
print("Type de données :", images.dtype)    # Généralement uint8 avant normalisation



################################# Visualisationdes données ######################################

indices = random.sample(range(images.shape[0]), 3)  # Tirage de 3 indices
for idx in indices:
    plt.figure()
    plt.imshow(images[idx])  # Affichage image RGB
    plt.title(f"Patch #{idx}")
    plt.axis('off')
    plt.show()



################################# Script ######################################
images = images.astype('float32') / 255.0  # Normalisation des pixels entre 0 et 1
X_train, X_val, y_train, y_val = train_test_split(images, coords, test_size=0.2, random_state=42)  # 80% train, 20% val

# Construction du réseau CNN
model = Sequential()  # Initialisation du modèle séquentiel

# Boucle pour empiler 5 blocs convolutionnels
for i, filtres in enumerate([16, 32, 64, 128, 256]):
    if i == 0:
        # Première couche : spécifier la taille des entrées (50x50x3)
        model.add(Conv2D(filtres, (3, 3), activation='relu', padding='same', input_shape=(50, 50, 3)))
    else:
        # Couches suivantes : même taille de filtre
        model.add(Conv2D(filtres, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())  # Normalisation pour stabilité de l’apprentissage
    model.add(MaxPooling2D((2, 2)))  # Réduction spatiale de moitié

# Passage en couche dense
model.add(Flatten())  # Aplatissement du tenseur 3D -> 1D pour couche dense
model.add(Dense(512, activation='relu'))  # Couche dense avec 512 neurones
model.add(Dropout(0.5))  # Dropout pour éviter le sur-apprentissage
model.add(Dense(256, activation='relu'))  # Deuxième couche dense
model.add(Dropout(0.5))
model.add(Dense(3))  # Couche de sortie : 3 valeurs continues (X, Y, Z)

# Compilation du modèle
model.compile(optimizer=RMSprop(), loss='mse', metrics=['mae'])  # MAE = Mean Absolute Error

# Entraînement du modèle
historique = model.fit(
    X_train, y_train, # Données d’entrée et cibles
    validation_data=(X_val, y_val), # Données de validation
    batch_size=32, # Taille des lots
    epochs=50, # Nombre d’époques d’apprentissage
    verbose=1 # Affichage détaillé
)

# Évaluation finale
score = model.evaluate(X_val, y_val, verbose=0)  # Évaluation du modèle sur données non vues
print(f"Erreur absolue moyenne sur test : {score[1]:.4f}")  # Affiche la MAE

# Affichage de l'évolution de la perte
plt.figure()
plt.plot(historique.history['loss'], label="Perte entraînement", color='darkred')
plt.plot(historique.history['val_loss'], label="Perte validation", color='navy')
plt.title("Courbe d’apprentissage - Erreur quadratique moyenne")
plt.xlabel("Époques")
plt.ylabel("Erreur (MSE)")
plt.legend()
plt.grid(True)
plt.show()
