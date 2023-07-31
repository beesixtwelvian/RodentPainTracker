import os
import sys
import openpyxl
import subprocess
import re
import shutil
import numpy as np
import pandas as pd# pour les dataframes
from scipy import interpolate# pour interpolate
import math
import time# pour la pause et pour la barre de progression
import csv# pour exports en csv
import datetime
from tqdm import tqdm# pour la barre de progression
# Spline
#from scipy.interpolate import splprep, splev
#from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
# SVM
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Regression logistique
from sklearn.linear_model import LogisticRegression
# KMeans
from sklearn.cluster import KMeans
#
import matplotlib.pyplot as plt

from Spline.Spline import Spline

# La fonction YOLOanalyze prend en paramètres :
# - 1er, PathSubFolderToAnalyze est le media d'entrée (peut être une image, une vidéo, un répertoire fichier image ou vidéo, en répertoire dossier contenant des images ou vidéos
# - 2ème, sensibilityrate est le seuil, la sensibilité, de détection, choisi
# - 3ème, IOU Intersection Over Union, s'il est True, supprime les boîtes superposées afin de ne détecter que un objet
# - 4ème, eachframe, si le fichier est une vidéo, n'applique le YOLO que toutes les tant de frames
def YoloBodyCsvToCleanedCsv(PathSubFolderToAnalyze):
    # Affichage utilisateur
    print(" └Exécution de YoloBodyCsvToCleanedCsv ... ")
    #
    # Liste les fichiers dans le sous-dossier, pour trouver *.body.csv
    for File in os.listdir(PathSubFolderToAnalyze):
        PathFile = os.path.join(PathSubFolderToAnalyze, File)
        # Vérifier si le fichier se termine par ".body.csv"
        if os.path.isfile(PathFile) and File.endswith(".body.csv"):
            if 1 == 1:# Ajouter des lignes vides pour chaque frame manquante
                df = pd.read_csv(PathFile, sep=';')  # Copie du .CSV dans un dataframe
                # Créer un nouveau dataframe pour stocker les données mises à jour
                new_df = pd.DataFrame(columns=df.columns)
                # Trouver les frames manquantes pour chaque groupe 'file' et les ajouter dans le nouveau dataframe
                for file_group, file_df in df.groupby('file'):
                    # Trouver les frames manquantes pour le groupe 'file' actuel
                    max_frame = file_df['localframe'].max()
                    all_frames = pd.Series(range(1, max_frame + 1))
                    missing_frames = all_frames[~all_frames.isin(file_df['localframe'])]
                    # Créer un dataframe contenant les lignes manquantes
                    missing_rows = pd.DataFrame({'localframe': missing_frames})
                    missing_rows['file'] = file_group  # Ajouter la colonne 'file' correspondante
                    # Concaténer le dataframe des lignes manquantes avec le dataframe initial pour le groupe 'file'
                    updated_file_df = pd.concat([file_df, missing_rows])
                    # Réorganiser les lignes par ordre croissant de frame
                    updated_file_df = updated_file_df.sort_values('localframe').reset_index(drop=True)
                    # Ajouter les données du groupe 'file' dans le nouveau dataframe
                    new_df = pd.concat([new_df, updated_file_df])
                # Trier
                new_df = new_df.sort_values(['file', 'localframe'])
                # Réinitialisation des index du nouveau dataframe
                new_df.reset_index(drop=True, inplace=True)
                # Réécrire dans le .CSV
                new_df.to_csv(PathFile, index=False, mode='w', sep=';')
            if 1 == 1:# Interpolation de X, Y, W, H
                df = pd.read_csv(PathFile, sep=';')# Copie du .CSV dans un dataframe
                if 'X' in df.columns:
                    df['X'] = (df['X'].interpolate(method='linear', limit_direction='both', axis=0)).astype(int)#.interpolate(method='spline', order=3, inplace=True)
                if 'Y' in df.columns:
                    df['Y'] = (df['Y'].interpolate(method='linear', limit_direction='both', axis=0)).astype(int)#.interpolate(method='spline', order=3, inplace=True)
                if 'W' in df.columns:
                    df['W'] = (df['W'].interpolate(method='linear', limit_direction='both', axis=0)).astype(int)#.interpolate(method='spline', order=3, inplace=True)
                if 'H' in df.columns:
                    df['H'] = (df['H'].interpolate(method='linear', limit_direction='both', axis=0)).astype(int)#.interpolate(method='spline', order=3, inplace=True)
                # localframe, remplacement des valeurs manquantes en remontant
                prev_value = None
                for i in range(len(df['localframe']) - 1, -1, -1):
                    value = df.at[i, 'localframe']
                    if pd.isna(value):
                        if prev_value is not None:
                            df.at[i, 'localframe'] = prev_value - 1
                        else:
                            df.at[i, 'localframe'] = None
                    else:
                        prev_value = value
                df['localframe'] = df['localframe'].astype(int)
                # object, remplacement des valeurs manquantes en remontant
                prev_value = None
                for i in range(len(df['object']) - 1, -1, -1):
                    value = df.at[i, 'object']
                    if pd.isna(value):
                        if prev_value is not None:
                            df.at[i, 'object'] = prev_value
                        else:
                            df.at[i, 'object'] = None
                    else:
                        prev_value = value
                # file, remplacement des valeurs manquantes en remontant
                prev_value = None
                for i in range(len(df['file']) - 1, -1, -1):
                    value = df.at[i, 'file']
                    if pd.isna(value):
                        if prev_value is not None:
                            df.at[i, 'file'] = prev_value
                        else:
                            df.at[i, 'file'] = None
                    else:
                        prev_value = value
                # folder, remplacement des valeurs manquantes en remontant
                prev_value = None
                for i in range(len(df['folder']) - 1, -1, -1):
                    value = df.at[i, 'folder']
                    if pd.isna(value):
                        if prev_value is not None:
                            df.at[i, 'folder'] = prev_value
                        else:
                            df.at[i, 'folder'] = None
                    else:
                        prev_value = value
                # Réécrire dans le .CSV
                df.to_csv(PathFile, index=False, mode='w', sep=';')
            if 1 == 1:# Création de Xcenter Ycenter R et arrondi de confidence
                df = pd.read_csv(PathFile, sep=';')# Copie du .CSV dans un dataframe
                # Création de Xcenter
                df['Xcenter'] = None
                df['Xcenter'] = (df['X'] + df['W'] / 2).astype(int)
                # Création de Ycenter
                df['Ycenter'] = None
                df['Ycenter'] = (df['Y'] + df['H'] / 2).astype(int)
                # Création de R
                df['R'] = None
                for index, row in df.iterrows():
                    cotemax = int(max(row['W'], row['H']))
                    cotemin = int(min(row['W'], row['H']))
                    hypotenuse = int(math.sqrt(cotemax**2 + cotemin**2))
                    rayon = (hypotenuse / 2)*(9/8)
                    rayon = int(rayon)
                    df.at[index, 'R'] = rayon
                # Arrondi de confidence
                if 'confidence' in df.columns:
                    df['confidence'] = df['confidence'].interpolate(method='linear', limit_direction='both', axis=0)#.interpolate(method='spline', order=3, inplace=True)
                    df['confidence'] = (df['confidence'].round(2) * 100).astype(int)
                # Réécrire dans le .CSV
                df.to_csv(PathFile, index=False, mode='w', sep=';')
            if 1 == 1:# Classif' des coordonnées pour identifier l'objet, par KMeans
                df = pd.read_csv(PathFile, sep=';')# Copie du .CSV dans un dataframe
                df['KmeansClass'] = 0
                X = df[['Xcenter', 'Ycenter']]
                # Créer un modèle K-means avec le nombre de clusters souhaité
                num_clusters = 4  # Spécifiez le nombre de clusters souhaité
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(X)# Entraîner le modèle
                predictions = kmeans.predict(X)# Prédire les clusters pour les données
                df['KmeansClass'] = predictions# Ecrit les classes dans le dataframe
                df.reset_index(drop=True, inplace=True)
                ##########
                # Ne garder que la class la plus fréquente
                # Pour tout garder, mettre nlargest = num_clusters
                value_counts = df['KmeansClass'].value_counts()
                most_frequent_values = value_counts.nlargest(4).index
                df = df[df['KmeansClass'].isin(most_frequent_values)]
                df['KmeansClass'] = df['KmeansClass'].astype(int)
                # Réécrire dans le .CSV
                df.to_csv(PathFile, index=False, mode='w', sep=';')
            if 1 == 0:# APPROCHE SupprimerSautsDistance : suppression de la ligne de df si d'un coup les valeurs diffèrent beaucoup des précédentes, en termes de distance parcourue
                df = pd.read_csv(PathFile, sep=';')# Copie du .CSV dans un dataframe
                df['Xcenter'] = pd.to_numeric(df['Xcenter'], errors='coerce')# Conversion des colonnes Xcenter et Ycenter en type numérique
                df['Ycenter'] = pd.to_numeric(df['Ycenter'], errors='coerce')
                df['distance'] = 0.0# Créer une nouvelle colonne 'distance'
                RowsToDrop = []
                for index, row in df.iterrows():# Pour chaque ligne
                    if index < 1:# Ignorer la première ligne
                        continue
                    else:
                        # Calculer la distance entre la ligne actuelle et la ligne précédente
                        distance = math.sqrt((int(df.loc[index, 'Xcenter']) - int(df.loc[index-1, 'Xcenter']))**2 + (int(df.loc[index, 'Ycenter']) - int(df.loc[index-1, 'Ycenter']))**2)
                        if distance > 8:# Si la distance est supérieure à 8,
                            RowsToDrop.append(index)
                            index -= 1# et soustraire l'index ainsi supprimé
                        else:# Sinon, écrire la distance dans la colonne 'distance'
                            print("distance jetee :" + str(distance))
                            df.loc[index, 'distance'] = distance
                df.drop(RowsToDrop, inplace=True)  # Supprimer les lignes en une seule opération
                df.reset_index(drop=True, inplace=True)# Réinitialiser l'indice du DataFrame après les suppressions de ligne
                # Réécrire dans le .CSV
                df.to_csv(PathFile, index=False, mode='w', sep=';')
            if 1 == 1:# APPROCHE AireMax : ne garder que la plus grande box pour chaque frame
                df = pd.read_csv(PathFile, sep=';')# Copie du .CSV dans un dataframe
                df['R'] = pd.to_numeric(df['R'], errors='coerce')
                # Obtenir l'indice de la ligne ayant la plus grande valeur de "R" pour chaque valeur de "frame"
                indices_to_keep = df.groupby(['file', 'localframe'])['R'].idxmax()
                # Supprimer les lignes qui ne correspondent pas aux indices à conserver
                df = df.drop(df.index.difference(indices_to_keep))
                # Réinitialiser les indices de position des lignes
                df = df.reset_index(drop=True)
                # Réécrire dans le .CSV
                df.to_csv(PathFile, index=False, mode='w', sep=';')
            if 1 == 1:# Lissage
                df = pd.read_csv(PathFile, sep=';')# Copie du .CSV dans un dataframe
                #df = Spline(df, 'Xcenter', 'Ycenter')
                df = Spline(df, 'X')
                df['X'] = df['X'].astype(int)
                df = Spline(df, 'Y')
                df['Y'] = df['Y'].astype(int)
                df = Spline(df, 'Xcenter')
                df['Xcenter'] = df['Xcenter'].astype(int)
                df = Spline(df, 'Ycenter')
                df['Ycenter'] = df['Ycenter'].astype(int)
                df = Spline(df, 'R')
                df['R'] = df['R'].astype(int)
                #
                # Réécrire dans le .CSV
                df.to_csv(PathFile, index=False, mode='w', sep=';')
            #
            #sys.exit()
    return 0
    #
    #
    #
    #
    def NettoyageCsv (df) :# NETTOYAGE DU DATAFRAME EN VUE DU CSV, NOTAMMENT INTERPOLATION
        df.reset_index(drop=True, inplace=True)
        # APPROCHE SupprimerSautsXcenter : suppression des sauts sur l'axe des X
        if 1 == 0:
            df['Xcenter'] = pd.to_numeric(df['Xcenter'], errors='coerce')# Conversion des colonnes Xcenter et Ycenter en type numérique
            df['XcenterRolling'] = df['Xcenter'].rolling(window=3).mean()
            RowsToDrop = []
            for index, row in df.iterrows():# Pour chaque ligne
                if index < 3:# Ignorer la première ligne
                    continue
                else:
                    if abs(int(row['Xcenter']) - int(row['XcenterRolling'])) > int(30 * eachframes):
                        RowsToDrop.append(index)
                        index -= 1# et soustraire l'index ainsi supprimé
            df.drop(RowsToDrop, inplace=True)  # Supprimer les lignes en une seule opération
            df.reset_index(drop=True, inplace=True)# Réinitialiser l'indice du DataFrame après les suppressions de ligne
        # APPROCHE SupprimerSautsDistance : suppression de la ligne de df si d'un coup les valeurs diffèrent beaucoup des précédentes, en termes de distance parcourue
        if 1 == 0:
            df['Xcenter'] = pd.to_numeric(df['Xcenter'], errors='coerce')# Conversion des colonnes Xcenter et Ycenter en type numérique
            df['Ycenter'] = pd.to_numeric(df['Ycenter'], errors='coerce')
            df['distance'] = 0.0# Créer une nouvelle colonne 'distance'
            RowsToDrop = []
            for index, row in df.iterrows():# Pour chaque ligne
                if index < 1:# Ignorer la première ligne
                    continue
                else:
                    # Calculer la distance entre la ligne actuelle et la ligne précédente
                    distance = math.sqrt((int(df.loc[index, 'Xcenter']) - int(df.loc[index-1, 'Xcenter']))**2 + (int(df.loc[index, 'Ycenter']) - int(df.loc[index-1, 'Ycenter']))**2)
                    if distance > 150:# Si la distance est supérieure à 50,
                        print("distance gardee :" + str(distance))
                        RowsToDrop.append(index)
                        index -= 1# et soustraire l'index ainsi supprimé
                    else:# Sinon, écrire la distance dans la colonne 'distance'
                        print("distance jetee :" + str(distance))
                        df.loc[index, 'distance'] = distance
            df.drop(RowsToDrop, inplace=True)  # Supprimer les lignes en une seule opération
            df.reset_index(drop=True, inplace=True)# Réinitialiser l'indice du DataFrame après les suppressions de ligne
        # APPROCHE Cluster : clustering par objet selon les différences de position des box
        if 1 == 0:
            X = df[['Xcenter', 'Ycenter']]# Sélectionnez les colonnes de position
            variances = []# Initialisez une liste pour stocker les valeurs de variance expliquée
            for num_clusters in range(1, 5):# Essayez différents nombres de clusters de 1 à 4
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(X)
                variances.append(kmeans.inertia_)
            # Tracez la courbe de variance expliquée en fonction du nombre de clusters
            plt.plot(range(1, 5), variances)
            plt.xlabel('Nombre de clusters')
            plt.ylabel('Variance expliquée')
            plt.title('Méthode du coude')
            plt.show(block=False)
            plt.pause(10.000)
            plt.close()
            # Utilisez le nombre optimal de clusters pour segmenter les données
            num_clusters_optimal = 2# Remplacez par votre nombre optimal de clusters
            kmeans = KMeans(n_clusters=num_clusters_optimal, random_state=42)
            kmeans.fit(X)
            df['cluster'] = kmeans.labels_# Ajoutez la colonne "cluster" au dataframe
            print(df)
        # APPROCHE AberrantesCorrigees : ajustement des valeurs aberrantes aux valeurs avant et après
        if 1 == 0:
            # Correction des valeurs aberrantes par la moyenne mobile
            for colonne in df.columns:
                if pd.api.types.is_numeric_dtype(df[colonne].dtype):
                    # Calcul de la moyenne mobile avec une fenêtre de taille 5
                    df['moyenne_mobile'] = df[colonne].rolling(window=8, min_periods=1).mean()
                    # Parcours des valeurs de la colonne
                    for i in range(len(df)):
                        value = df.at[i, colonne]
                        mean = df.at[i, 'moyenne_mobile']
                        # Vérification de la différence avec la moyenne mobile
                        if abs(value - mean) > 150:
                            df.at[i, colonne] = mean
                    df = df.drop('moyenne_mobile', axis=1)
            #
            if pd.api.types.is_numeric_dtype(df['X'].dtype):
                print(str(colonne) + str(" est numérique."))
            #
            # Convertit pour les colonnes numeric_dtype les valeurs float en int arrondi à la première décimale
            for colonne in df.columns:
                if colonne == 'frame' or pd.api.types.is_numeric_dtype(df[colonne].dtype):
                    df[colonne] = df[colonne].apply(lambda x: int(round(x, 1)) if pd.notnull(x) else x)
            #
            df['X'] = pd.to_numeric(df['X'], errors='coerce')
            #
        return 0