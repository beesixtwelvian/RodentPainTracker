import os
from os import walk
import sys
import openpyxl
import subprocess
import re
import shutil
# Outils vidéo
import cv2
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
import matplotlib.pyplot as plt

from Spline.Spline import Spline

# La fonction YOLOanalyze prend en paramètres :
# - 1er, PathSubFolderToAnalyze est le media d'entrée (peut être une image, une vidéo, un répertoire fichier image ou vidéo, en répertoire dossier contenant des images ou vidéos
# - 2ème, sensibilityrate est le seuil, la sensibilité, de détection, choisi
# - 3ème, IOU Intersection Over Union, s'il est True, supprime les boîtes superposées afin de ne détecter que un objet
# - 4ème, eachframe, si le fichier est une vidéo, n'applique le YOLO que toutes les tant de frames
def YoloBodyCleanedCsvToVideo(PathSubFolderToAnalyze):
    # Affichage utilisateur
    print(" └Exécution de YoloBodyCleanedCsvToVideo ... ")
    ###################################
    def CropToCircleCV2img (cv2img, xcenter, ycenter, rayon):
        cv2img_height, cv2img_width, cv2img_channel = cv2img.shape
        # Affichage de chaque zone
        DeltaTop = DeltaBottom = DeltaLeft = DeltaRight = 0
        DeltaTop = int((ycenter - rayon))
        DeltaBottom = int(cv2img_height - (ycenter + rayon))
        DeltaLeft = int((xcenter - rayon))
        DeltaRight = int(cv2img_width - (xcenter + rayon))
        #
        if DeltaTop > 0:
            ToCropTop = DeltaTop
            ToAddTop = 0
        else:
            ToCropTop = 0
            ToAddTop = - DeltaTop
        if DeltaBottom >0:
            ToCropBottom = DeltaBottom
            ToAddBottom = 0
        else:
            ToCropBottom = 0
            ToAddBottom = - DeltaBottom
        if DeltaLeft > 0:
            ToCropLeft = DeltaLeft
            ToAddLeft = 0
        else:
            ToCropLeft = 0
            ToAddLeft = - DeltaLeft
        if DeltaRight > 0:
            ToCropRight = DeltaRight
            ToAddRight = 0
        else:
            ToCropRight = 0
            ToAddRight = - DeltaRight
        # Proj
        proj = cv2img[ToCropTop:cv2img_height-ToCropBottom, ToCropLeft:cv2img_width-ToCropRight]
        proj_height, proj_width, proj_channel = proj.shape
        # Sqr
        sqr = cv2.copyMakeBorder(proj, ToAddTop, ToAddBottom, ToAddLeft, ToAddRight, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        sqr_height, sqr_width, sqr_channel = sqr.shape
        # SqrImpair
        SqrPairOrImpair = sqr.copy()
        SqrPairOrImpair_height, SqrPairOrImpair_width, _ = SqrPairOrImpair.shape
        if SqrPairOrImpair_width % 2 == 0:# Si largeur d'image pair, alors rogner l'image d'un pixel par la droite
            SqrPairOrImpair = SqrPairOrImpair[:, :-1]
        if SqrPairOrImpair_height % 2 == 0:# Si hauteur d'image pair, alors rogner l'image d'un pixel par le bas
            SqrPairOrImpair = SqrPairOrImpair[:-1, :]
        sqrimpair = SqrPairOrImpair.copy()
        sqrimpair_height, sqrimpair_width, _ = sqrimpair.shape
        # Circle
        circle = sqrimpair.copy()
        circle_height, circle_width, _ = circle.shape
        x_circlecenter = int(circle_width // 2) + 1
        y_circlecenter = int(circle_height // 2) + 1
        #
        circularize = True
        if circularize == True:
            # Itérer sur chaque pixel de l'image
            for x_circle in range(circle_width):
                for y_circle in range(circle_height):
                    # Si pixel hors du cercle, alors éteindre le pixel
                    if math.sqrt((x_circle-x_circlecenter)**2 + (y_circle-y_circlecenter)**2) >= int(circle_height // 2):
                        circle[y_circle,x_circle] = (0, 0, 0)
        # Redimensionner la frame pour la sortie vidéo
        circle = cv2.resize(circle, (416, 416))
        return circle
    # Liste les fichiers dans le sous-dossier, pour trouver *.body.csv
    for File in os.listdir(PathSubFolderToAnalyze):
        PathFile = os.path.join(PathSubFolderToAnalyze, File)
        # Vérifier si le fichier se termine par ".body.csv"
        if os.path.isfile(PathFile) and File.endswith(".body.csv"):
            df = pd.read_csv(PathFile, sep=';')  # Copie du .CSV dans un dataframe
            #
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")# ajouter %f pour avoir les millisecondes
            #
            # Si vidéos, initialise les paramètres de la vidéo de sortie
            for FilePath in os.listdir(PathSubFolderToAnalyze):
                FilePath = os.path.join(PathSubFolderToAnalyze, FilePath)
                print(FilePath)
                File = str(FilePath).split('\\')[-1]
                print(File)
                if File.lower().endswith((".mts")):#, ".mp4"
                    #chemin_video = os.path.join(PathSubFolderToAnalyze, File)
                    with open(FilePath, "r") as f:
                        vid = cv2.VideoCapture(FilePath)# Chargement de la vidéo
                        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frames_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                        vid.release()
            # Spécifiez les informations de la vidéo de sortie
            CropOrNot = True
            if CropOrNot == True:# Découper
                frame_size = (416, 416)# width puis height
            else:
                frame_size = (frame_width, frame_height)# width puis height
            OutputName = str(str(PathSubFolderToAnalyze) + "\\" + str(PathSubFolderToAnalyze.split("\\")[-1]) + str(timestamp) + '.body' + '.mp4')
            eachframes = input("  Vous voulez traiter toutes les ... frames (saisir 1 pour garder toutes les frames) : ")
            eachframes = int(eachframes)
            fps = int(25 / eachframes)
            ###########################################################
            # Créez un objet VideoWriter pour écrire la vidéo de sortie
            out = cv2.VideoWriter(OutputName, cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size, isColor=True)
            ###########################################################
            #
            for file_group, file_df in df.groupby('file'):
                # Réinitialisation des index du nouveau dataframe
                file_df.reset_index(drop=True, inplace=True)
                FilePath = os.path.join(PathSubFolderToAnalyze, file_df.loc[0, 'file'])
                print(FilePath)
                File = str(file_df.loc[0, 'file'])
                print(File)
                if File.lower().endswith((".mts")):
                    with open(FilePath, "r") as f:
                        vid = cv2.VideoCapture(FilePath)# Chargement de la vidéo
                    # initialiser la barre de progression
                    ProgressBarBoxDraw = tqdm(total=len(file_df['localframe']))
                    ProgressBarBoxDraw.set_description(str('  Dessin des boîtes objets sur les frames de ' + str(File)))
                    #
                    while True:# Cette boucle secondaire lit le dataframe df pour ajouter sur l'image les dessins correspondants (rectangles de détection)
                        # Lire l'image de la vidéo
                        ret, frame = vid.read()
                        # Récupère dimensions frame
                        if ret:
                            rang_frame = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
                            if rang_frame % eachframes == 0:
                                dftoextractfiltered = file_df.loc[file_df['localframe'] == rang_frame]
                                ################
                                # Dernier filtre
                                # APPROCHE SupprimerEcartsMediane : suppression des boîtes ayant un écart par rapport à la médiane des valeurs de Xcenter puis de Ycenter
                                if 1 == 1:
                                    # Calculer la médiane de chaque colonne
                                    xcenter_median = np.median(dftoextractfiltered['Xcenter'])
                                    ycenter_median = np.median(dftoextractfiltered['Ycenter'])
                                    # Définir la condition pour filtrer les lignes
                                    condition = (abs(dftoextractfiltered['Xcenter'] - xcenter_median) <= 100) & (abs(dftoextractfiltered['Ycenter'] - ycenter_median) <= 100)
                                    # Appliquer le filtrage en utilisant la condition
                                    dftoextractfiltered = dftoextractfiltered[condition]
                                for index, row in dftoextractfiltered.iterrows():
                                    x = int(row.loc['X'])
                                    y = int(row.loc['Y'])
                                    w = int(row.loc['W'])
                                    h = int(row.loc['H'])
                                    xcenter = int(row.loc['Xcenter'])
                                    ycenter = int(row.loc['Ycenter'])
                                    rayon = int(row.loc['R'])
                                    label = str(str(row.loc['object']) + ' ' + str(file_df.loc[index, 'confidence']) + '%')
                                    #color = colors[class_ids[index]]
                                    color = (0, 255, 255)#BGR
                                    if 1 == 0:
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                                        cv2.circle(frame, (xcenter, ycenter), 8, (0, 0, 255), 16)# Dessiner un cercle rouge de rayon 15px et épaisseur 8px centré en (xcenter, ycenter)
                                        font_scale = float(h / 300)
                                        cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                                #############
                                if CropOrNot == True:
                                    frame = CropToCircleCV2img(frame, xcenter, ycenter, rayon)
                                #############
                                #
                                WindowName = "YOLOvideo"
                                cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)  # Ajout de cette ligne pour créer une fenêtre redimensionnable
                                #cv2.resizeWindow(WindowName, int(img_width / 3), int(img_height / 3))  # Définition de la taille initiale de la fenêtre
                                cv2.imshow(WindowName, frame)
                                #
                                key = cv2.waitKey(1)
                                if key == ord("q"):
                                    break
                                #
                                out.write(frame)
                                #
                            ProgressBarBoxDraw.update(1)
                        if not ret:
                            break
                        #
                ######################
                vid.release()
                cv2.destroyAllWindows()
                # Fermer la barre de progression
                ProgressBarBoxDraw.close()
                #
            # Referme la vidéo alors terminée ###
            out.release()########################
            #####################################
            # Réécrire dans le .CSV
            df.to_csv(PathFile, index=False, mode='w', sep=';')
    return 0