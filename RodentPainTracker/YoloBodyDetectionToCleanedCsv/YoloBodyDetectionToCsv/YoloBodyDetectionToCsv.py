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
def YoloBodyDetectionToCsv(PathSubFolderToAnalyze, sensibilityrate=0.5, IOU=True, eachframes=5):
    # Affichage utilisateur
    print(" └Exécution de YoloBodyDetectionToCsv ... ")
    #
    chronostart = datetime.datetime.now()
    # Liste les fichiers dans le sous-dossier
    ListOfFilePaths = sorted([f for f in os.listdir(PathSubFolderToAnalyze)])
    ListOfFilePaths = [PathSubFolderToAnalyze + "\\" + elem for elem in ListOfFilePaths]
    #####################################
    YOLOv3weights = str(str("\\".join(os.path.realpath(__file__).split("\\")[:-1])) + '\\' + 'yolov3.weights')
    YOLOv3cfg = str(str("\\".join(os.path.realpath(__file__).split("\\")[:-1])) + '\\' + 'yolov3.cfg')
    net = cv2.dnn.readNet(str(YOLOv3weights), str(YOLOv3cfg))# Chargement du réseau YOLOv3 pré-entraîné
    #net = cv2.dnn.readNetFromDarknet("E:\STAGE\Projet\yolov3-tiny.cfg", "E:\STAGE\Projet\yolov3-tiny.weights")
    classes = []# Chargement des noms des classes
    coconames = str(str("\\".join(os.path.realpath(__file__).split("\\")[:-1])) + '\\' + 'coco.names')
    with open(str(coconames), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_name = net.getLayerNames()
    output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))# Création d'un vecteur de couleurs pour les boîtes englobantes
    #outs = net.forward(output_layer)
    #####
    def YOLOdetection(frame):
        #
        class_ids = []
        confidences = []
        boxes = []
        # Définir les dimensions de l'image et la normaliser
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # Passer l'image à travers le modèle YOLOv3 pour détecter des objets
        net.setInput(blob)
        outs = net.forward(output_layer)
        # Analyser les sorties du modèle pour détecter des objets et dessiner des boîtes de délimitation
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > sensibilityrate:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indexes=[]
        if IOU == True:# Si paramètre IOU = True, alors suppression des boîtes qui se supperposent (suppression des non-maxima)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, sensibilityrate, 0.4)
            for i in reversed(range(len(boxes))):
                if i not in indexes:
                    boxes.pop(i)
        return class_ids, confidences, boxes, indexes
    #
    def YOLOdetectionToDf(boxes, File, frame_num, LastFrameNum, class_ids, confidences, indexes):# Récupération, d'après les boxes, des Locations : x_center, y_center, rayon
        tempdf = pd.DataFrame(columns=['folder', 'file', 'localframe', 'object', 'X', 'Y', 'W', 'H', 'confidence'])
        for i in range(len(boxes)):
            #if str(classes[class_ids[i]]) == "cat" or str(classes[class_ids[i]]) == "bear" or str(classes[class_ids[i]]) == "sheep" or str(classes[class_ids[i]]) == "dog":# Ne sélectionne que 'cat' 'bear' 'sheep' 'dog'
            if 1 == 1:
                #
                tempdf = pd.concat([tempdf, pd.DataFrame({'folder': str(PathSubFolderToAnalyze.split("\\")[-1]),
                                                          'file': str(File),
                                                          'localframe': [int(frame_num)],
                                                          'object': [str(classes[class_ids[i]])],
                                                          'X': [str(boxes[i][0])],
                                                          'Y': [str(boxes[i][1])],
                                                          'W': [str(boxes[i][2])],
                                                          'H': [str(boxes[i][3])],
                                                          'confidence': [str(confidences[i])]})],
                                                          ignore_index=True)
                tempdf = pd.DataFrame(tempdf)
        return tempdf
    #
    def DfToCsv(df):# ECRITURE DANS UN CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")# ajouter %f pour avoir les millisecondes
        pathtomove = str(str(PathSubFolderToAnalyze) + '\\' + str(PathSubFolderToAnalyze.split("\\")[-1]) + str(timestamp) + '.body.csv')
        df.to_csv(pathtomove, index=False, sep=';')# Export en CSV de df
        return 0

    ############################################################################
    # SCRIPT ###################################################################
    #
    #
    # INITIALISATION DE DF
    df = pd.DataFrame(columns=['folder', 'file', 'localframe', 'object', 'X', 'Y', 'W', 'H'])
    #
    eachframes = input("  Vous voulez traiter toutes les ... frames (saisir 1 pour garder toutes les frames) : ")
    eachframes = int(eachframes)
    fps = int(25 / eachframes)
    #
    ###################################################################################################################################
    #
    LastFrameNum = 0
    #
    for FilePath in ListOfFilePaths:
        File = str(FilePath).split('\\')[-1]
        # Cette section concerne le traitement d'images et non de vidéos. Ignorer.
        if File.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):
            #chemin_image = os.path.join(PathSubFolderToAnalyze, File)
            with open(FilePath, "r") as f:
                # img
                #cv2img = cv2.imread(FilePath)
                print("H.S.")
                ###########################################################################
        if File.lower().endswith((".mts")):#, ".mp4"
            #chemin_video = os.path.join(PathSubFolderToAnalyze, File)
            with open(FilePath, "r") as f:
                vid = cv2.VideoCapture(FilePath)# Chargement de la vidéo
                frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_processed = 0
                frames_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                #
                # initialiser la barre de progression
                ProgressBarBoxDetection = tqdm(total=frames_total)
                ProgressBarBoxDetection.set_description(str('  Détection des boîtes objets dans la vidéo ' + str(File)))
                #
                #
                while True:# Cette boucle principale analyse chaque frame et retourne les données sous forme de dataframe df
                    ret, frame = vid.read()# Lire l'image de la vidéo
                    if ret:# Récupère dimensions frame. Ret = booléen True si la lecture de l'image est réussie
                        frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
                        ProgressBarBoxDetection.update(1)# Mettre à jour la barre de progression
                        if frame_num % eachframes == 0:
                            img_height, img_width, img_channel = frame.shape
                            ######################
                            class_ids, confidences, boxes, indexes = YOLOdetection(frame)
                            tempdf = pd.DataFrame(YOLOdetectionToDf(boxes, File, frame_num, LastFrameNum, class_ids, confidences, indexes))
                            df = pd.concat([df, tempdf], ignore_index=True)
                            ######################
                    if not ret:
                        break
                vid.release()
                # Fermer la barre de progression
                ProgressBarBoxDetection.close()
                #
        #
        LastFrameNum += frame_num
        #
    #
    ######################
    #
    DfToCsv(df)
    #
    return 0