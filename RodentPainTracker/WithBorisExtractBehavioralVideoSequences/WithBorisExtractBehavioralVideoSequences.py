import os
import sys
import pandas as pd
import openpyxl
import subprocess
import re
import shutil
import readchar# Saisie clavier de caractère unique
import cv2# Outil vidéo
import numpy as np
#
def WithBorisExtractBehavioralVideoSequences(PathSubFolderToAnalyze, ExtractWhat, AntiBehavior, SeqTrim, ImagesStep=1):
    ######################
    # SOUS-FONCTIONS #####
    def valueApointBtoStringA(value):
      return str(str(value).split('.')[0])
    def PathToLast9Chars(path):
      return path[-9:]
    def VideoPathToVideoDuration(video_path):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(frame_count / fps)
        video.release()
        return duration
    def ExtractSuffixAfterSpecificChars(string, chars):
        suffix = ''
        for char in reversed(string):
            if char in chars:
                break
            suffix = char + suffix
        return suffix
    ###########################################
    # SCRIPT ##################################
    print(' └Exécution de "WithBorisExtractBehavioralVideoSequences" ...')
    # Recherche du premier fichier BORIS .xlsx dans le dossier
    ListOfBoris = [file for file in os.listdir(PathSubFolderToAnalyze) if file.lower().endswith(".xlsx")]
    # Vérifier qu'au moins un fichier BORIS est présent
    if len(ListOfBoris) == 0:
        print("Dans ce sous-dossier, il n'y a pas de fichier BORIS au format .xlsx.")
    else:
        # Récupération dans une liste des emplacements absolus des vidéos présentes dans le dossier à analyser
        ListOfVideos = sorted([f for f in os.listdir(PathSubFolderToAnalyze) if f.lower().endswith(".mts") or f.lower().endswith(".mp4")])
        ListOfVideos = [PathSubFolderToAnalyze + "\\" + elem for elem in ListOfVideos]
        # S'il y a une vidéo de suivi .body.mp4, alors ne garder que cette vidéo dans la liste des vidéos à analyser
        UseTrackingVideo = False
        for Video in ListOfVideos:
            if Video.lower().endswith(".body.mp4"):
                ListOfVideos = [Video]
                UseTrackingVideo = True
                print("Utilisation de la vidéo de suivi au lieu des séquences vidéo.")
        # Traiter chaque fichier BORIS
        for Boris in ListOfBoris:
            BorisPath = str(PathSubFolderToAnalyze) + "\\" + str(Boris)
            # Lecture des données du fichier .xlsx dans un dataframe
            df = pd.read_excel(BorisPath)
            # Suppression des lignes qui ne sont pas de type STATE
            df = df.drop(df[df['Behavior type'] != 'STATE'].index)
            # Affichage du dataframe
            df = df.loc[:, ['Observation id', 'Media file', 'Behavior', 'Start (s)', 'Stop (s)']]
            for index, row in df.iterrows():
                splitter = row['Media file']
                splitter = ExtractSuffixAfterSpecificChars(splitter, "/\\")
                df.at[index,'Media file'] = str(splitter)
                df.at[index,'Behavior'] = row['Behavior'].replace(" ", "-")
            # Conversion des timemarks vidéo continue en timemarks par séquence vidéo
            # si vidéo séquences MTS et non vidéo continue .body.mp4
            if UseTrackingVideo == False:
                duration = 0
                i = 0
                for index, row in df.iterrows():
                  if PathToLast9Chars(ListOfVideos[i].lower()) != row['Media file'].lower():
                    duration = duration + VideoPathToVideoDuration(ListOfVideos[i])
                    i = i + 1
                  df.at[index, 'Start (s)'] = round(row['Start (s)'] - duration, 3)
                  df.at[index, 'Stop (s)'] = round(row['Stop (s)'] - duration, 3)
            if UseTrackingVideo == True:
                df['Media file'] = str(os.path.basename(ListOfVideos[0]))
            #
            # Liste les comportements
            ListOfBehaviors = df['Behavior'].unique()
            # Création des dossiers par comportement
            for behav in ListOfBehaviors:
                print(str(behav))
                subdf = df[df['Behavior'] == behav]
                subdf.reset_index(inplace=True, drop=True)
                subdf = pd.DataFrame(subdf)
                print("vvv")
                print("subdf")
                print(subdf)
                print("^^^")
                TheDfs = [subdf]
                if AntiBehavior == True:
                    # Créer le dataframe 'antisubdf'
                    antisubdf = pd.DataFrame(columns=subdf.columns)
                    # Parcourir chaque ligne de 'subdf'
                    for index, row in subdf.iterrows():
                        # Créer une ligne copie qui sera corrigée puis retenue
                        copy_row = row.copy()
                        # Vérifier la première ligne
                        if index == 0 and row['Start (s)'] > 0:
                            # Créer une ligne copie avec les modifications nécessaires
                            copy_row['Stop (s)'] = row['Start (s)']
                            copy_row['Start (s)'] = 0.000
                            # Ajouter la ligne copie à 'antisubdf'
                            antisubdf = pd.concat([antisubdf, copy_row.to_frame().T])
                        # Vérifier les lignes intermédiaires
                        if index > 0 and index < len(subdf):
                            # Stop devient Start
                            copy_row['Stop (s)'] = subdf.loc[index, 'Start (s)']
                            # Calcul de durée entre fin vidéo précédente et Stop comportement
                            video_path = str(os.path.join(PathSubFolderToAnalyze, str(copy_row['Media file'])))
                            cap = cv2.VideoCapture(video_path)
                            # count the number of frames
                            fps = float(cap.get(cv2.CAP_PROP_FPS))
                            totalNoFrames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration = totalNoFrames // fps
                            #
                            Debord = subdf.loc[index - 1, 'Stop (s)'] - duration
                            # Si le comportement ne débordait pas sur la séquence suivante, alors Start devient Stop précédent
                            if not Debord > 0:
                                copy_row['Start (s)'] = subdf.loc[index - 1, 'Stop (s)']
                            else:# Si le comportement débordait, tenir compte de la différence
                                copy_row['Start (s)'] = Debord
                                copy_row['Media file'] = subdf.loc[index, 'Media file']
                            # Ajouter la ligne copie à 'antisubdf'
                            antisubdf = pd.concat([antisubdf, copy_row.to_frame().T])
                        # Vérifier la dernière ligne
                        if index == len(subdf) - 1:
                            video_path = str(os.path.join(PathSubFolderToAnalyze, str(copy_row['Media file'])))
                            cap = cv2.VideoCapture(video_path)
                            fps = float(cap.get(cv2.CAP_PROP_FPS))
                            totalNoFrames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration = totalNoFrames // fps
                            if UseTrackingVideo == False:
                                if row['Stop (s)'] < duration:
                                    copy_row['Start (s)'] = round(float(subdf.loc[len(subdf)-1, 'Stop (s)']), 3)
                                    copy_row['Stop (s)'] = round(duration, 3)
                            if UseTrackingVideo == True:
                                valeurs_uniques = []
                                for index, row in df.iterrows():
                                    valeur = row['Media file']
                                    if valeur not in valeurs_uniques:
                                        valeurs_uniques.append(valeur)
                                total_duration = 0.0
                                for valeur_unique in valeurs_uniques:
                                    video_path = str(os.path.join(PathSubFolderToAnalyze, str(valeur_unique)))
                                    cap = cv2.VideoCapture(video_path)
                                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                                    totalNoFrames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    subduration = totalNoFrames // fps
                                    total_duration += subduration
                                if row['Stop (s)'] < total_duration:
                                    copy_row['Start (s)'] = round(float(subdf.loc[len(subdf)-1, 'Stop (s)']), 3)
                                    copy_row['Stop (s)'] = round(total_duration, 3)
                            # Ajouter la ligne copie à 'antisubdf'
                            antisubdf = pd.concat([antisubdf, copy_row.to_frame().T])
                    antisubdf.reset_index(inplace=True, drop=True)
                    print("vvv")
                    print("antisubdf")
                    print(antisubdf)
                    print("^^^")
                    TheDfs = [subdf, antisubdf]
                #
                os.makedirs(os.path.join(PathSubFolderToAnalyze, 'database'), exist_ok=True)
                #
                BehavOrAntiBehav=""
                for TheDf in TheDfs:
                    if TheDf is subdf:
                        BehavOrAntiBehav="POS"
                    else:
                        BehavOrAntiBehav="NEG"
                    DatasetName = behav + BehavOrAntiBehav
                    os.makedirs(os.path.join(PathSubFolderToAnalyze, 'database', DatasetName), exist_ok=True)
                    # Trimming
                    # Augmenter les valeurs de la colonne 'Start (s)' de 'Trimming'
                    TheDf['Start (s)'] = pd.to_numeric(TheDf['Start (s)'], errors='coerce')
                    TheDf['Start (s)'] = round((TheDf['Start (s)'] + SeqTrim), 3)
                    # Diminuer les valeurs de la colonne 'Stop (s)' de 'Trimming'
                    TheDf['Stop (s)'] = pd.to_numeric(TheDf['Stop (s)'], errors='coerce')
                    TheDf['Stop (s)'] = round((TheDf['Stop (s)'] - SeqTrim), 3)
                    # Filtrer le DataFrame selon la condition 'Stop (s)' > 'Start (s)'
                    TheDf = TheDf[TheDf['Stop (s)'] > TheDf['Start (s)']]
                    # Balisage des séquences comportementales à cheval sur deux vidéos
                    TheDf.insert(len(TheDf.columns), 'acheval', [None] * len(TheDf))
                    for index, row in TheDf.iterrows():
                        TheDf.at[index, 'acheval'] = 0
                    for index, row in TheDf.iterrows():
                        if index < len(TheDf) - 1:
                            pathtocheck = PathSubFolderToAnalyze + '\\' + row['Media file']
                            achevallimite = VideoPathToVideoDuration(pathtocheck)
                            achevalreel = round(row['Stop (s)'], 3)
                            if achevallimite < achevalreel:
                                TheDf.at[index, 'acheval'] = 1
                                TheDf.at[index, 'Stop (s)'] = round(achevalreel - (achevalreel - achevallimite), 3)
                                duplicat = row.copy()
                                duplicat['Media file'] = TheDf.loc[index+1, 'Media file']
                                duplicat['Start (s)'] = str(0.000)
                                duplicat['Stop (s)'] = round(achevalreel - achevallimite, 3)
                                duplicat['acheval'] = 1
                                TheDf.loc[index+0.5] = duplicat
                                TheDf = TheDf.sort_index()
                                TheDf = TheDf.reset_index(drop=True)
                    TheDf.reset_index(inplace=True, drop=True)
                    #
                    print(TheDf)
                    #
                    # EXTRACTION, pour chaque ligne,
                    if ExtractWhat == "videos":
                        thevidFilenameAfter = None
                        start = None
                        stop = None
                        for index, row in TheDf.iterrows():
                          PathBehav = os.path.join(PathSubFolderToAnalyze, 'database', DatasetName)
                          if row['acheval'] == 0 or row['acheval'] == 1:
                            currentBehav = row['Behavior']
                            prestart = start
                            start = float(row['Start (s)'])
                            prestop = stop
                            stop = float(row['Stop (s)'])
                            #diff = float(stop) - float(start)
                            if UseTrackingVideo == False:
                                thevidFilenameBefore = str(row['Media file'])
                            if UseTrackingVideo == True:
                                thevidFilenameBefore = str(ListOfVideos[0])
                            thevidFilenameAfterPrecedent = thevidFilenameAfter
                            thevidFilenameAfter = str(row['Observation id']) + "_" + os.path.basename(thevidFilenameBefore)[:-4] + "_" + valueApointBtoStringA(start) + "-" + valueApointBtoStringA(stop) + "_" + str(currentBehav) + "_" + str(row['acheval']) + ".mp4"
                            #################################################
                            # Ouverture de la vidéo d'où extraire la séquence
                            video_origine = cv2.VideoCapture(os.path.join(PathSubFolderToAnalyze, thevidFilenameBefore))
                            resolution = (int(video_origine.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_origine.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                            fps = int(video_origine.get(cv2.CAP_PROP_FPS))
                            codec = cv2.VideoWriter_fourcc(*'mp4v')
                            position_debut = int(start * fps)# Position de la première frame de la séquence
                            position_fin = int(stop * fps)# Position de la dernière frame de la séquence
                            # Répertoire de la vidéo de sortie
                            video_sortie = cv2.VideoWriter(os.path.join(PathBehav, thevidFilenameAfter), codec, fps, resolution)
                            # Pour chaque frame de la position de début à la position de fin, ajouter la frame à la suite de la vidéo de sortie qui est active
                            for i in range(position_debut, position_fin + 1):
                                video_origine.set(cv2.CAP_PROP_POS_FRAMES, i)
                                ret, frame = video_origine.read()
                                video_sortie.write(frame)
                            # Fermeture des vidéos d'origine et de sortie
                            video_origine.release()
                            video_sortie.release()
                            #################################################
                          # Concaténation des séquences vidéo à cheval
                          if index-1 >= 0:
                            if row['acheval'] == TheDf.loc[index-1]['acheval'] == 1:
                                thevidFilenameConcat = str(TheDf.loc[index-1]['Observation id']) + '_' + str(TheDf.loc[index-1]['Media file'][:-4]) + '_' + valueApointBtoStringA(prestart) + '-' + valueApointBtoStringA(prestop + stop) + '_' + str(currentBehav) + '_' + str(2)
                                #
                                # Avec cv2
                                # Ouverture des vidéos à concaténer
                                video1 = cv2.VideoCapture(os.path.join(PathBehav, thevidFilenameAfterPrecedent))
                                video2 = cv2.VideoCapture(os.path.join(PathBehav, thevidFilenameAfter))
                                # Récupération des métadonnées pour configurer la nouvelle vidéo concaténée
                                resolution = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                                fps = video1.get(cv2.CAP_PROP_FPS)
                                codec = cv2.VideoWriter_fourcc(*'mp4v')
                                # Ouverture de la nouvelle vidéo concaténée
                                nouvelle_video = cv2.VideoWriter(os.path.join(PathBehav, thevidFilenameConcat + ".mp4"), codec, fps, resolution)
                                # Parcours des frames des deux vidéos, lesquelles sont ajoutées à la suite de la nouvelle vidéo
                                while video1.isOpened():
                                    ret, frame = video1.read()
                                    if not ret:
                                        break
                                    nouvelle_video.write(frame)
                                while video2.isOpened():
                                    ret, frame = video2.read()
                                    if not ret:
                                        break
                                    nouvelle_video.write(frame)
                                # Fermeture des vidéos à concaténer et de la nouvelle vidéo concaténée
                                video1.release()
                                video2.release()
                                nouvelle_video.release()
                                #
                                for file in os.listdir(PathBehav):
                                    if file.lower().endswith("_1.mp4"):
                                        os.remove(os.path.join(PathBehav, file))
                    if ExtractWhat == "images":
                        for index, row in TheDf.iterrows():
                            PathBehav = os.path.join(PathSubFolderToAnalyze, 'database', DatasetName)
                            # Ouvrir la vidéo
                            video = cv2.VideoCapture(os.path.join(PathSubFolderToAnalyze, row['Media file']))
                            # Obtenir la fréquence d'images de la vidéo
                            fps = float(video.get(cv2.CAP_PROP_FPS))
                            # Convertir les temps de début et de fin en nombres d'images
                            start_frame = int(float(row['Start (s)']) * fps)
                            end_frame = int(float(row['Stop (s)']) * fps)
                            # ImagesStep donné dans config.yaml
                            # Aller à la position de départ dans la vidéo
                            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                            # Parcourir les images et les enregistrer
                            frame_counter = start_frame
                            while frame_counter <= end_frame:
                                # Lire une image de la vidéo
                                ret, frame = video.read()
                                # Vérifier si la lecture a réussi
                                if not ret:
                                    break
                                # Enregistrer l'image dans le dossier de sortie
                                output_path = os.path.join(PathBehav, f"{frame_counter}.png")
                                cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                                frame_counter += ImagesStep
                            # Fermer la vidéo
                            video.release()