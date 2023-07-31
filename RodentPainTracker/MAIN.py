import os# Modules système
from os import walk
import sys
import tkinter as tk# Module de formulaires par pop'up Windows
from tkinter import filedialog
#########################################################################
# Sélection par popup du dossier contenant les sous-dossiers à analyser #
#########################################################################
root = tk.Tk()
root.withdraw()
PathFolderToAnalyze = filedialog.askdirectory(title="Veillez sélectionner le dossier contenant le(s) sous-dossier(s) à analyser ...") # affiche la boîte de dialogue pour sélectionner un dossier
PathFolderToAnalyze = str(os.path.normpath(PathFolderToAnalyze))
#########################################################################
import pandas as pd# Module de dataframes Panda
import openpyxl
import subprocess
import re
import shutil
import readchar# Saisie clavier de caractère unique
import cv2
import numpy as np

from WithBorisExtractBehavioralVideoSequences.WithBorisExtractBehavioralVideoSequences import WithBorisExtractBehavioralVideoSequences
from ApplyModelsToVideo.ApplyModelsToVideo import ApplyModelsToVideo
from YoloBodyDetectionToCleanedCsv.YoloBodyDetectionToCleanedCsv import YoloBodyDetectionToCleanedCsv
from YoloBodyCleanedCsvToVideo.YoloBodyCleanedCsvToVideo import YoloBodyCleanedCsvToVideo

from CreateBinaryModel.CreateBinaryModel import CreateBinaryModel

print(os.path.realpath(__file__))

from ForRGSextractFramesEach10minFromMTSsIn.ForRGSextractFramesEach10minFromMTSsIn import ForRGSextractFramesEach10minFromMTSsIn

# FONCTIONS

def ExtractSuffixAfterSpecificChars(string, chars):
    suffix = ''
    for char in reversed(string):
        if char in chars:
            break
        suffix = char + suffix
    return suffix

# SCRIPT ################################################
print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
print("Démarrage de RodentPainTracker ...")
#
###############################
# Formulaire de configuration #
###############################
command=""
#
root = tk.Tk()
root.title("Rodent Pain Tracker")
# icône logo.ico
chemin_icone = os.path.join(os.getcwd(),'logo','logo.ico')
root.iconbitmap(chemin_icone)
#
label = tk.Label(root, text="Pour chaque vidéo,")
label.pack()
#
instrc = ""
command = ""
#for foldername in os.listdir(os.getcwd()):
for foldername in ['ForRGSextractFramesEach10minFromMTSsIn','YoloBodyDetectionToCleanedCsv','YoloBodyCleanedCsvToVideo','WithBorisExtractBehavioralVideoSequences','CreateBinaryModel','ApplyModelsToVideo']:
    folderpath = os.path.join(os.getcwd(), foldername)
    if os.path.isdir(folderpath) and foldername != "__pycache__":
        for filename in os.listdir(folderpath):
            filepath = os.path.join(folderpath, filename)
            if filename.endswith(".yaml"):
                with open(filepath, "r") as file:
                    for line in file:
                        # Syntaxe des .yaml : nom var & val var & préfixe instruction formulaire & variable instruction formulaire & suffixe instruction formulaire
                        elements = line.strip().split("&")
                        #
                        third_element = elements[2] if len(elements) > 2 else ""
                        second_element = elements[1] if len(elements) > 1 else ""
                        fourth_element = elements[3] if len(elements) > 3 else ""
                        #
                        elements = [str(third_element), str(second_element), str(fourth_element)]
                        #
                        instrc = instrc+"".join(elements)+";"+"\n"
exec(instrc)

# Créer un bouton de soumission du formulaire
command += "root.quit()"
validation_button = tk.Button(root, text="Valider", bg="green", command=lambda: exec(command))
validation_button.pack(side=tk.BOTTOM)
#
root.mainloop()
#
#
#

################################################################################
# Appliquer les analyses choisies à chaque sous-dossier du dossier sélectionné #
################################################################################
for subfolder in os.listdir(PathFolderToAnalyze):# pour chaque sous-dossier à analyser
    PathSubFolderToAnalyze = os.path.join(PathFolderToAnalyze, subfolder)
    if os.path.isdir(PathSubFolderToAnalyze):
        print('└Traitement de "' + str(str(PathSubFolderToAnalyze).split('\\')[-1]) + '" ...')
        #
        # Captures écran des MTS pour le RGS toutes les 10min
        if RGS10min_var_check == True:
            ForRGSextractFramesEach10minFromMTSsIn(PathSubFolderToAnalyze, RGS10min_var_eachtime)
        # YOLO body détection
        if YoloBodyDetectionToCleanedCsv_var_check == True:
            YoloBodyDetectionToCleanedCsv(PathSubFolderToAnalyze)
            # contient :
            # - YoloBodyDetectionToCsv
            # - YoloBodyCsvToCleanedCsv
        # YOLO body générer vidéo
        if YoloBodyCleanedCsvToVideo_var_check == True:
            YoloBodyCleanedCsvToVideo(PathSubFolderToAnalyze)
        # Extraction des séquences vidéos de comportement selon BORIS
        if BorisToVideoSequences_var_check == True:
            WithBorisExtractBehavioralVideoSequences(PathSubFolderToAnalyze, ExtractWhat, AntiBehavior, SeqTrim, ImagesStep)
        # Avec les banques binaires d'images extraites (si extraites), créer les modèles de classification binaire d'images
        if CreateBinaryModel_var_check == True:
            CreateBinaryModel(PathSubFolderToAnalyze)
        # Appliquer les modèles de classification binaire à la / aux vidéo(s)
        if ApplyModelsToVideo_var_check == True:
            ApplyModelsToVideo(PathSubFolderToAnalyze)