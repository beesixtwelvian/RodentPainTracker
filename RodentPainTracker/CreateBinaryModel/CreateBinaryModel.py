def CreateBinaryModel(PathSubFolderToAnalyze):
    #Dans testing, le dossier jeu d'images positif et le négatif
    #images numérotées dans l'ordre croissant
    #
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image as ImageKeras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.optimizers import RMSprop
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import csv
    import cv2
    import os
    import numpy as np
    import pandas as pd
    import sys
    from timeit import default_timer as timer
    ######################
    # SOUS-FONCTIONS #####
    def AddResultLine(line):
        print(line)
        with open(PathSubFolderToAnalyze + '/Results.csv', 'a') as results_file:
            results_file.write(line + '\n')
    ################
    # SCRIPT #######
    # Initialisation
    pathdatabase = os.path.join(PathSubFolderToAnalyze, 'database')
    # Récupération des emplacements des dossiers présents dans pathdatabase
    folders = [os.path.join(pathdatabase, folder) for folder in os.listdir(pathdatabase) if os.path.isdir(os.path.join(pathdatabase, folder))]
    # Modification des noms de dossier et élimination des trois dernières lettres
    FoldersWithoutPOSNEG = [folder[:-3] for folder in folders]
    # Élimination des doublons en utilisant un ensemble (set)
    unique_behavs = list(set(FoldersWithoutPOSNEG))
    unique_behavs = [os.path.basename(unique_behav) for unique_behav in unique_behavs]
    open(PathSubFolderToAnalyze + '/Results.csv', 'w').close()# servira à stocker les résultats
    #
    # Pour chaque comportement et anticomportement dans la database
    for unique_behav in unique_behavs:
        pathssubfolders = []
        subfolders = [str(unique_behav + "POS"), str(unique_behav + "NEG")]
        for subfolder in subfolders:
            pathsubfolder = os.path.join(pathdatabase, subfolder)
            if os.path.isdir(pathsubfolder):
                pathssubfolders.append(pathsubfolder)
        subfoldersfiles = []
        for pathsubfolder in pathssubfolders:
            temp = os.listdir(pathsubfolder)
            for subfolderfile in temp:
                subfoldersfiles.append(os.path.join(pathsubfolder, subfolderfile))
        subfolder = []
        pathframe = []
        print("df")
        for pathsubfolderfile in subfoldersfiles:
            subfolder.append(os.path.basename(os.path.dirname(pathsubfolderfile)))
            pathframe.append(pathsubfolderfile)
        df = pd.DataFrame({
            'subfolder': subfolder,
            'pathsubfolderfile': pathframe
        })
        df = df.sample(frac=1)# Mélanger aléatoirement les lignes du DataFrame
        df = df.reset_index(drop=True)
        print(df)
        # Enregistrer le DataFrame en tant que fichier CSV dans le dossier actif
        df.to_csv(os.path.join(PathSubFolderToAnalyze, 'EmplacementsImagesPourModele.csv'), index=False, sep=';')
        #
        ##############
        #
        i = 0
        while i < 1:# //!\\ Va effectuer n itération(s) //!\\
            start = timer()
            ##################################################################################################
            # Choisir de réutiliser (True), ou non (False), le même jeu de données sélectionné aléatoirement #
            ReuseTheSameAleatoryDataSample = False
            ##################################################################################################
            if ReuseTheSameAleatoryDataSample == True:
                AddResultLine("ReuseTheSameAleatoryDataSample")
                if os.path.isfile(os.path.join(PathSubFolderToAnalyze, "EmplacementsImagesPourModele.csv")):
                    df = pd.read_csv(os.path.join(PathSubFolderToAnalyze, 'EmplacementsImagesPourModele.csv'), sep=';')# Récupérer le fichier CSV dans un nouveau DataFrame
                else:    
                    df = df.sample(frac=1)# Mélanger aléatoirement les lignes du DataFrame
                    df = df.reset_index(drop=True)
                    df = pd.read_csv(os.path.join(PathSubFolderToAnalyze, 'EmplacementsImagesPourModele.csv'), sep=';')# Récupérer le fichier CSV dans un nouveau DataFrame
            else:
                AddResultLine("DoNotReuseTheSameAleatoryDataSample")
                df = df.sample(frac=1)# Mélanger aléatoirement les lignes du DataFrame
                df = df.reset_index(drop=True)
                df = pd.read_csv(os.path.join(PathSubFolderToAnalyze, 'EmplacementsImagesPourModele.csv'), sep=';')# Récupérer le fichier CSV dans un nouveau DataFrame
            ##################################################################################################
            # dfPOS
            dfPOS = df.loc[df['subfolder'] == str(unique_behav) + 'POS'].reset_index(drop=True)# Réinitialiser les index du DataFrame
            dfPOSPrompt="dfPOS, longueur"+';'+str(len(dfPOS))+';'+"NbClasses"+';'+str(dfPOS['subfolder'].nunique())
            AddResultLine(dfPOSPrompt)
            # dfNEG
            dfNEG = df.loc[df['subfolder'] == str(unique_behav) + 'NEG'].reset_index(drop=True)# Réinitialiser les index du DataFrame
            dfNEGPrompt="dfNEG, longueur"+';'+str(len(dfNEG))+';'+"NbClasses"+';'+str(dfNEG['subfolder'].nunique())
            AddResultLine(dfNEGPrompt)
            N = int(min(len(dfPOS),len(dfNEG)))
            print("N" + str(N))
            t = int(min(0.1*N, 100))# pourcentages 60 30 10
            print("t" + str(t))
            T = int((3/4)*(N-t))
            print("T" + str(T))
            V = int(T/3)
            print("V" + str(V))
            if T <= 0:
                print("Pas assez d'images !")
                break
            # dftraining
            dftraining = pd.concat([dfPOS.iloc[:T], dfNEG.iloc[:T]], ignore_index=True)
            dftraining = dftraining.reset_index(drop=True)
            dftrainingPrompt="dftraining, longueur"+';'+str(len(dftraining))+';'+"NbClasses"+';'+str(dftraining['subfolder'].nunique())
            AddResultLine(dftrainingPrompt)
            dftrainingNbPOS="dftrainingNbPOS"+';'+str(dftraining['subfolder'].value_counts()[str(unique_behav) + 'POS'])
            AddResultLine(dftrainingNbPOS)
            dftrainingNbNEG="dftrainingNbNEG"+';'+str(dftraining['subfolder'].value_counts()[str(unique_behav) + 'NEG'])
            AddResultLine(dftrainingNbNEG)
            # dfvalidation
            dfvalidation = pd.concat([dfPOS.iloc[T:T+V], dfNEG.iloc[T:T+V]], ignore_index=True)
            dfvalidation = dfvalidation.reset_index(drop=True)
            dfvalidationPrompt="dfvalidation, longueur"+';'+str(len(dfvalidation))+';'+"NbClasses"+';'+str(dfvalidation['subfolder'].nunique())
            AddResultLine(dfvalidationPrompt)
            dfvalidationNbPOS="dfvalidationNbPOS"+';'+str(dfvalidation['subfolder'].value_counts()[str(unique_behav) + 'POS'])
            AddResultLine(dfvalidationNbPOS)
            dfvalidationNbNEG="dfvalidationNbNEG"+';'+str(dfvalidation['subfolder'].value_counts()[str(unique_behav) + 'NEG'])
            AddResultLine(dfvalidationNbNEG)
            # dftesting
            dftesting = pd.concat([dfPOS.iloc[T+V:T+V+t], dfNEG.iloc[T+V:T+V+t]], ignore_index=True)
            dftesting = dftesting.reset_index(drop=True)
            dftestingPrompt="dftesting, longueur"+';'+str(len(dftesting))+';'+"NbClasses"+';'+str(dftesting['subfolder'].nunique())
            AddResultLine(dftestingPrompt)
            #
            TestingSampleSizePOS = dftesting['subfolder'].value_counts()[str(unique_behav) + 'POS']
            AddResultLine("TestingSampleSizePOS"+';'+str(TestingSampleSizePOS))
            TestingSampleSizeNEG = dftesting['subfolder'].value_counts()[str(unique_behav) + 'NEG']
            AddResultLine("TestingSampleSizeNEG"+';'+str(TestingSampleSizeNEG))
            #
            pathmodelsaved = str(os.path.join(PathSubFolderToAnalyze, str(unique_behav) + '.h5'))
            #
            #################################################################
            # Choisir la taille de redimensionnement des images (pixels côté)
            PathFirstImageTrainingToExtractShape = dftraining.loc[0, 'pathsubfolderfile']
            image = cv2.imread(PathFirstImageTrainingToExtractShape)
            FenetreHauteur, FenetreLargeur, canaux = image.shape
            # Définir la taille de batch
            batch = int(16)
            #################################################################
            #
            ################################
            # TRAINING
            train=ImageDataGenerator(rescale=1/255)
            train_dataset = train.flow_from_dataframe(dataframe=dftraining,
                                                      x_col='pathsubfolderfile',
                                                      y_col='subfolder',
                                                      target_size=(FenetreHauteur, FenetreLargeur),
                                                      batch_size=batch,# IMPORTANT //!\\
                                                      class_mode='binary',
                                                      shuffle=False)
            train_dataset.class_indices# Affiche des classes et leur index
            train_dataset.classes# Pour les images du dossier, affiche l'index de classe associé (doit être [0, 0 ... 1, 1])
            # VALIDATION
            validation=ImageDataGenerator(rescale=1/255)
            validation_dataset = validation.flow_from_dataframe(dataframe=dfvalidation,
                                                                x_col='pathsubfolderfile',
                                                                y_col='subfolder',
                                                                target_size=(FenetreHauteur, FenetreLargeur),
                                                                batch_size=batch,
                                                                class_mode='binary',
                                                                shuffle=False)
            #
            AddResultLine("terminé en" + ';' + str(timer()-start) + ';' + "secondes (mélange données)")
            #
            # Vérification de la disponibilité du GPU pour TensorFlow
            from tensorflow.python.client import device_lib# Vérifier la disponibilité des GPU
            gpu_devices = [device.name for device in device_lib.list_local_devices() if device.device_type == "GPU"]
            print(gpu_devices)
            if not gpu_devices:
                print("Aucun GPU disponible. Veuillez vous assurer que CUDA et les pilotes NVIDIA sont correctement installés.")
                XPUs = ["/CPU:0"]
            else:
                XPUs = ["/GPU:0"]
            #
            for XPU in XPUs:
                AddResultLine(XPU)
                #
                start = timer()
                #
                ############################################################
                # Définition de l'architecture du réseau de neurones choisie
                model=tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(8,(6,6),activation='relu',input_shape=(FenetreHauteur,FenetreLargeur,3)),
                    tf.keras.layers.MaxPool2D(2,2),
                    tf.keras.layers.Conv2D(16,(6,6),activation='relu'),
                    tf.keras.layers.MaxPool2D(2,2),
                    tf.keras.layers.Conv2D(32,(6,6),activation='relu'),
                    tf.keras.layers.MaxPool2D(2,2),
                    tf.keras.layers.Conv2D(64,(6,6),activation='relu'),
                    tf.keras.layers.MaxPool2D(2,2),
                    tf.keras.layers.Conv2D(128,(6,6),activation='relu'),
                    tf.keras.layers.MaxPool2D(2,2),
                    tf.keras.layers.Flatten(),#=neurones en ligne
                    tf.keras.layers.Dense(512,activation='relu'),
                    tf.keras.layers.Dense(1,activation='sigmoid')
                    ])
                #######################
                # Compilation du modèle
                model.compile(loss='binary_crossentropy',
                            optimizer=RMSprop(learning_rate=0.001),#RootMeanSquare moy quadratique
                            metrics=['accuracy'])
                AddResultLine("terminé en" + ';' + str(timer()-start) + ';' + "secondes (compilation modèle)")
                #
                start = timer()
                #
                with tf.device(XPU):
                    model_fit=model.fit(train_dataset,
                                        steps_per_epoch=int(N / batch),#steps_per_epoch = N images // batch_size si on veut que toutes les img soient traitées # = à chaque epoch, c'est le nombre de pas d'entraînement effectué
                                        epochs=10,#10 # Nb. de fois que le modèle va passer par l'ensemble des données
                                        validation_data=validation_dataset)
                AddResultLine("steps_per_epoch" + ';' + str(model_fit.params['steps']))
                AddResultLine("epochs" + ';' + str(model_fit.params['epochs']))
                #
                history = model_fit.history
                loss_values = history['loss']
                accuracy_values = history['accuracy']
                val_loss_values = history['val_loss']
                val_accuracy_values = history['val_accuracy']
                for epoch, accuracy, loss, val_accuracy, val_loss in zip(range(1, len(accuracy_values) + 1), accuracy_values, loss_values, val_accuracy_values, val_loss_values):
                    AddResultLine("Epoch"+';'+str(epoch)+';'+"Loss"+';'+ str(loss)+';'+"Accuracy"+';'+str(accuracy)+';'+"ValidLoss"+';'+str(val_loss)+';'+"ValidAccuracy"+';'+str(val_accuracy))
                #
                # Sauvegarde du modèle dans le dossier actif
                model.save(pathmodelsaved)
                #
                ################################################
                # Récupération du nombre de paramètres du modèle
                model = keras.models.load_model(pathmodelsaved)# Chargement du modèle
                with open(os.path.join(PathSubFolderToAnalyze, 'ModelSummary.txt'), 'w') as file:
                    sys.stdout = file  # Rediriger la sortie standard vers le fichier
                    model.summary()
                    sys.stdout = sys.__stdout__  # Rétablir la sortie standard
                with open(os.path.join(PathSubFolderToAnalyze, 'ModelSummary.txt'), 'r') as file:
                    model_summary = file.read().splitlines()
                output_rows = []
                for line in model_summary:
                    output_rows.append(line.split())
                with open(os.path.join(PathSubFolderToAnalyze, 'ModelSummary.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerows(output_rows)
                ######################
                ModelSummaryDf = pd.read_csv(os.path.join(PathSubFolderToAnalyze, 'ModelSummary.csv'), delimiter=' ')
                # Récupérer le nom de la colonne contenant 'params'
                col_name = None
                for col in ModelSummaryDf.columns:
                    if ModelSummaryDf[col].str.contains('params:').any():
                        col_name = col
                        break
                # Récupérer df2 en filtrant les lignes avec 'params' dans la colonne appropriée
                if col_name is not None:
                    ModelSummaryDf = ModelSummaryDf[ModelSummaryDf[col_name].str.contains('params:')]
                else:
                    ModelSummaryDf = pd.DataFrame()# Aucune colonne ne contient 'params'
                # Afficher la valeur de chaque colonne pour chaque ligne
                for index, row in ModelSummaryDf.iterrows():
                    ToPrompt = ""
                    for col in ModelSummaryDf.columns:
                        ToPrompt = ToPrompt + str(row[col]) + ' '
                    AddResultLine(ToPrompt)
                ################################################
                #
                #
                AddResultLine("Model saved.")
                AddResultLine("terminé en" + ';' + str(timer()-start) + ';' + "secondes")
                start = timer()
                #
                #
                # Applique le modèle une fois entraîné sur les données de testing
                #############
                threshold=0.1 # Seuil de sensibilité de détection | Seuil = 0.5 | Si diminué, alors + tolérant.
                AddResultLine("threshold" + ';' + str(threshold))
                #############
                model = keras.models.load_model(pathmodelsaved)# Chargement du modèle
                model.summary()# Résumé du modèle
                # TheTestingArrayFace
                TheTestingArrayPOS = dftesting.loc[dftesting['subfolder'] == str(unique_behav) + 'POS']
                print("TheTestingArrayPOS créé, de longueur " + str(len(TheTestingArrayPOS)) + ", et " + str(TheTestingArrayPOS['subfolder'].nunique()) + " classe(s).")
                TheTestingArrayPOS = TheTestingArrayPOS['pathsubfolderfile'].tolist()
                # TheTestingArrayNoFace
                TheTestingArrayNEG = dftesting.loc[dftesting['subfolder'] == str(unique_behav) + 'NEG']
                print("TheTestingArrayNEG créé, de longueur " + str(len(TheTestingArrayNEG)) + ", et " + str(TheTestingArrayNEG['subfolder'].nunique()) + " classe(s).")
                TheTestingArrayNEG = TheTestingArrayNEG['pathsubfolderfile'].tolist()
                ############################
                for TestingArray in (TheTestingArrayPOS, TheTestingArrayNEG):
                    TestingSampleSizePOSDetected = 0
                    TestingSampleSizeNEGDetected = 0
                    for frame in TestingArray:
                        img = ImageKeras.load_img(str(frame), target_size=(FenetreHauteur, FenetreLargeur))
                        X=ImageKeras.img_to_array(img)
                        X=np.expand_dims(X,axis=0)
                        images=np.vstack([X])
                        val=model.predict(images)
                        if val < threshold:
                            TestingSampleSizePOSDetected += 1
                        else:
                            TestingSampleSizeNEGDetected += 1
                    pos = str(int((TestingSampleSizePOSDetected / TestingSampleSizePOS)*t))
                    neg = str(int((TestingSampleSizeNEGDetected / TestingSampleSizeNEG)*t))
                    print("pos ",pos," neg ",neg)
                    # Afficher les noms des variables
                    ArrayTested = ""
                    if TestingArray is TheTestingArrayPOS:
                        ArrayTested = "TheTestingArrayPOS"
                    if TestingArray is TheTestingArrayNEG:
                        ArrayTested = "TheTestingArrayNEG"
                    AddResultLine(str(ArrayTested) + ';' + pos + ';' + neg)
                #
                AddResultLine("terminé en" + ';' + str(timer()-start) + ';' + "secondes")
                #
                DeleteModelAtTheEnd = False
                if DeleteModelAtTheEnd == True:
                    os.remove(pathmodelsaved)
            i += 1
    if os.path.exists(os.path.join(PathSubFolderToAnalyze, 'EmplacementsImagesPourModele.csv')):
        os.remove(os.path.join(PathSubFolderToAnalyze, 'EmplacementsImagesPourModele.csv'))
    if os.path.exists(os.path.join(PathSubFolderToAnalyze, 'ModelSummary.csv')):
        os.remove(os.path.join(PathSubFolderToAnalyze, 'ModelSummary.csv'))
    if os.path.exists(os.path.join(PathSubFolderToAnalyze, 'ModelSummary.txt')):
        os.remove(os.path.join(PathSubFolderToAnalyze, 'ModelSummary.txt'))