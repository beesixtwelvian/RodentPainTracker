def ApplyModelsToVideo(PathSubFolderToAnalyze):
    import os
    ################################
    # Lister les modèles disponibles
    ListOfModels = []
    # Parcours des fichiers et dossiers dans le chemin spécifié
    for element in os.listdir(PathSubFolderToAnalyze):
        chemin_absolu = os.path.join(PathSubFolderToAnalyze, element)
        # Vérification si l'élément est un fichier se terminant par l'extension '.h5'
        if os.path.isfile(chemin_absolu) and element.lower().endswith('.h5'):
            ListOfModels.append(chemin_absolu)
    # Affichage des fichiers trouvés
    for model in ListOfModels:
        print(model)
    print(ListOfModels)
    ###########################
    # Lister les fichiers vidéo
    ListOfVideos = []
    # Parcours des fichiers et dossiers dans le chemin spécifié
    for element in os.listdir(PathSubFolderToAnalyze):
        chemin_absolu = os.path.join(PathSubFolderToAnalyze, element)
        # Vérification si l'élément est un fichier vidéo se terminant par les extensions '.mts' ou '.mp4'
        if os.path.isfile(chemin_absolu) and (element.lower().endswith('.mts') or element.lower().endswith('.mp4')):
            ListOfVideos.append(chemin_absolu)
    for element in ListOfVideos:
        if element.lower().endswith('.body.mp4'):
            ListOfVideos = [element]
    print(ListOfVideos)
    ##################################################################
    # Pour chaque modèle, appliquer pour chaque vidéo, et pour chaque image de la vidéo, le modèle
    import cv2
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.optimizers import RMSprop
    import tensorflow as tf
    import numpy as np
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
    for XPU in XPUs:
        with tf.device(XPU):
            #
            for Model in ListOfModels:
                model = keras.models.load_model(Model)# Chargement du modèle
                input_shape = model.input_shape[1:3]
                for PathVideo in ListOfVideos:
                    # Ecriture des résultats dans un .csv
                    import csv
                    ResultPath = os.path.join(PathSubFolderToAnalyze, str(os.path.basename(Model)) + 'Ethogram.csv')# Chemin du fichier CSV
                    # Ouvrir le fichier CSV en mode append (ajout) ou write (écriture)
                    with open(ResultPath, 'a', newline='') as file:
                        writer = csv.writer(file, delimiter=';')# Écrire la ligne dans le fichier CSV
                        #############################
                        # Sélection de la vidéo
                        video = cv2.VideoCapture(PathVideo)
                        # Vérification si la vidéo est ouverte correctement
                        if not video.isOpened():
                            print("Impossible d'ouvrir la vidéo.")
                            exit()
                        # Boucle pour parcourir chaque frame de la vidéo
                        frame_count = 1
                        while True:
                            # Lecture de la frame suivante
                            ret, frame = video.read()
                            # Vérification si la lecture de la frame a réussi
                            if not ret:
                                break
                            ##########################################
                            # Appliquer chaque modèle sur chaque frame
                            threshold=0.5 # Seuil de sensibilité de détection | Seuil = 0.5 | Si diminué, alors + tolérant.
                            #cote, cote2, canaux = frame.shape
                            #
                            img = np.array(frame, dtype=np.float32)  # Convertir l'image OpenCV en tableau numpy
                            img = cv2.resize(img, input_shape)
                            img = np.expand_dims(img, axis=0)  # Ajouter une dimension pour correspondre aux attentes du modèle
                            val=model.predict(img)
                            if val < threshold:
                                writer.writerow([str(frame_count), 'POS'])# Écrire la ligne résultat
                            else:
                                writer.writerow([str(frame_count), 'NEG'])# Écrire la ligne résultat
                            # Incrémentation du compteur de frame
                            frame_count += 1
                            # Vérification si la touche 'q' est pressée pour quitter la boucle
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        # Libération des ressources
                        video.release()
                        cv2.destroyAllWindows()
                ################################
                # Plot de l'éthogramme résultant
                import matplotlib.pyplot as plt
                # Lire le fichier CSV et extraire les valeurs de colonne 1 et les étiquettes de colonne 2
                valeurs_colonne1 = []
                etiquettes_colonne2 = []
                with open(ResultPath, 'r') as file:
                    reader = csv.reader(file, delimiter=';')
                    for row in reader:
                        valeurs_colonne1.append(int(row[0]))
                        etiquettes_colonne2.append(row[1])
                # Créer une matrice pour représenter le heatmap
                heatmap = np.zeros((1, max(valeurs_colonne1)))
                for valeur, etiquette in zip(valeurs_colonne1, etiquettes_colonne2):
                    if etiquette == 'POS':
                        heatmap[0, valeur-1] = 1
                # Créer le heatmap
                plt.imshow(heatmap, cmap='Greens_r', aspect='auto')
                # Personnaliser le graphique
                plt.xlabel('Frame')
                plt.title(str(Model))
                plt.xticks(range(len(valeurs_colonne1)), valeurs_colonne1, rotation='vertical')
                plt.yticks([])
                # Afficher le graphique
                plt.show()