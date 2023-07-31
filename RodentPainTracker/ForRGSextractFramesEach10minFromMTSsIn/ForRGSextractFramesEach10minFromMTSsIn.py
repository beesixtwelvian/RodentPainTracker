def ForRGSextractFramesEach10minFromMTSsIn(PathSubFolderToAnalyze, RGS10min_var_eachtime):
    import os
    import cv2# Outil de traitement vidéo
    print(' └Exécution de ForRGSextractFramesEach10minFromMTSsIn ...')
    eachtime = float(RGS10min_var_eachtime)
    #
    # Récupération de la liste des fichiers MTS dans le sous-dossier
    MTSs = sorted([f for f in os.listdir(PathSubFolderToAnalyze) if f.lower().endswith(".mts")])
    MTSs = [PathSubFolderToAnalyze + "\\" + elem for elem in MTSs]
    #
    CumulatedInMTSFrameTime = eachtime * 60
    FramesAhead = int(0)
    for mts in MTSs:
        # Ouverture de la vidéo
        video = cv2.VideoCapture(mts)
        # Calcul de la durée de la vidéo
        TotalFramesInTheVideo = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        FramesAhead = int(max(FramesAhead, TotalFramesInTheVideo))
        video.release()
    # Si au moins une vidéo dure plus que l'intervalle d'extraction des images RGS, alors procéder à l'extraction
    if FramesAhead >= eachtime * 60 * fps:
        # Créer un dossier RGS où on mettra les images extraites
        PathSubFolderRGS = str(PathSubFolderToAnalyze + "\\" + "RGS")
        os.makedirs(PathSubFolderRGS, exist_ok=True)
        # Extraction images des MTS pour le RGS toutes les 10min (par défaut) ou autre
        for mts in MTSs:
            CurrentFrameToBeExtracted = int(1)
            # Ouverture de la vidéo
            video = cv2.VideoCapture(mts)
            # Calcul de la durée de la vidéo
            TotalFramesInTheVideo = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            CumulatedFrames = eachtime * 60 * fps
            while CurrentFrameToBeExtracted <= TotalFramesInTheVideo:
                video.set(cv2.CAP_PROP_POS_MSEC, CurrentFrameToBeExtracted)
                ret, frame = video.read()# Lire l'image du cadre actuel
                cv2.imwrite(os.path.join(PathSubFolderToAnalyze, 'RGS', str(os.path.basename(PathSubFolderToAnalyze)) + '_' + str(os.path.splitext(os.path.basename(mts))[0]) + '_' + str(int(CurrentFrameToBeExtracted)) + '.png'), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                CurrentFrameToBeExtracted = CurrentFrameToBeExtracted + int(eachtime * 60 * fps)
            video.release()