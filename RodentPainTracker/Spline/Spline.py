import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def Spline(df, X, Y=None):# /!\ prend en entrée une (1D) ou deux (2D) colonne(s) de dataframe panda
    if Y is not None:
        print("Lissage 2D")
        # Teste si en entrée ce sont des colonnes de dataframe
        if str(X) in df.columns and str(Y) in df.columns:
            # Récupérer les données de X et Y en tant que listes
            x = df[str(X)].tolist()
            y = df[str(Y)].tolist()
            #x = [int(e) for e in df[str(X)]]
            #y = [int(e) for e in df[str(Y)]]
            print(x)
            print(y)
            # Paramétrage
            spline_smoothness = 1# Définition de la précision de la spline, [0;∞]
            spline_degree = 3# 1 pour linéaire, 2 pour quadratique, 3 pour cubique, 4-5 splines d'ordre plus haut
            # Effectuer l'interpolation spline
            tck, u = splprep([x, y], s=spline_smoothness, k=spline_degree)
            SplinedData = splev(u, tck)
            xx, yy = SplinedData
            #xx = [int(e) for e in xx]
            #yy = [int(e) for e in yy]
            # Créer de nouvelles colonnes pour les coordonnées splinées dans le DataFrame
            df[str(X)+'Splined'] = xx
            df[str(Y)+'Splined'] = yy
            #
            # Plot
            # Affichage de la trajectoire originale sous forme de points
            plt.plot(x, y, 'b.', label='Trajectoire 2D originale')
            # Affichage de la trajectoire lissée sous forme d'une courbe incurvée
            plt.plot(xx, yy, 'r.-', label='Trajectoire 2D lissée')
            plt.title("Lissage de trajectoire 2D par la méthode des splines")
            plt.legend()
            plt.show()
            #
            return df
        #
        if 1 == 0:
            x = [int(e) for e in df[X]]
            y = [int(e) for e in df[Y]]
            print(x)
            print(y)
            #
            print(str("x est : ") + str(type(x)))
            print(str("y est : ") + str(type(y)))
            #
            # Répéter la méthode de spline i fois
            i = 1
            for _ in range(i):
                # Utilisation de la méthode des splines pour ajuster une courbe à la trajectoire
                tck, u = splprep([x, y], s=spline_smoothness, k=spline_degree)
                new_points = splev(u, tck)
                xx, yy = new_points
            xx = [int(e) for e in xx]
            yy = [int(e) for e in yy]
            df[X] = xx
            df[Y] = yy
            # Affichage de la trajectoire originale sous forme de points
            plt.plot(x, y, 'b.', label='Trajectoire 2D originale')
            # Affichage de la trajectoire lissée sous forme d'une courbe incurvée
            plt.plot(xx, yy, 'r.-', label='Trajectoire 2D lissée')  
            #plt.plot(df_smooth['x'], df_smooth['y'], '-o', markevery=50, label='Trajectoire lissée')
            plt.title("Lissage de trajectoire 2D par la méthode des splines")
            plt.legend()
            plt.show(block=True)
    else:
        print("Lissage 1D")
        # Liste de valeurs entières à lisser
        x = [int(e) for e in df[str(X)]]
        # Paramètres de lissage
        window_length = 25  # Longueur de la fenêtre de lissage
        polyorder = 3  # Ordre du polynôme d'ajustement
        # Conversion de la liste en un tableau NumPy
        values_array = np.array(x)
        # Application du lissage avec savgol_filter
        xx = savgol_filter(values_array, window_length, polyorder)
        # int
        xx = [int(e) for e in xx]
        # to_df
        df[str(X)] = xx
        # PLOT
        plt.clf()# efface les plots précédents
        # Plot des anciennes valeurs (points bleus)
        plt.plot(x, 'b.', label='Trajectoire 1D originale')
        plt.plot(xx, 'r.-', label='Trajectoire 2D lissée')
        plt.title("Lissage de trajectoire 1D par la méthode de Savitzky-Golay")
        plt.legend()
        plt.show(block=True)
        #
    return df