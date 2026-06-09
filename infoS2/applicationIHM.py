# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:38:26 2018

@author: toumiab
"""

import sys
from PyQt5 import QtGui, QtCore, QtWidgets, uic

from interface import Ui_principale_ihm
from ecosysteme import Ecosysteme
# l'approche par héritage simple de la classe QMainWindow (même type de notre fenêtre 
# créée avec QT Designer. Nous configurons après l'interface utilisateur 
# dans le constructeur (la méthode init()) de notre classe

class MonAppli(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
              
        
        # TO DO
        # Chargement de votre fenetre ui.
        self.ui=Ui_principale_ihm()
        self.ui.setupUi(self)

        self.ui.bouton_pas.clicked.connect(self.un_pas)
        self.ui.bouton_gen.clicked.connect(self.generer)
        self.ui.bouton_sim.clicked.connect(self.simuler)


    def un_pas(self):
        print("un_pas")
    def generer(self):
        def generer(self):
            # Récupération des dimensions du conteneur
            largeur = self.ui.conteneur.width()
            hauteur = self.ui.conteneur.height()

            # Calcul du nombre de nourritures (1/20ème de la taille du conteneur)
            # On prend la moyenne entre largeur et hauteur pour être plus équilibré
            taille_moyenne = (largeur + hauteur) // 2
            nb_nourritures = taille_moyenne // 20

            # Création de l'écosystème avec les paramètres spécifiés
            self.ecosys = Ecosysteme(nb_insectes=60, nb_tours=150,largeur=largeur,hauteur=hauteur,nb_nourritures=nb_nourritures)

            print(f"Écosystème généré : {largeur}x{hauteur} px, {nb_nourritures} nourritures")
    def simuler(self):
        print("simuler")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MonAppli()
    window.show()
    app.exec_()
