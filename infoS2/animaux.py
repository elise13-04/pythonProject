# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import randint
import sys
from abc import ABC, abstractmethod
# from dprint import dprint

def sign(x):
    if x<0:
        return -1
    elif x==0:
        return 0
    else:
        return 1


class Animal(ABC):
    """
    Classe décrivant les comportement par défaut des animaux. Peut-être 
    utilisée en l'état ou sous classée pour définir des comportements de
    déplacement différents.
    """
    def __init__(self, abscisse, ordonnee, eco, capacite=20):
        """
        Crée un animal aux coordonnées désirées.
        
        Paramètres
        ----------
        abscisse, ordonnée: int
            Les coordonnées auxquelles l'animal sera créé.
            
        capacité: int
            niveau de santé maximal de l'animal. Vaut 10 par défaut.
        """
        
        self.__sante = randint(capacite//2, capacite)
        self._max = capacite
        self._eco = eco
        self.coords = abscisse, ordonnee

    def __str__(self):
        """
        Affiche l'état courant de l'animal.
        
        Paramètres
        ----------
        Aucun
        
        Renvoie
        -------
        s: str
            La chaîne de caractères qui sera affichée via ''print''
        """
        return "%c : position (%i, %i) etat %i/%i"%(
            self.car(), self.x, self.y,
            self.sante, self._max
            )
    
    def car(self):
        """
        Renvoie l'identifiant de l'espèce de l'animal.
        
        Paramètres
        ----------
        Aucun
        
        Renvoie
        -------
        c: str
            Le caractère représentant l'animal.
        """
        return 'A'    

    def manger(self):
        """
        L'animal perd un niveau de sante, puis se nourrit s'il se trouve
        sur une case de nourriture. Il affiche "Je meurs de faim" si sa
        sante est à 0.
        """
        self.sante -= 1
        if self._eco.case(self.x, self.y)==1:
            self.sante = self._max
        if self.sante<0:
            print(str(self)+". Je meurs de faim")

    @abstractmethod
    def bouger(self):
        """
        À instancier dans les classes filles
        """
        ...

    def mouvAlea(self):
        self.coords = (self.x+randint(-3,4),
                       self.y+randint(-3,4))

    def mouvNour(self):
        v, xmin, ymin = self._eco.vue(self.x, self.y, 4)
        liste_nour = []
        for i in range(len(v)):
            for j in range(len(v[0])):
                if v[i][j] == 1:
                    cx = i+xmin
                    cy = j+ymin
                    d = max(abs(cx-self.x), abs(cy-self.y))
                    liste_nour.append((d, cx, cy))
        if liste_nour == []:
            self.mouvAlea()
        else:
            np.random.shuffle(liste_nour)
            liste_nour.sort()
            objx, objy = liste_nour[0][1:]
            self.coords = (self.x + sign(objx-self.x),
                           self.y + sign(objy-self.y))
        
    @property
    def coords(self):
        """
        coords: tuple
            Les coordonnées de l'animal sur le plateau de jeu
        """
        return self.__coords

    @property
    def x(self):
        """
        x: nombre entier
            Abscisse de l'animal
        """
        return self.coords[0]

    @property
    def y(self):
        """
        y: nombre entier
            Abscisse de l'animal
        """
        return self.coords[1]

    @coords.setter
    def coords(self, nouv_coords):
        """
        Met à jour les coordonnées de l'insecte.
        Garantit qu'elles arrivent dans la zone définie par
        l'écosystème self._eco.
    
        Paramètres
        ----------
        Aucun
        """
        x, y = nouv_coords
        x = min(x, self._eco.dims[0]-1)
        x = max(x, 0)
        y = min(y, self._eco.dims[1]-1)
        y = max(y, 0)
        self.__coords = (x, y)

    @property
    def sante(self):
        """
        sante: float
            Le niveau de santé de l'animal. Si ce niveau arrive à 0 l'animal
            est marqué comme mort et sera retiré du plateau de jeu
        """
        return self.__sante
    
    @sante.setter
    def sante(self, value):
        if value <= self._max:
            self.__sante = value
        if value <= 0:  # <= car certaines cases enlèvent plus de 1 en santé
            value = 0   # ce qui gèrera les décès plus tard


class Fourmi(Animal):
    def car(self):
        return 'F'
        
    def bouger(self):
        """
        Effectue un mouvement aléatoire (défini dans la superclasse) si 
        sante>=3. Essaye de se rapprocher d'une case vers une réserve de
        nourriture sinon.
        """
        if self.sante>=4:
            self.mouvAlea()
        else:
            self.mouvNour()
    
class Cigale(Animal):
    def __init__(self, x, y, eco):      # *args, **kwargs):
        """Le constructeur de la classe Cigale.
        x, y : int

        Note : Les lignes commentées dans le code source sont des
        façons alternatives d'invoquer le constructeur de la
        classe-mère.
        """
        # super().__init__(*args, **kwargs)
        # super().__init__(args[0], args[1])
        super().__init__(x, y, eco)
        self.sante = self._max
    
    def car(self):
        return 'C'

    def bouger(self):
        """
        Probabilité 1/3 de danser, 1/3 de chanter, 1/3 d'avoir
        un comportement semblable à une Fourmi.
        """
        action = randint(3)
        if action == 1:
            print("Je danse")
        elif action == 2:
            print("Je chante")
        elif self.sante>=8:
            self.mouvAlea()
        else:
            self.mouvNour()

