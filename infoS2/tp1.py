import numpy as np
from numpy.random import randint
import sys


# from dprint import dprint

class Animal():
    def __init__(self, abscisse, ordonnee, eco, capacite=20):
        self.__sante = randint(capacite // 2, capacite)
        self._max = capacite
        self._eco = eco
        self.coords = abscisse, ordonnee

    def __str__(self):
        return "%c : position (%i, %i) etat %i/%i" % (
            self.car(), self.x, self.y,
            self.sante, self._max
        )

    def car(self):
        return 'A'

    def manger(self):
        self.sante -= 1
        if self.x % 5 == 0 and self.y % 5 == 0:
            self.sante = self._max
        if self.sante < 0:
            print(str(self) + ". Je meurs de faim")

    def bouger(self):
        self.coords = (self.x + randint(-3, 3),
                       self.y + randint(-3, 3))

    @property
    def coords(self):
        return self.__coords

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @coords.setter
    def coords(self, nouv_coords):
        x, y = nouv_coords
        x = min(x, self._eco.dims[0] - 1)
        x = max(x, 0)
        y = min(y, self._eco.dims[1] - 1)
        y = max(y, 0)
        self.__coords = (x, y)

    @property
    def sante(self):
        return self.__sante

    @sante.setter
    def sante(self, value):
        if value <= self._max:
            self.__sante = value
        if value <= 0:  # <= car certaines cases enlÃ¨vent plus de 1 en santÃ©
            value = 0  # ce qui gÃ¨rera les dÃ©cÃ¨s plus tard


class Fourmi(Animal):
    def car(self):
        return 'F'

    def bouger(self):
        if self.sante >= 3:
            super().bouger()
        else:
            if self.x % 5 == 1 or self.x % 5 == 2:
                nx = self.x - 1
            elif self.x % 5 == 0:
                nx = self.x
            else:
                nx = self.x + 1
            if self.y % 5 == 1 or self.y % 5 == 2:
                ny = self.y - 1
            elif self.y % 5 == 0:
                ny = self.y
            else:
                ny = self.y + 1
            self.coords = (nx, ny)


class Cigale(Animal):
    def __init__(self, x, y, eco):  # *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # super().__init__(args[0], args[1])
        super().__init__(x, y, eco)
        self.sante = self._max

    def car(self):
        return 'C'

    def bouger(self):
        action = randint(3)
        if action == 1:
            print("Je danse")
        elif action == 2:
            print("Je chante")
        elif self.sante >= 3:
            super().bouger()
        else:
            if self.x % 5 == 1 or self.x % 5 == 2:
                nx = self.x - 1
            elif self.x % 5 == 0:
                nx = self.x
            else:
                nx = self.x + 1
            if self.y % 5 == 1 or self.y % 5 == 2:
                ny = self.y - 1
            elif self.y % 5 == 0:
                ny = self.y
            else:
                ny = self.y + 1
            self.coords = (nx, ny)

import numpy as np
from numpy.random import randint
import time

# il est inutile d'importer Animal :
from animaux import Fourmi, Cigale

"""
Ce module contient la définition de la classe principale servant à gérer le jeu
"""


class Ecosysteme(list):
    """
    Classe gérant le déroulement du jeu.
    """

    def __init__(self, nb_ins, nbt, xmax, ymax):
        self.__xmax = xmax
        self.__ymax = ymax
        self.nbtour = nbt
        for i in range(nb_ins):
            if randint(0, 2) == 0:
                self.append(Fourmi(randint(0, xmax), randint(0, ymax), self))
            else:
                self.append(Cigale(randint(0, xmax), randint(0, ymax), self))

    @property
    def dims(self):
        """
        Renvoies les dimensions du plateau de jeu
        """
        return (self.__xmax, self.__ymax)

    def __str__(self):
        """Affiche le plateau de jeu en mode texte suivant les codes couleur
        définis dans les sous classes de ''Terrain'' et les caractères définis
        dans les sous classes de ''Animal''.

        Paramètres
        ----------
        Aucun

        Renvoie
        -------
        s: string
            La chaîne de caractères qui sera affichée via ''print''

        Notes
        -----
        Pour l'affichage en couleur, le terminal utilisé pour
        l'affichage devra savoir gérer les codes d'échappement bash.
        """

        # decommenter la ligne choisie :
        # return self.strCol()  # Pour l'affichage en couleur
        return self.str2()  # Pour l'affichage sur deux caractères

    def strCol(self):
        """
        Conversion en chaîne pour affichage sur terminal bash en couleur
        """
        pos = {}
        for ins in self:
            pos[ins.coords] = ins.car()
        s = ""
        for i in range(self.__xmax):
            for j in range(self.__ymax):
                if i % 5 == 0 and j % 5 == 0:
                    s += "\x1b[102;31m"
                else:
                    s += "\x1b[43;31m"
                if (i, j) in pos:
                    s += pos[(i, j)]
                else:
                    s += "."
            s += "\x1b[0m\n"
        return s

    def str2(self):
        """
        Conversion en chaîne avec deux caractères par case.
        """
        pos = {}
        for ins in self:
            pos[ins.coords] = ins.car()
        s = ""
        for i in range(self.__xmax):
            for j in range(self.__ymax):
                if i % 5 == 0 and j % 5 == 0:
                    s += "#"
                else:
                    s += "."
                if (i, j) in pos:
                    s += pos[(i, j)]
                else:
                    s += " "
            s += "\n"
        return s

    def unTour(self):
        """
        Effectue toutes les actions liées à un tour de jeu.

        Paramètres
        ----------
        Aucun

        Renvoie
        -------
        Rien
        """

        # rnd.shuffle(self)    Utile si gestion des collisions
        for ins in self:  # fonctionne car Ecosysteme descend de list
            ins.bouger()
            ins.manger()

    def simuler(self):
        """
        Contrôle l'évolution du jeu, affiche le résultat de chaque tour dans
        un terminal.

        Paramètres
        ----------
        Aucun

        Renvoie
        -------
        Rien
        """

        for t in range(self.nbtour):
            print("### Tour %i ###" % (t))
            self.unTour()
            print(self)
            time.sleep(0.2)


if __name__ == "__main__":
    nbins = 12
    nbtour = 43
    ecosys = Ecosysteme(nbins, nbtour, 20, 15)
    print(ecosys)
    ecosys.simuler()