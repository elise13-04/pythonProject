import unittest  # Importe le module unittest pour les tests
from dataclasses import dataclass  # Nécessaire pour redéfinir la classe Card
from typing import Any

# Redéfinition de la classe Card pour le contexte des tests
@dataclass
class Card:
    value: int  # Valeur numérique de la carte (1-9)
    color: str  # Couleur de la carte (par ex. "rouge", "bleu")

    def __repr__(self) -> str:
        # Retourne une représentation compacte : initiale de la couleur + valeur
        return f"{self.color[0].upper()}{self.value}"

class TestCard(unittest.TestCase):  # Définit la suite de tests pour la classe Card
    """Tests unitaires pour valider la classe Card et sa représentation."""

    def test_dataclass_fields(self):
        """Vérifie que Card stocke correctement value et color."""
        card = Card(7, 'rouge')  # Création d'une carte de valeur 7, couleur rouge
        # On s'assure que les attributs existent et sont corrects
        self.assertEqual(card.value, 7)
        self.assertEqual(card.color, 'rouge')

    def test_repr_uppercase(self):
        """Vérifie que __repr__ renvoie la lettre de couleur en majuscule suivie de la valeur."""
        # Test pour plusieurs couleurs et valeurs
        examples = [
            (Card(5, 'bleu'), 'B5'),
            (Card(1, 'Jaune'), 'J1'),
            (Card(9, 'vert'), 'V9'),
        ]
        for c, expected in examples:
            # Chaque carte doit se représenter comme prévu
            self.assertEqual(repr(c), expected)

    def test_repr_non_letter_color(self):
        """Gère les couleurs commençant par un caractère non alphabétique."""
        card = Card(3, '1décor')  # Couleur commençant par un chiffre
        # La première lettre telle quelle + valeur
        self.assertEqual(repr(card), '13')  # devrais être '1' + '3'

    def test_repr_empty_color(self):
        """Comportement si la couleur est une chaîne vide."""
        card = Card(2, '')
        # repr tente d'accéder à color[0] -> IndexError attendu
        with self.assertRaises(IndexError):
            _ = repr(card)

# --- Résumé ---
# Ce fichier définit des tests pour la classe Card :
# 1. test_dataclass_fields : vérifie l'assignation de value et color.
# 2. test_repr_uppercase : assure la bonne sortie de __repr__ pour des couleurs alphabétiques.
# 3. test_repr_non_letter_color : teste une couleur démarrant par un chiffre.
# 4. test_repr_empty_color : s'assure qu'une couleur vide lève IndexError dans __repr__.

if __name__ == '__main__':
    unittest.main()  # Exécute la suite de tests si lancé directement
