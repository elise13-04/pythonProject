import unittest  # Importe le module de tests unitaires de la bibliothèque standard
from Battleground_and_boards.Battleground import Battleground  # Importe la classe Battleground
from Cards_and_Decks.Card import Card  # Importe la classe Card

class TestBattleground(unittest.TestCase):
    """Tests unitaires pour la classe Battleground."""

    def test_incomplete_hands(self):
        """Si l'un des joueurs n'a pas encore 3 cartes, evaluate_battlefield renvoie None."""
        bg = Battleground()  # Instancie un nouvel objet Battleground
        # Joueur 1 pose 3 cartes, joueur 2 seulement 2
        for c in (Card(1, 'Red'), Card(2, 'Blue'), Card(3, 'Green')):
            bg.add_card(1, c)  # Ajoute successivement trois cartes au joueur 1
        for c in (Card(4, 'Yellow'), Card(5, 'Purple')):
            bg.add_card(2, c)  # Ajoute deux cartes au joueur 2
        self.assertIsNone(bg.evaluate_battlefield())  # Vérifie que la méthode renvoie None

    def test_rank_order(self):
        """
        Vérifie que l'ordre des rangs est correct :
        quinte-flush > brelan > flush > suite > somme seule
        """
        # Prépare des mains pour chaque rang
        sf = [Card(3, 'Red'), Card(4, 'Red'), Card(5, 'Red')]      # quinte-flush
        three = [Card(7, 'Red'), Card(7, 'Blue'), Card(7, 'Green')]  # brelan
        flush = [Card(2, 'Blue'), Card(5, 'Blue'), Card(9, 'Blue')]   # couleur
        straight = [Card(10, 'Red'), Card(11, 'Blue'), Card(12, 'Green')]  # suite
        simple = [Card(1, 'Red'), Card(4, 'Blue'), Card(6, 'Green')]     # somme seule

        # quinte-flush bat brelan
        bg = Battleground()
        for c in sf:
            bg.add_card(1, c)  # J1 reçoit la quinte-flush
        for c in three:
            bg.add_card(2, c)  # J2 reçoit le brelan
        self.assertEqual(bg.evaluate_battlefield(), 1)  # Le gagnant doit être le joueur 1

        # brelan bat flush
        bg = Battleground()
        for c in three:
            bg.add_card(1, c)  # J1 reçoit le brelan
        for c in flush:
            bg.add_card(2, c)  # J2 reçoit la couleur
        self.assertEqual(bg.evaluate_battlefield(), 1)  # Le brelan gagne

        # couleur (rank 3) bat suite (rank 2)
        bg = Battleground()
        for c in flush:
            bg.add_card(1, c)  # J1 reçoit la couleur
        for c in straight:
            bg.add_card(2, c)  # J2 reçoit la suite
        self.assertEqual(bg.evaluate_battlefield(), 1)  # J1 gagne

        # suite bat somme seule
        bg = Battleground()
        for c in straight:
            bg.add_card(1, c)  # J1 reçoit la suite
        for c in simple:
            bg.add_card(2, c)  # J2 reçoit la main simple
        self.assertEqual(bg.evaluate_battlefield(), 1)  # La suite gagne

    def test_sum_tiebreak_and_play_order(self):
        """
        Si deux mains de même rang ont la même somme,
        c'est celui qui a fini de poser sa 3e carte en premier qui gagne.
        """
        # Deux mains différentes mais même somme = 9
        h1 = [Card(1, 'Red'), Card(3, 'Blue'), Card(5, 'Green')]  # somme = 9
        h2 = [Card(2, 'Red'), Card(2, 'Blue'), Card(5, 'Yellow')]  # somme = 9

        # Cas où joueur1 termine sa main avant joueur2 → J1 gagne
        bg = Battleground()
        for c in h1:
            bg.add_card(1, c)  # J1 pose ses 3 cartes
        for c in h2:
            bg.add_card(2, c)  # J2 pose ses 3 cartes après
        self.assertEqual(bg.evaluate_battlefield(), 1)  # J1 remporte le tie-break

        # Cas inverse : joueur2 termine avant joueur1 → J2 gagne
        bg = Battleground()
        for c in h2:
            bg.add_card(2, c)  # J2 pose ses cartes en premier
        for c in h1:
            bg.add_card(1, c)  # J1 pose ensuite
        self.assertEqual(bg.evaluate_battlefield(), 2)  # J2 gagne

    def test_perfect_tie(self):
        """
        Simule une égalité parfaite en forçant un même ordre de 3e carte pour les deux joueurs.
        """
        bg = Battleground()
        # Deux mains identiques (somme et rang identiques)
        h = [Card(2, 'Red'), Card(4, 'Blue'), Card(6, 'Green')]
        for c in h:
            bg.add_card(1, c)  # J1 pose une carte
            bg.add_card(2, c)  # J2 pose la même carte juste après
        # Forçage de l'ordre de 3e pose identique
        bg.player1_third_play_order = 6  # Ordre fictif égal
        bg.player2_third_play_order = 6
        self.assertIsNone(bg.evaluate_battlefield())  # En cas d'égalité parfaite

    # Résumé du fichier:
    # - test_incomplete_hands: évalue le comportement quand un joueur n'a pas 3 cartes (retour None).
    # - test_rank_order: vérifie la hiérarchie des combinaisons de mains (quinte-flush, brelan, couleur, suite, simple).
    # - test_sum_tiebreak_and_play_order: en cas d'égalité de rang et de somme, le gagnant est celui qui termine sa main en premier.
    # - test_perfect_tie: simule une égalité parfaite (même rang, même somme, même ordre de pose) et retourne None.

if __name__ == '__main__':
    unittest.main()  # Lance tous les tests si exécuté directement