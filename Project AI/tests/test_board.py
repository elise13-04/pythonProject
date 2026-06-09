import unittest  # Importe le module unittest pour créer des tests unitaires
from Battleground_and_boards.Board import Board  # Importe la classe Board qui gère le plateau de jeu
from Cards_and_Decks.Card import Card  # Importe la classe Card pour créer des objets carte

class TestBoard(unittest.TestCase):  # Définit une suite de tests pour la classe Board
    """Tests unitaires pour valider le comportement de Board."""

    def setUp(self):
        """Configure un nouveau plateau avant chaque test."""
        self.board = Board()  # Crée une instance fraîche de Board pour l'isolation des tests

    def test_initial_state(self):
        """Vérifie que le plateau démarre sans terrains résolus ni gagnants."""
        # Tous les terrains doivent être marqués non résolus
        self.assertEqual(self.board.resolved, [False] * 9)
        # Aucun gagnant n'est enregistré pour chaque terrain
        self.assertEqual(self.board.field_winner, [None] * 9)

    def test_play_card_incomplete(self):
        """Vérifie le comportement lorsqu'une seule carte est jouée sur un terrain."""
        card = Card(3, 'Red')  # Crée une carte de valeur 3 et couleur Red
        # Joue la carte sur le terrain 0 par le joueur 1
        reward, winner, done = self.board.play_card(1, card, 0)
        # Attendu : pas de résolution, pas de récompense, pas de gagnant, partie non terminée
        self.assertEqual((reward, winner, done), (0.0, None, False))

    def test_play_card_resolution(self):
        """Teste la résolution d'un terrain quand chaque joueur pose 3 cartes."""
        # Main du joueur 1 (somme élevée)
        hand1 = [Card(5, 'Red'), Card(5, 'Blue'), Card(5, 'Green')]  # Total = 15
        # Main du joueur 2 (somme faible)
        hand2 = [Card(1, 'Red'), Card(1, 'Blue'), Card(1, 'Green')]  # Total = 3
        # Joueur 1 pose ses 3 cartes sur le terrain 0
        for c in hand1:
            self.board.play_card(1, c, 0)
        # Après trois cartes de J1, le terrain n'est pas encore résolu
        self.assertFalse(self.board.resolved[0])
        # Joueur 2 pose ses 3 cartes et on capture le résultat sur la dernière
        for c in hand2:
            reward, winner, done = self.board.play_card(2, c, 0)
        # Le terrain doit maintenant être marqué comme résolu
        self.assertTrue(self.board.resolved[0])
        # Le gagnant local devrait être le joueur 1 (somme 15 vs 3)
        self.assertEqual(self.board.field_winner[0], 1)
        # La récompense locale pour J2 (perdant) doit être -0.1
        self.assertAlmostEqual(reward, -0.1)
        # Le ``winner`` retourné doit être 1
        self.assertEqual(winner, 1)
        # La partie ne doit pas être terminée après un seul terrain
        self.assertFalse(done)

    def test_play_on_resolved_field(self):
        """Vérifie qu'on ne peut pas jouer sur un terrain déjà gagné."""
        # Prépare une victoire sur le terrain 1
        h1 = [Card(2, 'Red'), Card(2, 'Blue'), Card(2, 'Green')]  # Main de J1
        h2 = [Card(1, 'Red'), Card(1, 'Blue'), Card(1, 'Green')]  # Main de J2
        # J1 résout d'abord le terrain 1
        for c in h1:
            self.board.play_card(1, c, 1)
        for c in h2:
            self.board.play_card(2, c, 1)
        # Vérifie que le terrain est marqué comme résolu
        self.assertTrue(self.board.resolved[1])
        # Tentative de J2 de jouer encore sur le terrain 1
        card = Card(3, 'Yellow')
        reward, winner, done = self.board.play_card(2, card, 1)
        # Doit renvoyer une pénalité -1.0, aucun gagnant, partie non terminée
        self.assertEqual((reward, winner, done), (-1.0, None, False))

    def test_global_five_field_win(self):
        """Teste la victoire globale lorsqu'un joueur gagne 5 terrains."""
        # Simule 5 victoires locales pour le joueur 1
        for idx in range(5):
            # Cartes de J1 qui assurent la victoire locale
            h1 = [Card(4, 'Red')] * 3
            # Cartes de J2 qui perdent localement
            h2 = [Card(1, 'Blue')] * 3
            for c in h1:
                self.board.play_card(1, c, idx)  # J1 pose ses cartes
            # J2 pose ses cartes et capture la troisième pose
            for c in h2:
                reward, winner, done = self.board.play_card(2, c, idx)
        # Après 5 terrains, la partie doit être considérée comme terminée
        self.assertTrue(done)
        # Récompense finale pour J2 : local -0.1 + match -1.0 = -1.1
        self.assertAlmostEqual(reward, -1.1)
        # Le gagnant global doit rester J1
        self.assertEqual(winner, 1)

    def test_to_feature_vector(self):
        """Vérifie la conversion de l'état du jeu en vecteur de caractéristiques."""
        hand = [Card(1, 'Red'), Card(2, 'Blue')]  # Main de deux cartes
        vec = self.board.to_feature_vector(current_player=2, hand=hand)
        # La longueur attendue : 9 terrains × 6 valeurs + 1 taille de main = 55
        self.assertEqual(len(vec), 9 * 6 + 1)
        # Pour le terrain 0, aucun jeu : [cnt1, cnt2, sum1, sum2, res, winp] = [0,0,0,0,0,0]
        self.assertEqual(vec[0:6], [0, 0, 0, 0, 0, 0])
        # Le dernier élément du vecteur est la taille de la main
        self.assertEqual(vec[-1], 2)

if __name__ == '__main__':
    unittest.main()  # Lance la suite de tests lorsque le script est exécuté directement

# --- Résumé ---
# Ce fichier définit plusieurs tests unitaires pour s'assurer que :
# 1. Le plateau démarre sans états marqués.
# 2. On ne peut pas résoudre un terrain tant que chaque joueur n'a pas 3 cartes.
# 3. La résolution locale suit la somme des valeurs des cartes.
# 4. Il est impossible de jouer sur un terrain déjà gagné.
# 5. Un joueur qui gagne 5 terrains remporte la partie globale.
# 6. L'état du jeu peut être encodé sous forme de vecteur utile pour un agent IA.
