import unittest  # Module de tests unitaires
import sys, os

# Ajoute le répertoire parent au PYTHONPATH pour importer Player
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Ajoute le répertoire parent au PYTHONPATH pour importer Player
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Players.Player import Player  # Importe la classe Player depuis le package Players  # Importe la classe Player définie dans player.py

# Classe factice pour simuler un deck
class DummyDeck:
    def __init__(self, cards):
        self._cards = list(cards)  # Copie des cartes disponibles
    def draw_card(self):
        # Renvoie la dernière carte ou None si vide
        return self._cards.pop() if self._cards else None
    def is_empty(self):
        # Indique si le deck est vide
        return len(self._cards) == 0

# Classe factice pour simuler un plateau
class DummyBoard:
    def __init__(self, ret):
        # ret est un tuple (reward, field_won, done) à renvoyer
        self._ret = ret
        self.calls = []  # Stocke les appels pour vérification
    def play_card(self, player_number, card, field_idx):
        # Enregistre les paramètres et renvoie le tuple prédéfini
        self.calls.append((player_number, card, field_idx))
        return self._ret

class TestPlayer(unittest.TestCase):
    """Tests unitaires pour la classe Player."""

    def setUp(self):
        # Crée un joueur de test
        self.player = Player('Alice')

    def test_refill_hand_until_max(self):
        """Vérifie que refill_hand remplit la main jusqu'à max_size ou épuisement."""
        # Deck avec 4 cartes
        deck = DummyDeck(['c1', 'c2', 'c3', 'c4'])
        # Pioche jusqu'à 6, mais deck ne contient que 4
        self.player.hand = []  # Assure main vide
        self.player.refill_hand(deck, max_size=6)
        # Doit avoir pioché 4 cartes
        self.assertEqual(len(self.player.hand), 4)
        # Deck doit être vide
        self.assertTrue(deck.is_empty())

    def test_refill_hand_until_full(self):
        """Vérifie que refill_hand s'arrête à max_size même si deck non vide."""
        deck = DummyDeck(['c1'] * 10)
        self.player.hand = []
        # Max_size plus petit que nombre de cartes disponibles
        self.player.refill_hand(deck, max_size=6)
        # Main doit contenir exactement 6 cartes
        self.assertEqual(len(self.player.hand), 6)
        # Deck doit avoir 4 cartes restantes
        self.assertEqual(len(deck._cards), 4)

    def test_play_card_capture_and_return(self):
        """Vérifie que play_card retire la carte, appelle board, met à jour captured et renvoie le bon tuple."""
        # Prépare main avec trois cartes
        self.player.hand = ['a', 'b', 'c']
        # Board qui renvoie victoire pour ce joueur
        dummy_board = DummyBoard((1.0, 1, False))
        # Joue la carte d'indice 1 ('b') sur le champ 2
        reward, field_won, done = self.player.play_card(dummy_board, field_idx=2, card_idx=1, player_number=1)
        # La carte 'b' doit avoir été retirée de la main
        self.assertNotIn('b', self.player.hand)
        # Le board doit avoir été appelé avec les bons paramètres
        self.assertIn((1, 'b', 2), dummy_board.calls)
        # Comme field_won == player_number, 2 doit être capturé
        self.assertIn(2, self.player.captured)
        # Les retours doivent correspondre au dummy
        self.assertEqual((reward, field_won, done), (1.0, 1, False))

    def test_play_card_no_capture(self):
        """Vérifie que play_card n'ajoute pas de capture si le champ n'est pas gagné."""
        self.player.hand = ['x', 'y']
        # Board qui renvoie victoire pour l'adversaire
        dummy_board = DummyBoard((0.0, 2, False))
        reward, field_won, done = self.player.play_card(dummy_board, field_idx=5, card_idx=0, player_number=1)
        # La carte d'indice 0 ('x') a été retirée
        self.assertNotIn('x', self.player.hand)
        # captured ne doit pas contenir 5
        self.assertNotIn(5, self.player.captured)
        # Reward et autres valeurs correspondent
        self.assertEqual((reward, field_won, done), (0.0, 2, False))

    def test_make_move_not_implemented(self):
        """make_move doit lever NotImplementedError par défaut."""
        with self.assertRaises(NotImplementedError):
            self.player.make_move(None)

# --- Résumé ---
# Ce fichier définit des tests pour la classe Player :
# 1. test_refill_hand_until_max : pioche jusqu'à épuisement si moins que max_size.
# 2. test_refill_hand_until_full : s'arrête à max_size même si deck non vide.
# 3. test_play_card_capture_and_return : play_card retire la carte, appelle board, enregistre capture et renvoie résultat.
# 4. test_play_card_no_capture : play_card sans capture si field_won diffère.
# 5. test_make_move_not_implemented : make_move lève NotImplementedError.

if __name__ == '__main__':
    unittest.main()  # Exécute les tests si lancé directement
