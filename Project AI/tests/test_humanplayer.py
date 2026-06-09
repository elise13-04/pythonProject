import unittest
from unittest.mock import patch
import sys, os

# Ajoute le répertoire parent pour importer HumanPlayer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Players.HumanPlayer import HumanPlayer

class DummyBoard:
    """Plateau factice pour test, non utilisé dans make_move."""
    pass

class TestHumanPlayer(unittest.TestCase):
    """Tests unitaires pour la classe HumanPlayer (make_move)."""

    def setUp(self):
        # Crée un HumanPlayer avec une main prédéfinie
        self.hp = HumanPlayer('Alice')
        self.hp.hand = ['C1', 'C2', 'C3']
        self.board = DummyBoard()

    @patch('builtins.input')
    @patch('builtins.print')
    def test_make_move_valid_inputs(self, mock_print, mock_input):
        """Vérifie que make_move affiche la main et retourne la bonne paire."""
        # Simule deux appels input successifs pour field=4, card_idx=2
        mock_input.side_effect = ['4', '2']
        move = self.hp.make_move(self.board)
        # Vérifie que print affiche la main
        mock_print.assert_called_with("Your hand:", ['C1', 'C2', 'C3'])
        # Vérifie que make_move renvoie le tuple attendu
        self.assertEqual(move, (4, 2))

    @patch('builtins.input')
    @patch('builtins.print')
    def test_make_move_invalid_input_raises(self, mock_print, mock_input):
        """Vérifie que make_move lève ValueError pour entrée non entière."""
        # Simule une entrée non numérique pour field
        mock_input.side_effect = ['a', '0']
        with self.assertRaises(ValueError):
            self.hp.make_move(self.board)

    @patch('builtins.input')
    @patch('builtins.print')
    def test_make_move_empty_hand(self, mock_print, mock_input):
        """Si la main est vide, make_move renvoie quand même les indices saisis."""
        self.hp.hand = []
        # Simule inputs valides
        mock_input.side_effect = ['0', '0']
        move = self.hp.make_move(self.board)
        self.assertEqual(move, (0, 0))

# --- Résumé ---
# Ce fichier teste la méthode make_move de HumanPlayer :
# 1. test_make_move_valid_inputs : pour deux inputs valides, renvoie (field, card).
# 2. test_make_move_invalid_input_raises : entrée non entière lance ValueError.
# 3. test_make_move_empty_hand : main vide, renvoie quand même (0,0).

if __name__ == '__main__':
    unittest.main()
