import unittest
import sys, os, importlib.util, types
import numpy as np

# Crée des modules factices pour éviter les importations réelles et les boucles circulaires
# Dummy Deck
mod_deck = types.ModuleType('Cards_and_Decks.Deck')
class DummyDeck:
    def __init__(self):
        self.cards = list(range(54))
    def draw(self, n):
        # Simule le tirage de n cartes
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn
mod_deck.Deck = DummyDeck
sys.modules['Cards_and_Decks.Deck'] = mod_deck

# Dummy Board
mod_board = types.ModuleType('Battleground_and_boards.Board')
class DummyBoard:
    def __init__(self):
        pass
    def to_feature_vector(self, current_player, hand):
        # Renvoie un vecteur de la bonne taille: 9*6 + len(hand)
        return [0] * (9*6 + len(hand))
mod_board.Board = DummyBoard
sys.modules['Battleground_and_boards.Board'] = mod_board

# Dummy HumanPlayer
mod_human = types.ModuleType('Players.HumanPlayer')
class DummyHuman:
    def __init__(self, name):
        self.hand = []
    def play_card(self, board, field, card_idx, player_num):
        return (0.1, None, False)
    def refill_hand(self, deck):
        pass
mod_human.HumanPlayer = DummyHuman
sys.modules['Players.HumanPlayer'] = mod_human

# Dummy AIPlayer
mod_ai = types.ModuleType('Reinforcement_Learning_Agent_and_AIPlayer.AIPlayer')
class DummyAI:
    def __init__(self, name, policy):
        self.hand = []
    def play_card(self, board, field, card_idx, player_num):
        return (0.2, None, False)
    def refill_hand(self, deck):
        pass
mod_ai.AIPlayer = DummyAI
sys.modules['Reinforcement_Learning_Agent_and_AIPlayer.AIPlayer'] = mod_ai

# Charge dynamiquement Game.py
game_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Game', 'Game.py'))
spec = importlib.util.spec_from_file_location('game_module', game_path)
game_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game_module)
Game = game_module.Game

class TestGame(unittest.TestCase):
    """Tests unitaires pour la classe Game avec dépendances moquées."""

    def setUp(self):
        # Instancie Game sans politique
        self.game = Game(ai_policy=None)

    def test_reset_initial_state(self):
        """reset() doit initialiser deck, board et mains."""
        state = self.game.reset()
        # Deck a tiré 12 cartes sur 54
        self.assertEqual(len(self.game.deck.cards), 42)
        # Chaque joueur a 6 cartes
        for p in self.game.players:
            self.assertEqual(len(p.hand), 6)
        # State est un numpy array de floats
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.dtype, float)
        # Longueur du vecteur = 9*6 + 6 = 60
        self.assertEqual(state.size, 9*6 + 6)

    def test_get_actions(self):
        """get_actions() énumère correctement toutes les actions légales."""
        self.game.reset()
        actions = self.game.get_actions()
        hand_size = len(self.game.players[self.game.current].hand)
        self.assertEqual(len(actions), 9 * hand_size)
        # Chaque combinaison de champ et index doit être présente
        expected = [(f, c) for f in range(9) for c in range(hand_size)]
        self.assertCountEqual(actions, expected)

    def test_step_turn_switch_and_reward(self):
        """step() doit renvoyer next_state, reward, done, info et changer de joueur."""
        self.game.reset()
        # Assure main initiale non vide
        self.assertTrue(self.game.players[0].hand)
        # Joue action (0,0)
        next_state, reward, done, info = self.game.step((0, 0))
        # Reward doit venir de DummyHuman (0.1) ou DummyAI (0.2) selon current
        self.assertIn(reward, (0.1, 0.2))
        self.assertFalse(done)
        self.assertIsInstance(info, dict)
        # current doit avoir basculé
        self.assertEqual(self.game.current, 1)
        # next_state doit être numpy array de bonne taille
        self.assertIsInstance(next_state, np.ndarray)
        self.assertEqual(next_state.size, 9*6 + len(self.game.players[1].hand))

    def test_is_over_always_false(self):
        """is_over() doit toujours retourner False."""
        self.assertFalse(self.game.is_over())

# Pas de test render pour éviter interaction avec print et structure de Board

if __name__ == '__main__':
    unittest.main()

# --- Résumé ---
# Ce fichier teste Game en isolant ses dépendances via des modules factices :
# - DummyDeck simule le paquet
# - DummyBoard fournit un vecteur d'état fixe
# - DummyHuman et DummyAI simulent play_card et refill_hand
# Il couvre reset(), get_actions(), step() et is_over().
