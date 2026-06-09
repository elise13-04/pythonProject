import numpy as np  # Pour gérer le vecteur d'état sous forme de tableau NumPy
from Cards_and_Decks.Deck import Deck  # Paquet de cartes
from Battleground_and_boards.Board import Board  # Plateau de 9 champs de bataille
from Players.HumanPlayer import HumanPlayer  # Joueur humain interactif
from Reinforcement_Learning_Agent_and_AIPlayer.AIPlayer import AIPlayer  # Joueur IA basé sur une politique RL

class Game:
    def __init__(self, ai_policy=None):
        # Condition de victoire : 5 lignes (ou 3 adjacentes, géré dans Board)
        self.victory_condition = 5
        # Politique d'action de l'IA (réseau ou dictionnaire de Q‑values)
        self.ai_policy = ai_policy
        self.reset()  # Initialise une nouvelle partie

    def reset(self):
        """Réinitialise complètement la partie (nouveau paquet, plateau et mains)."""
        self.deck = Deck()              # Crée et mélange un nouveau paquet de 54 cartes
        self.board = Board()            # Nouveau plateau vide
        # Instancie les deux joueurs : humain (joueur1) et IA (joueur2)
        self.players = [HumanPlayer("Player 1"), AIPlayer("AI", self.ai_policy)]
        # Donne 6 cartes de départ à chaque joueur
        for p in self.players:
            p.hand = self.deck.draw(5)
        self.current = 0  # 0 = index du joueur courant (J1 au départ)
        return self.get_state()  # Renvoie l'état initial pour l'agent RL

    def get_state(self):
        """Encode l'état actuel en vecteur numérique –main + plateau– pour l'agent."""
        return np.array(
            self.board.to_feature_vector(
                current_player=self.current+1,
                hand=self.players[self.current].hand
            ),
            dtype=float
        )

    def get_actions(self):
        """Renvoie la liste exhaustive des actions légales (champ, index de carte)."""
        hand_size = len(self.players[self.current].hand)
        return [(f, c) for f in range(9) for c in range(hand_size)]

    def step(self, action):
        """Joue une action et retourne (next_state, reward, done, info) façon gym."""
        field, card_idx = action             # Décompose l'action
        player_num = self.current + 1        # Numéro 1ou2 du joueur courant
        player = self.players[self.current]  # Instance du joueur courant
        # Le joueur joue la carte : renvoie reward local, gagnant de la ligne, done global
        reward, win, done = player.play_card(self.board, field, card_idx, player_num)
        player.refill_hand(self.deck)        # Pioche pour revenir à 6 cartes si possible
        self.current ^= 1  # Change de joueur (0 -> 1, 1 -> 0)
        next_state = self.get_state()        # Nouvel état pour l'agent RL
        return next_state, reward, done, {}

    def is_over(self):
        """Méthode placeholder : le status de fin est géré dans Board.play_card."""
        return False

    def render(self):
        """Affiche une vue texte du plateau et des mains, pour débogage/console."""
        for i, bf in enumerate(self.board.battlefields):
            print(
                f"[{i}] P1={bf.player1_cards} vs P2={bf.player2_cards} | "
                f"Resolved={self.board.resolved[i]} Winner={self.board.field_winner[i]}"
            )
        print("Hands:", [p.hand for p in self.players], "Next:", self.current+1)

# ---------------------------------------------------------------
# Résumé rapide
# La classe Game orchestre une partie complète de Shotten-Totten:
#  • reset() crée un paquet, un plateau et distribue 6 cartes à chaque joueur.
#  • get_state() encode l'état courant en vecteur NumPy pour l'IA.
#  • get_actions() génère toutes les actions légales (choix du champ et de la carte).
#  • step() applique une action, met à jour l'état, récompense et fin de partie.
#  • render() affiche le plateau et les mains pour un suivi en console.
# Elle sert de couche d'interface entre un agent RL (AIPlayer) et la logique de jeu.
