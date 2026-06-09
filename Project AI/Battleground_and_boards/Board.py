from Battleground_and_boards.Battleground import Battleground  # Importe la classe Battleground

class Board:
    def __init__(self):
        # Crée un plateau composé de 9 terrains (Battleground)
        self.battlefields = [Battleground() for _ in range(9)]  # Liste de 9 terrains
        self.resolved = [False] * 9   # Indique si chaque terrain est déjà résolu
        self.field_winner = [None] * 9  # Gagnant de chaque terrain (1, 2 ou None)

    def play_card(self, player: int, card, field_idx: int):
        """Joue une carte sur le terrain indiqué et met à jour l'état du plateau."""
        bf = self.battlefields[field_idx]  # Terrain ciblé
        if self.resolved[field_idx]:
            return -1.0, None, False  # Terrain déjà gagné : pénalité et pas de changement
        bf.add_card(player, card)  # Ajoute la carte au terrain

        # ----- Vérifie si le terrain peut être revendiqué -----
        # Cas normal : les deux joueurs ont maintenant 3 cartes
        if len(bf.player1_cards) == 3 and len(bf.player2_cards) == 3:
            winner = bf.evaluate_battlefield()  # Détermine le gagnant
            self.resolved[field_idx] = winner is not None
            self.field_winner[field_idx] = winner
            # Récompense locale : 0.1 si c'est le joueur courant, -0.1 sinon
            fr = 0.1 if winner == player else -0.1

        # Cas exceptionnel : vous avez 3 cartes, l'adversaire <3 et votre main est imbattable (non implémenté)
        elif (player == 1 and len(bf.player1_cards) == 3 and len(bf.player2_cards) < 3) or \
             (player == 2 and len(bf.player2_cards) == 3 and len(bf.player1_cards) < 3):
            # TODO: implémenter la revendication anticipée
            winner = None
            fr = 0.0
        else:
            # Le terrain n'est pas encore résolu
            return 0.0, None, False

        # ----- Vérifie si la partie entière est terminée -----
        counts = [self.field_winner.count(1), self.field_winner.count(2)]  # Nombre de terrains gagnés par chaque joueur
        done = False

        # Condition de victoire : 5 terrains au total
        for p_idx, cnt in enumerate(counts, start=1):
            if cnt >= 5:
                done = True
                tr = 1.0 if p_idx == player else -1.0  # Récompense finale
                return fr + tr, winner, True

        # Condition alternative : 3 terrains adjacents
        for p_idx in [1, 2]:
            wins = self.field_winner
            for i in range(7):
                if wins[i] == p_idx and wins[i+1] == p_idx and wins[i+2] == p_idx:
                    done = True
                    tr = 1.0 if p_idx == player else -1.0
                    return fr + tr, winner, True

        return fr, winner, False  # Partie non terminée

    def to_feature_vector(self, current_player: int, hand: list):
        """Encode l'état du plateau + la main courante en un vecteur numérique (pour IA)."""
        vec = []
        for idx, bf in enumerate(self.battlefields):
            cnt1, cnt2 = len(bf.player1_cards), len(bf.player2_cards)
            sum1 = sum(c.value for c in bf.player1_cards)
            sum2 = sum(c.value for c in bf.player2_cards)
            res = int(self.resolved[idx])       # 1 si terrain résolu
            winp = int(self.field_winner[idx] == current_player)  # 1 si terrain gagné par le joueur courant
            vec.extend([cnt1, cnt2, sum1, sum2, res, winp])
        # On ajoute la taille de la main
        vec.append(len(hand))
        return vec

# --- Résumé rapide ---
# La classe Board représente le plateau complet de Shotten Totten:
# • 9 terrains (Battleground) dans lesquels les joueurs posent leurs cartes.
# • play_card gère la pose d'une carte, la résolution d'un terrain et détecte
#   la fin de partie (victoire par 5 terrains gagnés ou 3 terrains adjacents).
# • to_feature_vector convertit l'état du jeu en vecteur pour entraîner un agent IA.
