from Players.Player import Player  # On hérite des fonctionnalités de la classe de base Player

class HumanPlayer(Player):
    def make_move(self, board):
        """Demande à l'utilisateur quel coup jouer sur le plateau."""
        print("Your hand:", self.hand)  # Affiche la main actuelle au joueur
        # Sélection du champ de bataille sur lequel poser une carte (indice 0‑8)
        field = int(input("Choose battleground (0-8): "))
        # Sélection de la carte à jouer en indiquant son indice dans la main
        card = int(input(f"Choose card index (0-{len(self.hand)-1}): "))
        return field, card  # Retourne le couple (champ, carte) exploitable par Game.step

# -------------------------------------------------------------
# Résumé rapide
# -------------------------------------------------------------
# HumanPlayer hérite de Player et fournit une méthode make_move
# interactive: elle affiche la main du joueur humain et recueille
# via la console l'indice du champ (0‑8) puis l'indice de la carte
# à jouer dans sa main. Elle renvoie ces deux valeurs pour que la
# boucle de jeu puisse exécuter l'action choisie.
