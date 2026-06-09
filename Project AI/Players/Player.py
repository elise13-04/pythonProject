class Player:
    def __init__(self, name):
        self.name = name              # Nom du joueur (chaîne affichée dans l'interface)
        self.hand = []                # Cartes actuellement en main (max 6)
        self.captured = []            # Indices des champs remportés par ce joueur

    def refill_hand(self, deck, max_size=5):
        """Pioche des cartes jusqu'à atteindre max_size ou épuisement du deck."""
        while len(self.hand) < max_size and not deck.is_empty():
            self.hand.append(deck.draw_card())  # Ajoute la carte du dessus à la main

    def play_card(self, board, field_idx, card_idx, player_number):
        """
        Joue la carte d'indice card_idx sur le champ field_idx.
        Renvoie le reward, le gagnant du champ (si résolu) et un booléen done si la partie se termine.
        """
        card = self.hand.pop(card_idx)                    # Retire la carte choisie de la main
        reward, field_won, done = board.play_card(
            player_number, card, field_idx               # Passe la carte au plateau
        )
        # Si le champ vient d'être remporté par ce joueur, on l'ajoute à sa liste "captured"
        if field_won == player_number:
            self.captured.append(field_idx)
        return reward, field_won, done

    def make_move(self, board):
        """
        Méthode abstraite: doit être implémentée par les sous‑classes (IA ou humain) pour
        décider quel coup jouer sur le plateau.
        """
        raise NotImplementedError

# -------------------------------------------------------
# Résumé rapide
# -------------------------------------------------------
# Player est une classe de base représentant un participant (humain ou IA).
# - stocke son nom, sa main de cartes et la liste des champs déjà capturés.
# - refill_hand pioche jusqu'à six cartes après chaque tour.
# - play_card applique le coup au plateau, renvoie la récompense et met à jour les captures.
# - make_move est laissée abstraite afin que chaque type de joueur définisse sa stratégie.
