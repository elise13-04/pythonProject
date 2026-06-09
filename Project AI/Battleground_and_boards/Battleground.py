class Battleground:
    def __init__(self):
        # Initialise le champ de bataille pour une escarmouche entre deux joueurs
        self.player1_cards = []  # Cartes jouées par le joueur 1 (maximum 3)
        self.player2_cards = []  # Cartes jouées par le joueur 2 (maximum 3)
        self.play_count = 0      # Compteur global du nombre total de cartes posées
        self.player1_third_play_order = None  # Ordre auquel le joueur1 a posé sa 3ème carte
        self.player2_third_play_order = None  # Ordre auquel le joueur2 a posé sa 3ème carte
        self.winner = None       # Vainqueur du champ de bataille: 1, 2 ou None

    def add_card(self, player: int, card):
        # Ajoute une carte sur ce champ pour le joueur indiqué
        # — On vérifie d'abord que le joueur ne dépasse pas la limite de 3 cartes
        if player == 1 and len(self.player1_cards) >= 3:
            raise ValueError("Player 1 cannot play more than 3 cards here")
        if player == 2 and len(self.player2_cards) >= 3:
            raise ValueError("Player 2 cannot play more than 3 cards here")

        self.play_count += 1  # Incrément de l'ordre de pose global

        if player == 1:
            self.player1_cards.append(card)    # On stocke la carte côté joueur1
            # Si c'est sa 3èmecarte, on mémorise l'ordre de pose pour les départages
            if len(self.player1_cards) == 3:
                self.player1_third_play_order = self.play_count
        else:
            self.player2_cards.append(card)    # On stocke la carte côté joueur2
            if len(self.player2_cards) == 3:
                self.player2_third_play_order = self.play_count

    def evaluate_battlefield(self):
        # Évalue le champ uniquement si les deux joueurs ont posé 3 cartes
        if len(self.player1_cards) < 3 or len(self.player2_cards) < 3:
            return None  # Pas encore de vainqueur

        # Fonction interne: calcule le rang et la somme des valeurs d'une main
        def eval_hand(cards):
            vals = sorted(card.value for card in cards)  # Valeurs triées croissantes
            cols = [card.color for card in cards]        # Couleurs des cartes
            unique_vals = set(vals)
            unique_cols = set(cols)

            is_flush = len(unique_cols) == 1               # Même couleur → flush
            is_straight = (
                len(unique_vals) == 3 and                 # Trois valeurs distinctes
                vals[2] - vals[0] == 2 and                # Extrêmes espacés de 2
                vals[1] - vals[0] == 1                    # Valeurs consécutives
            )                                              # → suite
            is_three = len(unique_vals) == 1               # Trois mêmes valeurs → brelan

            # Barème: 5 = quinte flush, 4 = brelan, 3 = couleur, 2 = suite, 1 = somme simple
            if is_flush and is_straight:
                rank = 5
            elif is_three:
                rank = 4
            elif is_flush:
                rank = 3
            elif is_straight:
                rank = 2
            else:
                rank = 1
            return rank, sum(vals)  # Renvoie le rang et la somme des valeurs

        # Évaluation des deux mains
        r1, s1 = eval_hand(self.player1_cards)
        r2, s2 = eval_hand(self.player2_cards)

        # Comparaison directe des rangs
        if r1 > r2:
            self.winner = 1
        elif r2 > r1:
            self.winner = 2
        else:
            # Rang identique: on regarde la somme des valeurs
            if s1 > s2:
                self.winner = 1
            elif s2 > s1:
                self.winner = 2
            else:
                # Toujours à égalité: le premier à avoir terminé sa main l'emporte
                if self.player1_third_play_order < self.player2_third_play_order:
                    self.winner = 1
                elif self.player2_third_play_order < self.player1_third_play_order:
                    self.winner = 2
                else:
                    self.winner = None  # Égalité parfaite (cas théorique)
        return self.winner
# --- Résumé rapide ---
# La classe Battleground représente un terrain dans Shotten Totten.
# Chaque joueur peut y poser jusqu'à trois cartes. Quand les deux mains sont complètes,
# evaluate_battlefield compare les mains selon un classement (quinte flush, brelan, etc.).
# En cas d'égalité de rang, on compare la somme des valeurs, puis l'ordre de pose de la 3ᵉ carte.
# Le résultat est stocké dans self.winner (1, 2 ou None).
