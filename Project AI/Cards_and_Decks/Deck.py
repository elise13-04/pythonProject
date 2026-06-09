import random  # Module pour le mélange aléatoire des cartes
from Cards_and_Decks.Card import Card  # Import de la classe Card

class Deck:
    def __init__(self):
        # 6 couleurs disponibles dans Shotten-Totten
        self.colors = ['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange']
        # Valeurs numériques de 1 à 9 pour chaque couleur
        self.values = list(range(1, 10))
        # Génère l'ensemble des 54 cartes en compréhension de liste
        self.cards = [Card(v, c) for c in self.colors for v in self.values]
        self.shuffle()  # Mélange initial du paquet

    def shuffle(self):
        """Mélange le paquet en place."""
        random.shuffle(self.cards)

    def draw(self, n=1):
        """Tire *n* cartes (par défaut 1) et les renvoie sous forme de liste."""
        drawn = []  # Stockage des cartes tirées
        for _ in range(n):
            if not self.cards:  # Arrête si le paquet est vide
                break
            drawn.append(self.cards.pop())  # Retire la dernière carte de la liste (efficace)
        return drawn

    def draw_card(self):
        """Version pratique pour tirer une seule carte (ou None si le paquet est vide)."""
        return self.cards.pop() if self.cards else None

    def is_empty(self):
        """Renvoie True si le paquet n'a plus de cartes."""
        return len(self.cards) == 0

    def __len__(self):
        """Permet d'utiliser len(deck) pour connaître la taille restante."""
        return len(self.cards)

    def __repr__(self):
        """Représentation textuelle conviviale pour le débogage."""
        return f"<Deck: {len(self.cards)} cards remaining>"

# -------------------------------------------------------
# Résumé rapide
# -------------------------------------------------------
#   • Crée un paquet de 54 cartes (6 couleurs x 9 valeurs) dès l'initialisation.
#   • Mélange le paquet automatiquement.
#   • Fournit des méthodes pour mélanger à nouveau, tirer une ou plusieurs cartes,
#     vérifier si le paquet est vide, obtenir la taille restante et afficher
#     l'état du paquet de façon lisible.
