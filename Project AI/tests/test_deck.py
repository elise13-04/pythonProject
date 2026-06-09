# test_deck.py

import unittest                          # Importe le module de tests unitaires de Python
from unittest.mock import patch         # Importe patch pour simuler des appels et vérifier leur exécution
from Cards_and_Decks.Card import Card  # Importe la classe Card à partir du module Card
from Cards_and_Decks.Deck import Deck  # Importe la classe Deck à partir du module Deck

class TestDeck(unittest.TestCase):     # Définit une classe de tests héritant de unittest.TestCase

    def test_init_creates_54_cards(self):
        deck = Deck()                                                      # Initialise un nouveau deck
        # 6 couleurs × 9 valeurs = 54 cartes
        self.assertEqual(len(deck), 54,                                   # Vérifie que len(deck) == 54
                         "Un nouveau deck doit contenir 54 cartes")
        # Toutes les cartes doivent être des instances de Card
        self.assertTrue(all(isinstance(c, Card) for c in deck.cards),      # Vérifie que chaque élément est une Card
                        "Tous les éléments de deck.cards doivent être des Card")

    def test_len_and_is_empty(self):
        deck = Deck()                                                      # Initialise un nouveau deck
        self.assertFalse(deck.is_empty(),                                  # Vérifie que le deck n'est pas vide au départ
                         "Un deck juste initialisé ne doit pas être vide")
        # Vider entièrement le deck
        _ = deck.draw(54)                                                  # Tire les 54 cartes restantes
        self.assertTrue(deck.is_empty(),                                   # Vérifie que deck.is_empty() renvoie True
                        "After drawing all cards, deck should be empty")
        self.assertEqual(len(deck), 0,                                     # Vérifie que len(deck) est maintenant 0
                         "len(deck) doit être 0 quand il est vide")

    def test_draw_reduces_length_and_returns_cards(self):
        deck = Deck()                                                      # Initialise un nouveau deck
        initial_len = len(deck)                                            # Sauvegarde la taille initiale
        drawn = deck.draw(5)                                               # Tire 5 cartes
        self.assertEqual(len(drawn), 5,                                    # Vérifie que 5 cartes ont été retournées
                         "draw(5) doit renvoyer 5 cartes")
        self.assertEqual(len(deck), initial_len - 5,                       # Vérifie que le deck a perdu 5 cartes
                         "len(deck) doit être réduit de 5 après draw(5)")
        # Les cartes tirées ne doivent plus être dans deck.cards
        for c in drawn:
            self.assertNotIn(c, deck.cards)                                # Vérifie que chaque carte tirée n'est plus dans le deck

    def test_draw_more_than_available(self):
        deck = Deck()                                                      # Initialise un nouveau deck
        # Tenter de tirer plus que ce qui reste
        drawn = deck.draw(60)                                              # Tire 60 cartes alors qu'il n'y en a que 54
        self.assertEqual(len(drawn), 54,                                   # Vérifie que seules 54 cartes sont retournées
                         "draw(60) ne peut renvoyer que les 54 cartes disponibles")
        self.assertTrue(deck.is_empty())                                   # Vérifie que le deck est vide après

    def test_draw_card_single_and_none(self):
        deck = Deck()                                                      # Initialise un nouveau deck
        top = deck.cards[-1]                                               # Récupère la "dernière" carte du deck
        c = deck.draw_card()                                               # Tire une carte
        self.assertEqual(c, top,                                           # Vérifie que c == top
                         "draw_card() doit retirer et renvoyer la dernière carte")
        # Vider le deck sauf une carte déjà tirée
        _ = deck.draw(53)                                                  # Tire les 53 cartes restantes
        self.assertIsNone(deck.draw_card(),                                # Tente de tirer une carte d'un deck vide
                          "draw_card() sur un deck vide doit renvoyer None")

    def test_shuffle_calls_random_shuffle(self):
        deck = Deck()                                                      # Initialise un nouveau deck
        original = list(deck.cards)                                        # Sauvegarde l'ordre original
        # Simule random.shuffle pour vérifier qu'il est appelé
        with patch('Cards_and_Decks.Deck.random.shuffle') as mock_shuffle:
            deck.shuffle()                                                 # Appelle shuffle()
            mock_shuffle.assert_called_once_with(deck.cards)               # Vérifie l'appel unique à random.shuffle()

    def test_repr(self):
        deck = Deck()                                                      # Initialise un nouveau deck
        rep = repr(deck)                                                   # Appelle __repr__()
        self.assertTrue(rep.startswith("<Deck: "),                         # Vérifie le préfixe de la chaîne
                        "__repr__ doit commencer par '<Deck: '")
        self.assertIn(" cards remaining>", rep,                            # Vérifie le suffixe de la chaîne
                      "__repr__ doit contenir ' cards remaining>'")
        # Le nombre indiqué dans la repr doit correspondre à la taille du deck
        count = int(rep.split()[1])                                        # Extrait le nombre de cartes indiqué
        self.assertEqual(count, len(deck),                                # Vérifie la cohérence avec len(deck)
                         "Le nombre dans __repr__ doit correspondre à len(deck)")

if __name__ == "__main__":
    unittest.main()  # Lance tous les tests définis dans cette classe

# -------------------------------------------------------
# Résumé :
# -------------------------------------------------------
# Ce fichier de tests unitaires vérifie le comportement de la classe Deck :
#   1. Initialisation avec 54 cartes (6 couleurs × 9 valeurs).
#   2. Les méthodes __len__ et is_empty() reflètent correctement l'état du deck.
#   3. draw(n) renvoie le bon nombre de cartes et réduit la taille interne,
#      même si n dépasse le nombre de cartes restantes.
#   4. draw_card() renvoie une seule carte ou None si le deck est vide.
#   5. shuffle() fait appel à random.shuffle sur la liste interne.
#   6. __repr__ produit une chaîne indiquant le nombre de cartes restantes.
# Les assertions utilisent des messages explicites pour faciliter le débogage.
