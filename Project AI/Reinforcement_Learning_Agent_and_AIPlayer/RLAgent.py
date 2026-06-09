import random  # Pour gérer l'exploration aléatoire (ε‑greedy)
from collections import defaultdict  # Dictionnaire avec valeurs par défaut (utile pour la Q‑table)


class RLAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.2):
        # Q‑table : { état → { action → valeur‑Q } }
        # defaultdict permet de créer automatiquement une valeur 0.0 pour les paires jamais vues
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate      # α – Taux d'apprentissage
        self.discount_factor = discount_factor  # γ – Facteur de réduction des récompenses futures
        self.epsilon = epsilon                  # ε – Taux d'exploration (ε‑greedy)

    def choose_action(self, state, possible_actions):
        """Choix d'une action selon la stratégie ε‑greedy."""
        # Exploration : on choisit une action aléatoire avec probabilité ε
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        # Exploitation : on prend l'action avec la plus grande valeur‑Q pour cet état
        q_values = self.q_table[state]
        best_action = max(possible_actions, key=lambda action: q_values[action])
        return best_action

    def update_q_value(self, state, action, reward, next_state, next_possible_actions):
        """Met à jour la valeur‑Q de (state, action) selon la formule du Q‑learning."""
        # Valeur maximale estimée pour le prochain état (0 si aucun coup possible)
        next_max = 0.0
        if next_possible_actions:
            next_max = max(self.q_table[next_state][a] for a in next_possible_actions)
        # Ancienne valeur Q(s,a)
        old_value = self.q_table[state][action]
        # Nouvelle valeur après prise en compte de la récompense et du futur estimé
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        # Stocke la mise à jour
        self.q_table[state][action] = new_value

    def train_episode(self, game):
        """Effectue une partie complète pour entraîner l'agent (un épisode)."""
        game.reset()  # Réinitialise le jeu
        state = game.get_current_state()  # État initial
        while not game.is_over():  # Boucle jusqu'à la fin de la partie
            # Liste des actions légales dans l'état courant
            possible_actions = game.get_possible_actions(state)
            # Sélectionne une action via ε‑greedy
            action = self.choose_action(state, possible_actions)
            # Exécute l'action dans l'environnement → renvoie récompense + nouvel état
            reward, next_state = game.perform_action(action)
            # Actions possibles après le coup (utiles pour la mise à jour de Q)
            next_possible_actions = game.get_possible_actions(next_state)
            # Mise à jour de la Q‑table
            self.update_q_value(state, action, reward, next_state, next_possible_actions)
            # Passage à l'état suivant
            state = next_state


# ------------------------------------------------------
# Résumé rapide
# ------------------------------------------------------
# RLAgent comporte :
#   • Une Q‑table stockée dans un defaultdict imbriqué pour associer (état, action) à une valeur.
#   • choose_action() : stratégie ε‑greedy (exploration vs exploitation).
#   • update_q_value() : mise à jour Q‑learning classique.
#   • train_episode() : joue une partie complète avec l'environnement Game pour entraîner l'agent.
# Les hyper‑paramètres (learning_rate, discount_factor, epsilon) peuvent être ajustés au besoin.
