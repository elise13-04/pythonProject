import unittest  # Importe le module unittest pour définir des tests unitaires
from Reinforcement_Learning_Agent_and_AIPlayer.RLAgent import RLAgent  # Importe la classe RLAgent à tester

class TestRLAgentUpdate(unittest.TestCase):  # Définit une suite de tests pour la mise à jour de la Q‑table
    """Vérifie la mise à jour de la Q‑table via update_q_value()."""

    def test_q_value_increases_on_positive_reward(self):
        """Assure que Q(s,a) augmente correctement pour une récompense positive."""
        # Crée un agent avec alpha=1.0 pour que la mise à jour soit directe et gamma=0 pour ignorer la valeur future
        agent = RLAgent(learning_rate=1.0, discount_factor=0.0)
        # Déclare état courant, action, récompense positive et nouvel état
        s, a, r, sp = "state0", ("field", 0), 1.0, "state1"
        # Initialise la valeur Q(s,a) à 0
        agent.q_table[s][a] = 0.0
        # Applique la mise à jour Q-learning
        agent.update_q_value(s, a, r, sp, [])
        # Avec alpha=1 et gamma=0, Q(s,a) doit devenir reward = 1.0
        self.assertEqual(agent.q_table[s][a], 1.0)

    def test_q_value_decays_on_negative_reward(self):
        """Vérifie que Q(s,a) décroit correctement pour une récompense négative."""
        # Crée un agent avec alpha=0.5 et gamma=0
        agent = RLAgent(learning_rate=0.5, discount_factor=0.0)
        # Déclare état, action, récompense négative et nouvel état
        s, a, r, sp = "state0", ("field", 0), -1.0, "state1"
        # Initialise Q(s,a) à 0
        agent.q_table[s][a] = 0.0
        # Mise à jour Q(s,a)
        agent.update_q_value(s, a, r, sp, [])
        # Avec alpha=0.5, Q(s,a) = 0 + 0.5 * (-1 - 0) = -0.5
        self.assertEqual(agent.q_table[s][a], -0.5)

# --- Résumé ---
# Ce fichier teste deux scénarios pour RLAgent.update_q_value :
# 1. test_q_value_increases_on_positive_reward : assure l'augmentation de Q(s,a) pour r>0.
# 2. test_q_value_decays_on_negative_reward : vérifie la décroissance de Q(s,a) pour r<0.
# Chaque test configure alpha et gamma pour isoler l'effet de la récompense immédiate.
