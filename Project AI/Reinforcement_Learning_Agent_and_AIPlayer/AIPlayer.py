import torch  # Importer PyTorch pour les opérations sur tenseurs et le support des réseaux de neurones
import random  # Importer random pour les choix aléatoires
from Players.Player import Player  # Importer la classe de base Player depuis le module Players

class AIPlayer(Player):
    """IA combinée : heuristique basique si policy_net est absent, sinon RL glouton."""

    def __init__(self, name, policy_net=None):
        super().__init__(name)  # Initialiser le Player de base avec le nom donné
        self.policy_net = policy_net  # Stocker un réseau de politique optionnel pour les coups RL

    def make_move(self, board):
        # Si aucun réseau de politique n'est fourni, utiliser une heuristique simple
        if self.policy_net is None:
            # Sélectionner les indices de champs valides (côté joueur2) non pleins et non résolus
            valid_fields = [i for i, bf in enumerate(board.battlefields)
                            if len(bf.player2_cards) < 3 and not board.resolved[i]]
            # Si aucun champ n'est valide, prendre tous les champs disponibles
            if not valid_fields:
                valid_fields = list(range(len(board.battlefields)))

            # Trier les indices des cartes en main par valeur décroissante
            sorted_indices = sorted(
                range(len(self.hand)),
                key=lambda i: self.hand[i].value,
                reverse=True
            )
            # Jouer les cartes de plus haute valeur en priorité, garder la plus faible pour plus tard
            for card_idx in sorted_indices[:-1]:
                return valid_fields[0], card_idx
            # En dernier recours, choisir un champ et une carte au hasard
            return random.choice(valid_fields), random.choice(range(len(self.hand)))

        # Sinon, utiliser la politique RL pour sélectionner le coup (stratégie gloutonne)
        # Construire le vecteur d'état à partir du plateau et de la main
        state = board.to_feature_vector(current_player=2, hand=self.hand)
        with torch.no_grad():  # Désactiver le calcul des gradients pour l'inférence
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(tensor)  # Obtenir les Q-values pour toutes les actions
        action = q_values.argmax().item()  # Sélectionner l'action ayant la Q-value la plus élevée
        # Décomposer l'action en index de champ et index de carte
        field, card_idx = divmod(action, len(self.hand))
        return field, card_idx  # Retourner le champ et la carte choisis

# Résumé rapide :
# But du code : fournir un joueur IA capable de jouer avec une heuristique basique ou une politique RL.
### Heuristique basique: désigne une règle simple et codée à la main, sans apprentissage ni calcul complexe, pour décider rapidement d’un coup (if elif else..)
### Politique RL (apprentissage par renforcement): désigne la stratégie qu’un agent (ici l'IA) utilise pour choisir une action dans un état donné, basée sur ce qu’il a « appris ». Au lieu d’utiliser des règles codées en dur, l’IA s’appuie sur un réseau entraîné pour estimer la « qualité » de chaque coup et choisit celui qui paraît le plus prometteur.
# - Le constructeur (__init__) initialise le joueur et stocke un éventuel réseau de politique pour l'apprentissage par renforcement.
# - make_move :
#   • Sans policy_net : utilise une heuristique simple pour choisir le champ et la carte (tri par valeur décroissante).
#   • Avec policy_net : convertit l'état du jeu en vecteur de caractéristiques, prédit les Q-values via le réseau, et joue l'action ayant la Q-value maximale.
