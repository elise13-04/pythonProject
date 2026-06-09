import sys  # Permet d'accéder aux arguments CLI et de quitter l'application
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QGridLayout, QHBoxLayout, QVBoxLayout,
    QLabel, QListWidget, QPushButton, QComboBox, QMessageBox
)  # Import des widgets nécessaires pour l'UI
from Game.Game import Game  # Logique du jeu (plateau, règles, etc.)
from Players.HumanPlayer import HumanPlayer  # Implémentation d'un joueur humain
from Reinforcement_Learning_Agent_and_AIPlayer.AIPlayer import AIPlayer  # Agent IA
from data_storage.data import log_move, save_game_log, clear_game_log  # Journalisation des coups

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()  # Initialise la fenêtre principale
        self.setWindowTitle("Schotten-Totten GUI")  # Titre de la fenêtre

        # --- Panneau principal et logique de jeu ---
        self.game = None  # Instance Game initialement vide
        central = QWidget()  # Widget central conteneur
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)  # Layout horizontal global

        # --- Grille des champs de bataille (3×3) ---
        self.battle_labels = []  # Liste pour stocker les QLabel de chaque champ
        battlefield_widget = QWidget()
        bf_layout = QGridLayout(battlefield_widget)
        for row in range(3):
            row_labels = []
            for col in range(3):
                idx = row * 3 + col  # Calcule l'indice linéaire des 9 champs
                lbl = QLabel(f"Field {idx}:\nP1:\nP2:")  # Étiquette vide initiale
                lbl.setFixedSize(120, 100)  # Dimensions fixes
                lbl.setStyleSheet("border: 1px solid black;")  # Bordure visible
                bf_layout.addWidget(lbl, row, col)
                row_labels.append(lbl)
            self.battle_labels.append(row_labels)
        main_layout.addWidget(battlefield_widget)

        # --- Panneau de contrôle ---
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)

        # Sélection de l'adversaire
        ctrl_layout.addWidget(QLabel("Opponent:"))
        self.opponent_combo = QComboBox()
        self.opponent_combo.addItems(["AI", "Human"])  # Choix IA ou humain
        ctrl_layout.addWidget(self.opponent_combo)
        self.new_btn = QPushButton("New Game")
        self.new_btn.clicked.connect(self.new_game)  # Lance une nouvelle partie
        ctrl_layout.addWidget(self.new_btn)

        # Affichage de la main du joueur
        ctrl_layout.addWidget(QLabel("Your hand:"))
        self.hand_list = QListWidget()
        ctrl_layout.addWidget(self.hand_list)

        # Sélecteur de champ valide
        ctrl_layout.addWidget(QLabel("Choose battlefield:"))
        self.field_combo = QComboBox()
        ctrl_layout.addWidget(self.field_combo)

        # Bouton pour jouer une carte
        self.play_btn = QPushButton("Play Card")
        self.play_btn.clicked.connect(self.play_card)
        ctrl_layout.addWidget(self.play_btn)

        main_layout.addWidget(controls)

        # Démarre une partie dès l'ouverture
        self.new_game()

    def new_game(self):
        clear_game_log()  # Vide l'historique précédent
        mode = self.opponent_combo.currentText()  # Récupère le type d'adversaire
        self.game = Game()  # Crée une nouvelle partie
        if mode == "Human":
            self.game.players[1] = HumanPlayer("Player 2")  # Remplace IA par humain
        self.game.reset()  # Initialise plateau et distribue 6 cartes
        self.draw_card_for_current()  # Pioche pour atteindre 7 cartes
        self.play_btn.setEnabled(True)  # Réactive le bouton de jeu
        self.update_ui()  # Rafraîchit l'affichage

    def draw_card_for_current(self):
        card = self.game.deck.draw_card()  # Tire une carte du talon
        if card:
            self.game.players[self.game.current].hand.append(card)  # L'ajoute à la main
        else:
            QMessageBox.information(self, "Deck Empty", "No more cards to draw.")

    def update_ui(self):
        # Met à jour tous les labels des champs
        for idx, lbl in enumerate(sum(self.battle_labels, [])):
            bf = self.game.board.battlefields[idx]
            p1 = ' '.join(str(c) for c in bf.player1_cards)
            p2 = ' '.join(str(c) for c in bf.player2_cards)
            lbl.setText(f"Field {idx}:\nP1: {p1}\nP2: {p2}")
        # Liste de la main du joueur courant
        self.hand_list.clear()
        current_player = self.game.players[self.game.current]
        for card in current_player.hand:
            self.hand_list.addItem(str(card))
        # Remplit la combo des champs encore jouables
        self.field_combo.clear()
        for idx, bf in enumerate(self.game.board.battlefields):
            count = len(bf.player1_cards) if self.game.current == 0 else len(bf.player2_cards)
            if count < 3 and not self.game.board.resolved[idx]:
                self.field_combo.addItem(f"Field {idx}")
        # Active le bouton si possible
        has_cards = len(current_player.hand) > 0
        has_fields = self.field_combo.count() > 0
        self.play_btn.setEnabled(has_cards and has_fields)

    def play_card(self):
        prev_state = list(self.game.get_state())  # Sauvegarde état avant coup
        card_idx = self.hand_list.currentRow()
        if card_idx < 0:
            QMessageBox.warning(self, "No Card Selected", "Please select a card to play.")
            return
        field_idx = int(self.field_combo.currentText().split()[-1])  # Extrait l'indice
        player_index = self.game.current
        action = (field_idx, card_idx)
        try:
            next_state, reward, done, _ = self.game.step(action)  # Joue le coup
        except Exception as e:
            QMessageBox.warning(self, "Invalid Move", str(e))
            return
        log_move(self.game.players[player_index].name, prev_state, action, reward)  # Journalise le coup
        self.draw_card_for_current()  # Pioche pour le joueur suivant
        self.update_ui()  # Rafraîchit l'UI

        # Si l'adversaire est une IA, on enchaîne son tour
        if not done and self.opponent_combo.currentText() == "AI" and self.game.current == 1:
            prev_state_ai = list(self.game.get_state())
            ai_action = self.game.players[1].make_move(self.game.board)
            _, ai_reward, done, _ = self.game.step(ai_action)
            log_move(self.game.players[1].name, prev_state_ai, ai_action, ai_reward)
            self.draw_card_for_current()
            self.update_ui()

        if done:
            winner = "You" if self.game.current == 1 else "AI"
            QMessageBox.information(self, "Game Over", f"{winner} won the game!")
            save_game_log('game_moves.csv')  # Sauvegarde finale du journal
            reply = QMessageBox.question(
                self, "Play Again?", "Do you want to start a new game?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.new_game()
            else:
                self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)  # Création de l'application Qt
    window = MainWindow()         # Instanciation de la fenêtre
    window.show()                 # Affichage
    sys.exit(app.exec_())         # Boucle d'événements Qt

# Résumé rapide :
# Cette classe MainWindow propose une interface graphique PyQt5 pour jouer à Schotten-Totten.
# - Initialisation : création/choix du mode (IA ou humain), reset du jeu et pioche initiale.
# - UI : grille 3×3 pour les champs, liste pour la main, combo pour choisir le champ, bouton Play.
# - log_move : chaque coup est journalisé (état avant, action, récompense) puis sauvegardé en fin de partie.
# - play_card : gère successivement le coup du joueur, celui de l'IA (si sélectionnée) et la fin de partie.
