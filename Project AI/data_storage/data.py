import json
import csv
from typing import Dict, Any, List

# 1) Q-table (JSON)

def save_q_table(q_table: Dict[Any, Dict[Any, float]], filename: str) -> None:
    """
    Sérialise la Q-table (état→action→valeur) dans un JSON.
    Comme les clés d'état/action peuvent être non-string,
    on convertit tout en str pour la sérialisation.
    """
    ser = {str(s): {str(a): v for a, v in actions.items()}
           for s, actions in q_table.items()}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ser, f, ensure_ascii=False, indent=2)


def load_q_table(filename: str) -> Dict[Any, Dict[Any, float]]:
    """
    Recharge la Q-table depuis le JSON.
    Les clés restent des strings ; si on a besoin de types originaux,
    il faudra les caster lors de la lecture.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        ser = json.load(f)
    return ser


# 2) Journal de parties (CSV) et journal global en mémoire

game_moves: List[Dict[str, Any]] = []


def log_move(player: str, state: Any, action: Any, reward: float) -> None:
    """
    Ajoute un enregistrement de coup pour le joueur spécifié.
    - player: nom du joueur ('Player 1', 'AI', ...)
    - state: état avant le coup
    - action: tuple (champ, indice de carte)
    - reward: valeur de la récompense reçue
    """
    # Sérialise state en JSON si ce n'est pas déjà une string
    ser_state = state if isinstance(state, str) else json.dumps(state)
    game_moves.append({
        'player': player,
        'state': ser_state,
        'action': str(action),
        'reward': reward
    })


def clear_game_log() -> None:
    """Vide le journal de jeu en mémoire."""
    game_moves.clear()


def save_game_log(filename: str, rows: List[Dict[str, Any]] = None) -> None:
    """
    Sauvegarde le journal de partie dans un CSV.
    Si rows n'est pas fourni, on utilise game_moves global.
    Colonnes: player, state, action, reward
    """
    data = rows if rows is not None else game_moves
    if not data:
        return
    headers = list(data[0].keys())
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            # Convertit toutes les valeurs en str pour CSV
            writer.writerow({h: str(row[h]) for h in headers})


def load_game_log(filename: str) -> List[Dict[str, str]]:
    """
    Lit le CSV de parties et renvoie une liste de dict(str→str).
    Il te revient de caster là où c'est nécessaire.
    """
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

# Exemple d’utilisation rapide (en mode script)
if __name__ == '__main__':
    # Remise à zéro du journal
    clear_game_log()
    # Simule quelques coups
    log_move('Player 1', [0,1,2], (3,1), 0.1)
    log_move('AI', [0,1,3], (4,2), -0.1)
    # Sauvegarde
    save_game_log('game_moves.csv')
    print('Journal sauvegardé dans game_moves.csv')

# Résumé rapide :
# But du code : gérer la persistance de la Q-table et des logs de parties.
# 
# - save_q_table / load_q_table : sérialisent et rechargent la Q-table en JSON.
# - log_move / clear_game_log / save_game_log / load_game_log :
#   permettent d’enregistrer chaque coup en mémoire, de le vider,
#   et de sauvegarder ou recharger un journal de parties au format CSV.
