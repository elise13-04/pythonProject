from dataclasses import dataclass  # Fournit le décorateur @dataclass

@dataclass
class Card:
    value: int  # Valeur numérique (1-9)
    color: str  # Couleur (par ex. "rouge", "bleu", ...)

    def __repr__(self):
        # Représente la carte sous la forme "R5" : R = première lettre de la couleur, 5 = valeur
        return f"{self.color[0].upper()}{self.value}"

# -------------------------------------------------------------------
# Résumé rapide – Card
# -------------------------------------------------------------------
# Classe immuable décrivant une carte individuelle (valeur + couleur).
# __repr__ fournit une notation compacte pour l'affichage ou le débogage.
# -------------------------------------------------------------------
