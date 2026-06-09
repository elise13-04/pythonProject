import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math

def arrondir_pair_superieur(x):
        return int(math.ceil(x / 2.0)) * 2
# === Fonction de transformation ===
def transform_matrices(theta_deg,C):
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    
    T_inv = np.array([
          [c**2, s**2, -np.sqrt(2)*s*c],
          [s**2, c**2,  np.sqrt(2)*s*c],
          [np.sqrt(2)*s*c, -np.sqrt(2)*s*c, c**2 - s**2]
    ])
    
    # Transformation
    C_xy = T_inv.T @ C @ T_inv
    S_xy = np.linalg.inv(C_xy)
    return C_xy, S_xy,T_inv,T_inv.T


def hashin_scaling_factor(stress_vec, Xt, Xc, Yt, Yc, Sc):
#    Calcule le facteur k tel que le critère de Hashin atteigne 1
#    et indique le mode de rupture :
#              1  -> traction fibre
#     -1  -> compression fibre
#      2  -> traction transverse
#     -2  -> compression transverse
    S11, S22, S12 = stress_vec
    S12=S12/np.sqrt(2)

    k_values = []
    modes = []

    # --- Traction fibre ---
    if S11 > 0:
        A = (S11/Xt)**2 + (S12/Sc)**2
        if A > 0:
            k_values.append(1 / np.sqrt(A))
            modes.append(1)

    # --- Compression fibre ---
    if S11 < 0:
        A = (S11/Xc)**2
        if A > 0:
            k_values.append(1 / np.sqrt(A))
            modes.append(-1)

    # --- Traction transverse ---
    if S22 > 0:
        A = (S22/Yt)**2 + (S12/Sc)**2
        if A > 0:
            k_values.append(1 / np.sqrt(A))
            modes.append(2)

    # --- Compression transverse ---
    if S22 < 0:
        A = (S22/Yc)**2 + (S12/Sc)**2
        if A > 0:
            k_values.append(1 / np.sqrt(A))
            modes.append(-2)

    if len(k_values) == 0:
           return np.inf, None  # aucune contrainte significative

    # --- Rupture critique ---
    k_values = np.array(k_values)
    idx_min = np.argmin(k_values)
    k_min = k_values[idx_min]
    mode = modes[idx_min]

    return k_min, mode


def hashin_check(stress_vec, Xt, Xc, Yt, Yc, Sc):
#    Vérifie si le critère de Hashin est dépassé.
#    Retourne :
#              rupture (bool), k (facteur limite), mode (type de rupture)
    k, mode = hashin_scaling_factor(stress_vec, Xt, Xc, Yt, Yc, Sc)
    rupture = k <= 1
    return rupture, k, mode


# === Donnees du matériau ===
E1 = 53000. #MPa 
E2 = 18000. # MPa
nu12 = 0.3
G12 = 5900. #MPa
Xt=1140.
Xc=570
Yt=40
Yc=135
Sc=61
# === Angles des plis (dans l’ordre souhaité) et pourcentage associé
angles = [0, 90, 45, -45]

pc=[0.25,0.25,0.25,0.25]
# === Matrice de souplesse locale (1,2) ===
S12 = np.array([
     [1/E1, -nu12/E1, 0],
     [-nu12/E1, 1/E2, 0],
     [0, 0, 1/(2*G12)]
])

# === Matrice de rigidité locale (1,2) ===
C12 = np.linalg.inv(S12)

# === Calcul des matrices T C et S pour chacune des orientations
C_list = []
S_list = []
T_list=[]

for angle in angles:
    C_xy, S_xy,T,T_inv = transform_matrices(angle,C12)
    C_list.append(C_xy)
    S_list.append(S_xy)
    T_list.append(T)

# === Calcul de l'elasticite du stratifie
Cstrat=np.zeros((3,3))
for i, angle in enumerate(angles):
    Cstrat=Cstrat+pc[i]*C_list[i]
# === Matrice moyenne de souplesse ===
Sstrat = np.linalg.inv(Cstrat)
# === Extraction des modules équivalents ===
Ex = 1 / Sstrat[0, 0]
Ey = 1 / Sstrat[1, 1]
nuxy = -Sstrat[0, 1] / Sstrat[0, 0]
Gxy = 1 / (2 * Sstrat[2, 2])
print("\n=== Propriétés équivalentes du stratifié ===")
print(f"Ex   = {Ex:.3f} MPa")
print(f"Ey   = {Ey:.3f} MPa")
print(f"nuxy = {nuxy:.4f}")
print(f"Gxy  = {Gxy:.3f} MPa")


# Determination du facteur de charge pour une contrainte appliquee respectant la direction de chargement
Sig_strat=np.array([1./2.,1.,0])
E_strat=Sstrat@Sig_strat
crit_hashin={}
for i, angle in enumerate(angles):
    print(f"\n=== Pli {i} : θ = {angle}° ===")
    epsilon_12_i=T_list[i].T@E_strat
    sigma_12_i=C12@epsilon_12_i
    rupture, k, mode = hashin_check(sigma_12_i, Xt, Xc, Yt, Yc, Sc)
    print(f"Facteur limite du critere de Hashin : k = {k:.3f}")
    print(f"Mode de rupture : {mode}")
    crit_hashin[str(angle)]=[k,mode]

plt.show()
