import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve
plt.close('all')

df_1=pd.read_csv('maillage_bien_vinlet200p_ke_realizable_2eme_ordre.csv')

recollement_exp=df_1['Cf_wall_inf_exp: X'][9]+(df_1['Cf_wall_inf_exp: X'][10]-df_1['Cf_wall_inf_exp: X'][9])*(-df_1['Cf_wall_inf_exp: Cf'][9])/(df_1['Cf_wall_inf_exp: Cf'][10]-df_1['Cf_wall_inf_exp: Cf'][9])

plt.figure(figsize=(10,6))

#plt.scatter(df_1['Cf_wall_inf_exp: X'],df_1['Cf_wall_inf_exp: Cf'],label='Exp (Driver,1985)', facecolors='none', edgecolors='red', s=25)
#plt.scatter([recollement_exp],[0],marker='+',color='red', s=25, label='Point de recollement experimentale')
#plt.scatter(df_1['Region1 2D Boundaries: Direction [1,0,0] (m)'],df_1['Region1 2D Boundaries: cf_num'],label='Simulation numérique', facecolors='none', edgecolors='blue', s=25, alpha=0.5)

plt.plot(
    df_1['Cf_wall_inf_exp: X'],
    df_1['Cf_wall_inf_exp: Cf'],
    color='red',
    linewidth=1.5,
    label='Exp (Driver,1985)'
)

plt.plot(
    [recollement_exp],
    [0],
    color='red',
    linewidth=0,  # pas de ligne continue
    marker='+',
    markersize=8,
    label='Point de recollement experimental'
)

plt.plot(
    df_1['Region1 2D Boundaries: Direction [1,0,0] (m)'],
    df_1['Region1 2D Boundaries: cf_num'],
    color='blue',
    linewidth=1.5,
    alpha=0.5,
    label='Simulation numérique'
)


plt.xlabel('position x [m]')
plt.ylabel('$C_f$ [-]')
plt.grid()
plt.legend()
plt.savefig('plot_exemple_python.png',dpi=600)

plt.show()