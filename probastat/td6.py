import scipy as sp
import numpy as np
import scipy.stats as st
import numpy.random as rnd
B = rnd.exponential(scale = 2, size = (42000,))
print(B)
Bm = B- np.mean(B)
print(Bm)
Bi = abs(Bm)>3
print(Bi)
p = sum(Bi)/42000
print(p)