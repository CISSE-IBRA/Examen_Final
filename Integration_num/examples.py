import numpy as np


# 4 exemples demandés
# 1) famille Chebyshev : 1 / sqrt(1 - x^2) sur [-0.9, 0.9]
f_cheb = lambda x: 1.0 / np.sqrt(1 - x**2)
cheb_domain = (-0.9, 0.9)


# 2) famille Laguerre : exp(-x) sur [0, +inf) -> on tronque numériquement
f_lag = lambda x: np.exp(-x)
lag_domain = (0.0, 20.0) # tronqué


# 3) combinaison des deux
f_mix = lambda x: np.exp(-x) / np.sqrt(np.maximum(1e-12, 1 - x**2))
mix_domain = (-0.8, 2.0)


# 4) quelconque (régulière)
f_gen = lambda x: np.sin(5*x) + x**2
gen_domain = (0.0, 3.0)


EXAMPLES = [
    ("Chebyshev", f_cheb, cheb_domain),
    ("Laguerre", f_lag, lag_domain),
    ("Mixte", f_mix, mix_domain),
    ("Général", f_gen, gen_domain),
]