"""
Exécute toutes les méthodes d’intégration numérique sur chaque exemple,
et collecte : 
    - les erreurs en fonction de n
    - les temps de calcul.
"""

import time
import numpy as np
from .methods import gauss_legendre, gauss_laguerre, gauss_chebyshev, simpson, spline_integral
from .examples import EXAMPLES

def true_int_trapz(f, a, b, N=200000):
    """
    Calcule une valeur de référence de l’intégrale en utilisant la méthode
    du trapèze avec un très grand nombre de points (200 000 par défaut).

    Sert de « valeur vraie » Itrue pour comparer les méthodes d’intégration.

    ⚠ Pour les intégrales sur un domaine infini, on tronque simplement à b.
    """
    x = np.linspace(a, b, N)
    y = f(x)
    return np.trapz(y, x)


def run_all():
    """
    Exécute toutes les méthodes (Gauss-Legendre, Gauss-Chebyshev,
    Gauss-Laguerre, Simpson, Spline) sur tous les exemples définis
    dans EXAMPLES.

    Renvoie un dictionnaire contenant :
        - Ns : les tailles n testées
        - errs : les erreurs par méthode
        - times : les temps de calcul par méthode
        - Itrue : valeur de référence
        - domain : domaine d’intégration
    """

    # Valeurs de n pour tester la convergence (de 4 à 52 par pas de 4)
    Ns = list(range(4, 56, 4))

    # Contiendra tous les résultats organisés par exemple
    results = {}
    
    # Boucle sur les 4 exemples définis dans examples.py
    for name, f, domain in EXAMPLES:

        a, b = domain

        # 1) Calcul de la valeur de référence (approximation trapèze)
        Itrue = true_int_trapz(f, a, b)

        # Dictionnaires pour stocker les erreurs et temps par méthode
        errs = {"leg": [], "cheb": [], "lag": [], "simp": [], "spl": []}
        times = {k: [] for k in errs}
        
        # Pour chaque valeur de n (nombre de points / ordre)
        for n in Ns:

            # ----------------------------------------------------------
            #  Méthode de Gauss-Legendre : seulement pour domaines finis
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                if np.isfinite(a) and np.isfinite(b):
                    val_leg = gauss_legendre(f, a, b, n)
                else:
                    val_leg = np.nan
            except Exception:
                val_leg = np.nan

            times["leg"].append(time.time() - t0)
            errs["leg"].append(abs(val_leg - Itrue) if np.isfinite(val_leg) else np.nan)
            

            # ----------------------------------------------------------
            #  Méthode de Gauss-Chebyshev : adaptée à [-1, 1]
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                if a == -1 and b == 1:
                    # cas idéal : domaine natif
                    val_cheb = gauss_chebyshev(f, n)
                else:
                    # sinon : transformation du domaine vers [-1,1]
                    nodes = np.cos((2*np.arange(1, n+1) - 1) * np.pi / (2 * n))
                    # transformation affine vers [a, b]
                    x = 0.5 * (nodes + 1) * (b - a) + a
                    # poids approximatif
                    w = (b - a) * np.pi / (2 * n)
                    val_cheb = w * np.sum(f(x))
            except Exception:
                val_cheb = np.nan

            times["cheb"].append(time.time() - t0)
            errs["cheb"].append(abs(val_cheb - Itrue) if np.isfinite(val_cheb) else np.nan)


            # ----------------------------------------------------------
            #  Méthode de Gauss-Laguerre : pour [0, +∞)
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                # condition : domaine semi-infini raisonnable
                val_lag = gauss_laguerre(f, n) if a >= 0 and b > 10 else np.nan
            except Exception:
                val_lag = np.nan

            times["lag"].append(time.time() - t0)
            errs["lag"].append(abs(val_lag - Itrue) if np.isfinite(val_lag) else np.nan)


            # ----------------------------------------------------------
            #  Méthode de Simpson
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                val_simp = simpson(f, a, b, n)
            except Exception:
                val_simp = np.nan

            times["simp"].append(time.time() - t0)
            errs["simp"].append(abs(val_simp - Itrue) if np.isfinite(val_simp) else np.nan)


            # ----------------------------------------------------------
            #  Méthode Spline (interpolation + intégrale)
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                # on impose au moins 10 points car spline fragile pour petits n
                val_spl = spline_integral(f, a, b, max(10, n))
            except Exception:
                val_spl = np.nan

            times["spl"].append(time.time() - t0)
            errs["spl"].append(abs(val_spl - Itrue) if np.isfinite(val_spl) else np.nan)


        # On stocke tous les résultats de cet exemple
        results[name] = {
            "Ns": Ns,
            "errs": errs,
            "times": times,
            "Itrue": Itrue,
            "domain": domain
        }

    return results
