"""
Méthodes d'intégration numérique :
    - Gauss-Legendre
    - Gauss-Laguerre
    - Gauss-Chebyshev
    - Simpson
    - Spline cubique
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.laguerre import laggauss
from scipy.interpolate import CubicSpline


# ================================================================
#   Méthode de Gauss–Legendre
# ================================================================
def gauss_legendre(f, a, b, n):
    """
    Intégration de f(x) sur [a, b] en utilisant la quadrature 
    de Gauss–Legendre à n points.

    Paramètres :
        f : fonction à intégrer
        a, b : bornes d’intégration (fini → obligatoire)
        n : ordre de la quadrature

    Principe :
        1) leggauss(n) fournit :
            - x : n racines du polynôme de Legendre (dans [-1, 1])
            - w : poids associés
        2) On transforme le domaine [-1, 1] vers [a, b]
        3) On applique la formule :
                ∫ f(x) dx ≈ Σ w_i * f(t_i) * (b-a)/2
    """

    # Racines et poids dans [-1, 1]
    x, w = leggauss(n)

    # Transformation affine t : [-1,1] → [a,b]
    t = 0.5 * (x + 1) * (b - a) + a

    # Formule finale
    return 0.5 * (b - a) * np.sum(w * f(t))


# ================================================================
#   Méthode de Gauss–Laguerre
# ================================================================
def gauss_laguerre(f, n):
    """
    Quadrature de Gauss–Laguerre à n points.
    Adaptée aux intégrales de la forme :
            ∫₀⁺∞ f(x) e^{-x} dx

    Remarque :
        - laggauss(n) donne les racines du polynôme de Laguerre
          + les poids correspondants.
        - Étant déjà définie pour exp(-x), la méthode nécessite
          que f(x) contienne l’éventuelle compensation exp(+x)
          si l’intégrale n’est pas au bon format.
    """

    x, w = laggauss(n)
    return np.sum(w * f(x))


# ================================================================
#   Méthode de Gauss–Chebyshev
# ================================================================
def gauss_chebyshev(f, n):
    """
    Quadrature de Gauss–Chebyshev pour le 1er type.
    Adaptée aux intégrales de la forme :
            ∫_{-1}^{1} f(x) / sqrt(1 - x^2) dx

    Paramètres :
        f : fonction à intégrer
        n : ordre de la quadrature

    Principe :
        - Les n points sont x_k = cos((2k-1)π / (2n))
        - Les poids sont tous égaux à w = π / n
    """

    k = np.arange(1, n+1)
    x = np.cos((2*k - 1) * np.pi / (2 * n))  # n points de Chebyshev
    w = np.pi / n                            # poids constants

    return w * np.sum(f(x))


# ================================================================
#   Méthode de Simpson
# ================================================================
def simpson(f, a, b, n):
    """
    Intégration par la méthode de Simpson composite.

    Paramètres :
        f : fonction
        a, b : bornes de l'intégrale
        n : nombre de sous-intervalles (doit être pair)

    Remarques :
        - Si n est impair, on incrémente de 1 pour rendre n pair.
        - Formule :
              ∫ ≈ h/3 [ f(x0) + f(xn) 
                          + 4 Σ f(xi impairs)
                          + 2 Σ f(xi pairs internes) ]
    """

    # Simpson exige un nombre pair de sous-intervalles
    if n % 2 == 1:
        n += 1

    h = (b - a) / n

    # Points d'échantillonnage
    x = np.linspace(a, b, n+1)
    y = f(x)

    return (h/3) * (y[0] + y[-1]
                    + 4 * np.sum(y[1:-1:2])   # indices impairs
                    + 2 * np.sum(y[2:-2:2])) # indices pairs internes


# ================================================================
#   Méthode Spline (interpolation cubique + intégration)
# ================================================================
def spline_integral(f, a, b, n):
    """
    Approche spline cubique :
        1) Évalue f en n points
        2) Construit une spline cubique interpolante
        3) Intègre la spline sur [a, b]

    Paramètres :
        f : fonction
        a, b : bornes de l'intégrale
        n : nombre de points d'interpolation

    Attention :
        - Si f(x) retourne NaN ou inf pour certains points → on renvoie NaN.
        - Avantage : très bonne précision pour fonctions lisses.
        - Inconvénient : coûteux en calcul.
    """

    # Points d'interpolation
    x = np.linspace(a, b, max(2, n))
    y = f(x)

    # Sécurité : éviter les valeurs infinies ou indéfinies
    if not np.all(np.isfinite(y)):
        return np.nan

    # Spline cubique interpolante
    cs = CubicSpline(x, y)

    # Intégrale exacte de la spline
    return cs.integrate(a, b)
