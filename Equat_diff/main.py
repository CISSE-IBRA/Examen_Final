"""
Programme principal pour exécuter les méthodes de résolution numérique
d'équations différentielles du cours
"""

import numpy as np
import matplotlib.pyplot as plt
from euler import methode_euler, exemple_euler
from heun import methode_heun, exemple_heun
from rk4 import methode_rk4, exemple_rk4

def menu_principal():
    """Menu principal interactif"""
    print("=" * 60)
    print("RÉSOLUTION NUMÉRIQUE D'ÉQUATIONS DIFFÉRENTIELLES")
    print("=" * 60)
    print("\nMéthodes disponibles:")
    print("1. Méthode d'Euler (ordre 1)")
    print("2. Méthode de Heun (ordre 2)")
    print("3. Méthode de Runge-Kutta d'ordre 4 (RK4)")
    print("4. Comparaison des 3 méthodes")
    print("5. Quitter")
    
    return input("\nChoisissez une option (1-5): ")

def exemple_personnalise():
    """Permet à l'utilisateur de définir son propre problème"""
    print("\n" + "=" * 60)
    print("EXEMPLE PERSONNALISÉ")
    print("=" * 60)
    
    # Choix de la fonction
    print("\nChoisissez une fonction:")
    print("1. f(x,y) = 0.1 * x * y")
    print("2. f(x,y) = π * cos(πx) * y")
    print("3. f(x,y) = -2 * x * y")
    choix_fonction = input("Votre choix (1-3): ")
    
    if choix_fonction == '1':
        f = lambda x, y: 0.1 * x * y
        solution_exacte = lambda x: np.exp(0.05 * x**2)
        nom_fonction = "0.1*x*y"
    elif choix_fonction == '2':
        f = lambda x, y: np.pi * np.cos(np.pi * x) * y
        solution_exacte = lambda x: np.exp(np.sin(np.pi * x))
        nom_fonction = "π*cos(πx)*y"
    else:
        f = lambda x, y: -2 * x * y
        solution_exacte = lambda x: np.exp(-x**2)
        nom_fonction = "-2*x*y"
    
    # Paramètres
    x0 = float(input(f"Valeur initiale x0 (défaut: 0): ") or 0)
    y0 = float(input(f"Valeur initiale y0 (défaut: 1): ") or 1)
    h = float(input(f"Pas h (défaut: 0.1): ") or 0.1)
    x_max = float(input(f"x maximum (défaut: 2): ") or 2)
    
    n_points = int((x_max - x0) / h) + 1
    
    # Choix de la méthode
    print("\nChoisissez la méthode:")
    print("1. Euler")
    print("2. Heun")
    print("3. RK4")
    choix_methode = input("Votre choix (1-3): ")
    
    if choix_methode == '1':
        x_vals, y_vals = methode_euler(f, x0, y0, h, n_points)
        nom_methode = "Euler"
        couleur = 'red'
    elif choix_methode == '2':
        x_vals, y_vals = methode_heun(f, x0, y0, h, n_points)
        nom_methode = "Heun"
        couleur = 'green'
    else:
        x_vals, y_vals = methode_rk4(f, x0, y0, h, n_points)
        nom_methode = "RK4"
        couleur = 'blue'
    
    # Calcul de la solution exacte
    x_exact = np.linspace(x0, x_max, 400)
    y_exact = solution_exacte(x_exact)
    
    # Affichage
    print(f"\nRésultats pour y' = {nom_fonction}, y({x0}) = {y0}")
    print(f"Méthode: {nom_methode}, h = {h}")
    print("\nDerniers points calculés:")
    for i in range(max(0, n_points-5), n_points):
        if i < len(x_vals):
            erreur = abs(y_vals[i] - solution_exacte(x_vals[i]))
            print(f"x = {x_vals[i]:.3f}, y_num = {y_vals[i]:.6f}, "
                  f"y_exact = {solution_exacte(x_vals[i]):.6f}, "
                  f"erreur = {erreur:.6f}")
    
    # Graphique
    plt.figure(figsize=(10, 6))
    plt.plot(x_exact, y_exact, 'k-', linewidth=2, label='Solution exacte')
    plt.plot(x_vals, y_vals, f'{couleur}o--', linewidth=1.5, markersize=6,
             label=f'{nom_methode} (h={h})')
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title(f"$y\' = {nom_fonction}$, $y({x0}) = {y0}$ - Méthode: {nom_methode}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'personnalise_{nom_methode}.png', dpi=150)
    plt.show()

def main():
    """Fonction principale"""
    while True:
        choix = menu_principal()
        
        if choix == '1':
            print("\nExécution de la méthode d'Euler...")
            exemple_euler()
            
        elif choix == '2':
            print("\nExécution de la méthode de Heun...")
            exemple_heun()
            
        elif choix == '3':
            print("\nExécution de la méthode RK4...")
            exemple_rk4()
            
        elif choix == '4':
            print("\nComparaison des 3 méthodes...")
            comparer_methodes()
            
        elif choix == '5':
            print("\nAu revoir!")
            break
            
        elif choix == '6':
            exemple_personnalise()
            
        else:
            print("\nOption invalide. Veuillez réessayer.")
        
        input("\nAppuyez sur Entrée pour continuer...")

def comparer_methodes():
    """Compare les 3 méthodes sur le même problème"""
    # Définition du problème
    def f(x, y):
        return np.pi * np.cos(np.pi * x) * y
    
    def solution_exacte(x):
        return np.exp(np.sin(np.pi * x))
    
    # Paramètres
    x0, y0 = 0, 0.0001
    h = 0.3
    n_points = 10
    
    # Calcul avec les 3 méthodes
    x_euler, y_euler = methode_euler(f, x0, y0, h, n_points)
    x_heun, y_heun = methode_heun(f, x0, y0, h, n_points)
    x_rk4, y_rk4 = methode_rk4(f, x0, y0, h, n_points)
    
    # Solution exacte
    x_exact = np.linspace(x0, x0 + (n_points-1)*h, 400)
    y_exact = solution_exacte(x_exact)
    
    # Affichage des résultats
    print("\n" + "=" * 70)
    print("COMPARAISON DES MÉTHODES - $y\' = \pi \cos(\pi x) y$, $y(0)=0$")
    print("=" * 70)
    print(f"Pas h = {h}, Intervalle: [{x0}, {x0 + (n_points-1)*h:.1f}]")
    
    print("\nErreur au point final (x = {:.1f}):".format(x_euler[-1]))
    print(f"  Euler:  {abs(y_euler[-1] - solution_exacte(x_euler[-1])):.6f}")
    print(f"  Heun:   {abs(y_heun[-1] - solution_exacte(x_heun[-1])):.6f}")
    print(f"  RK4:    {abs(y_rk4[-1] - solution_exacte(x_rk4[-1])):.6f}")
    
    # Graphique de comparaison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Solutions
    axes[0].plot(x_exact, y_exact, 'k-', linewidth=3, label='Solution exacte')
    axes[0].plot(x_euler, y_euler, 'ro--', markersize=6, label='Euler')
    axes[0].plot(x_heun, y_heun, 'gs--', markersize=6, label='Heun')
    axes[0].plot(x_rk4, y_rk4, 'b^--', markersize=6, label='RK4')
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y(x)')
    axes[0].set_title("Comparaison des méthodes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Erreurs
    erreur_euler = [abs(y_euler[i] - solution_exacte(x_euler[i])) for i in range(len(x_euler))]
    erreur_heun = [abs(y_heun[i] - solution_exacte(x_heun[i])) for i in range(len(x_heun))]
    erreur_rk4 = [abs(y_rk4[i] - solution_exacte(x_rk4[i])) for i in range(len(x_rk4))]
    
    axes[1].plot(x_euler, erreur_euler, 'ro-', label='Erreur Euler')
    axes[1].plot(x_heun, erreur_heun, 'gs-', label='Erreur Heun')
    axes[1].plot(x_rk4, erreur_rk4, 'b^-', label='Erreur RK4')
    
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Erreur absolue')
    axes[1].set_title("Évolution des erreurs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('comparaison_methodes.png', dpi=150)
    plt.show()
    
    # Graphique supplémentaire: convergence avec différents pas
    print("\n" + "=" * 70)
    print("ÉTUDE DE CONVERGENCE")
    print("=" * 70)
    
    pas_liste = [0.5, 0.3, 0.1, 0.05]
    erreurs_euler = []
    erreurs_heun = []
    erreurs_rk4 = []
    
    for h_test in pas_liste:
        n_test = int(2 / h_test) + 1
        x_final = x0 + (n_test-1) * h_test
        
        xe, ye = methode_euler(f, x0, y0, h_test, n_test)
        xh, yh = methode_heun(f, x0, y0, h_test, n_test)
        xr, yr = methode_rk4(f, x0, y0, h_test, n_test)
        
        erreurs_euler.append(abs(ye[-1] - solution_exacte(x_final)))
        erreurs_heun.append(abs(yh[-1] - solution_exacte(x_final)))
        erreurs_rk4.append(abs(yr[-1] - solution_exacte(x_final)))
    
    # Graphique de convergence
    plt.figure(figsize=(10, 6))
    plt.loglog(pas_liste, erreurs_euler, 'ro-', linewidth=2, markersize=8, label='Euler')
    plt.loglog(pas_liste, erreurs_heun, 'gs-', linewidth=2, markersize=8, label='Heun')
    plt.loglog(pas_liste, erreurs_rk4, 'b^-', linewidth=2, markersize=8, label='RK4')
    
    # Ajout des pentes théoriques
    x_fit = np.array([0.05, 0.5])
    plt.loglog(x_fit, 10*x_fit, 'r:', alpha=0.5, label='Pente 1 (ordre 1)')
    plt.loglog(x_fit, 10*x_fit**2, 'g:', alpha=0.5, label='Pente 2 (ordre 2)')
    plt.loglog(x_fit, 10*x_fit**4, 'b:', alpha=0.5, label='Pente 4 (ordre 4)')
    
    plt.xlabel('Pas h')
    plt.ylabel('Erreur au point final')
    plt.title('Convergence des méthodes')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('convergence_methodes.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()