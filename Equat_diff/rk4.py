import numpy as np
import matplotlib.pyplot as plt

def methode_rk4(f, x0, y0, h, n_points):
    """
    Méthode de Runge-Kutta d'ordre 4 (RK4)
    
    Parameters:
    -----------
    f : fonction f(x, y) = dy/dx
    x0 : valeur initiale de x
    y0 : valeur initiale de y
    h : pas d'intégration
    n_points : nombre de points à calculer
    
    Returns:
    --------
    x_values, y_values : arrays des points calculés
    """
    # Initialisation
    x_values = np.zeros(n_points)
    y_values = np.zeros(n_points)
    
    x_values[0] = x0
    y_values[0] = y0
    
    # Itération RK4
    for i in range(n_points - 1):
        x = x_values[i]
        y = y_values[i]
        
        # Calcul des 4 pentes
        k1 = f(x, y)
        k2 = f(x + h/2, y + h * k1/2)
        k3 = f(x + h/2, y + h * k2/2)
        k4 = f(x + h, y + h * k3)
        
        # Combinaison des pentes
        pente_moyenne = (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Mise à jour
        x_values[i+1] = x + h
        y_values[i+1] = y + h * pente_moyenne
    
    return x_values, y_values

def exemple_rk4():
    """Exemple d'utilisation de la méthode RK4"""
    # Fonction f(x, y) = π cos(πx) y (même exemple que Heun)
    def f(x, y):
        return np.pi * np.cos(np.pi * x) * y
    
    # Conditions initiales
    x0 = 0
    y0 = 0.0001  # Presque 0 pour éviter division par 0
    h = 0.5  # Pas d'intégration plus grand
    n_points = 10  # Nombre de points
    
    # Solution exacte pour comparaison
    def solution_exacte(x):
        return np.exp(np.sin(np.pi * x))
    
    # Application de la méthode RK4
    x_rk4, y_rk4 = methode_rk4(f, x0, y0, h, n_points)
    
    # Pour comparaison: Euler et Heun avec le même pas
    def methode_euler_simple(f, x0, y0, h, n_points):
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        x[0], y[0] = x0, y0
        for i in range(n_points-1):
            y[i+1] = y[i] + h * f(x[i], y[i])
            x[i+1] = x[i] + h
        return x, y
    
    def methode_heun_simple(f, x0, y0, h, n_points):
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        x[0], y[0] = x0, y0
        for i in range(n_points-1):
            k1 = f(x[i], y[i])
            k2 = f(x[i] + h, y[i] + h * k1)
            y[i+1] = y[i] + h * (k1 + k2) / 2
            x[i+1] = x[i] + h
        return x, y
    
    x_euler, y_euler = methode_euler_simple(f, x0, y0, h, n_points)
    x_heun, y_heun = methode_heun_simple(f, x0, y0, h, n_points)
    
    # Génération de la solution exacte
    x_exact = np.linspace(x0, x0 + (n_points-1)*h, 400)
    y_exact = solution_exacte(x_exact)
    
    # Affichage des résultats
    print("MÉTHODE DE RUNGE-KUTTA D'ORDRE 4 (RK4)")
    print("=" * 60)
    print(f"Pas h = {h} (comparaison avec Euler et Heun)")
    print(f"Intervalle: [{x0}, {x0 + (n_points-1)*h:.1f}]")
    print("\nComparaison des erreurs au point final:")
    print(f"Euler:   erreur = {abs(y_euler[-1] - solution_exacte(x_euler[-1])):.6f}")
    print(f"Heun:    erreur = {abs(y_heun[-1] - solution_exacte(x_heun[-1])):.6f}")
    print(f"RK4:     erreur = {abs(y_rk4[-1] - solution_exacte(x_rk4[-1])):.6f}")
    
    # Tracé du graphique comparatif
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Solutions
    axes[0].plot(x_exact, y_exact, 'k-', linewidth=3, label='Solution exacte')
    axes[0].plot(x_euler, y_euler, 'r--', linewidth=1.5, markersize=6, marker='o',
                label=f'Euler (h={h})')
    axes[0].plot(x_heun, y_heun, 'g--', linewidth=1.5, markersize=6, marker='s',
                label=f'Heun (h={h})')
    axes[0].plot(x_rk4, y_rk4, 'b--', linewidth=1.5, markersize=6, marker='^',
                label=f'RK4 (h={h})')
    
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y(x)', fontsize=12)
    axes[0].set_title("Comparaison des méthodes - $y\' = \pi \cos(\pi x) y$", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Graphique 2: Erreurs
    erreur_euler = [abs(y_euler[i] - solution_exacte(x_euler[i])) for i in range(len(x_euler))]
    erreur_heun = [abs(y_heun[i] - solution_exacte(x_heun[i])) for i in range(len(x_heun))]
    erreur_rk4 = [abs(y_rk4[i] - solution_exacte(x_rk4[i])) for i in range(len(x_rk4))]
    
    axes[1].plot(x_euler, erreur_euler, 'r-o', label='Erreur Euler', alpha=0.7)
    axes[1].plot(x_heun, erreur_heun, 'g-s', label='Erreur Heun', alpha=0.7)
    axes[1].plot(x_rk4, erreur_rk4, 'b-^', label='Erreur RK4', alpha=0.7)
    
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('Erreur absolue', fontsize=12)
    axes[1].set_title("Évolution des erreurs", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('rk4_comparaison.png', dpi=150)
    plt.show()
    
    # Graphique supplémentaire: schéma RK4
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Visualisation des 4 pentes pour le premier pas
    if len(x_rk4) >= 2:
        x = x_rk4[0]
        y = y_rk4[0]
        
        # k1
        k1 = f(x, y)
        ax2.arrow(x, y, h/4, h/4 * k1, head_width=0.02, head_length=0.05,
                 fc='red', ec='red', width=0.002, label='k1')
        
        # k2
        ax2.arrow(x + h/2, y + h/2 * k1, h/4, h/4 * k2, head_width=0.02, head_length=0.05,
                 fc='green', ec='green', width=0.002, label='k2')
        
        # k3
        ax2.arrow(x + h/2, y + h/2 * k2, h/4, h/4 * k3, head_width=0.02, head_length=0.05,
                 fc='blue', ec='blue', width=0.002, label='k3')
        
        # k4
        ax2.arrow(x + h, y + h * k3, h/4, h/4 * k4, head_width=0.02, head_length=0.05,
                 fc='purple', ec='purple', width=0.002, label='k4')
        
        # Solution exacte
        ax2.plot(x_exact, y_exact, 'k-', linewidth=2, alpha=0.5, label='Solution exacte')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title("Schéma RK4: les 4 pentes pour un pas")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rk4_schema.png', dpi=150)
        plt.show()
    
    return x_rk4, y_rk4

if __name__ == "__main__":
    exemple_rk4()