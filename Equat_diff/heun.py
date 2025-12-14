import numpy as np
import matplotlib.pyplot as plt

def methode_heun(f, x0, y0, h, n_points):
    """
    Méthode de Heun (Runge-Kutta d'ordre 2)
    
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
    
    # Itération de Heun
    for i in range(n_points - 1):
        x = x_values[i]
        y = y_values[i]
        
        # Étape 1: pente au point initial
        k1 = f(x, y)
        
        # Étape 2: estimation au point milieu
        y_pred = y + h * k1
        
        # Étape 3: pente au point estimé
        k2 = f(x + h, y_pred)
        
        # Étape 4: moyenne des pentes
        pente_moyenne = (k1 + k2) / 2
        
        # Mise à jour
        x_values[i+1] = x + h
        y_values[i+1] = y + h * pente_moyenne
    
    return x_values, y_values

def exemple_heun():
    """Exemple d'utilisation de la méthode de Heun"""
    # Fonction f(x, y) = π cos(πx) y (exemple du cours)
    def f(x, y):
        return np.pi * np.cos(np.pi * x) * y
    
    # Conditions initiales
    x0 = 0
    y0 = 0.0001  # Presque 0 pour éviter division par 0
    h = 0.3  # Pas d'intégration
    n_points = 15  # Nombre de points
    
    # Solution exacte pour comparaison
    def solution_exacte(x):
        return np.exp(np.sin(np.pi * x))
    
    # Application de la méthode de Heun
    x_heun, y_heun = methode_heun(f, x0, y0, h, n_points)
    
    # Génération de la solution exacte
    x_exact = np.linspace(x0, x0 + (n_points-1)*h, 400)
    y_exact = solution_exacte(x_exact)
    
    # Affichage des résultats
    print("MÉTHODE DE HEUN")
    print("=" * 50)
    print(f"Pas h = {h}")
    print(f"Intervalle: [{x0}, {x0 + (n_points-1)*h:.1f}]")
    print("\nPoints calculés (premiers et derniers):")
    for i in [0, 1, 2, -3, -2, -1]:
        if i < len(x_heun):
            print(f"x = {x_heun[i]:.2f}, y_heun = {y_heun[i]:.6f}, "
                  f"y_exact = {solution_exacte(x_heun[i]):.6f}, "
                  f"erreur = {abs(y_heun[i] - solution_exacte(x_heun[i])):.6f}")
    
    # Tracé du graphique
    plt.figure(figsize=(12, 6))
    
    # Solution exacte
    plt.plot(x_exact, y_exact, 'b-', linewidth=2, 
             label='Solution exacte: $e^{\sin(\pi x)}$')
    
    # Solution Heun
    plt.plot(x_heun, y_heun, 'go--', linewidth=1.5, markersize=8,
             label=f"Méthode de Heun (h={h})")
    
    # Ajout des pentes pour le premier point
    if len(x_heun) >= 2:
        x = x_heun[0]
        y = y_heun[0]
        
        # Pente initiale k1
        k1 = f(x, y)
        plt.arrow(x, y, h/3, h/3 * k1, head_width=0.05, head_length=0.1, 
                  fc='orange', ec='orange', alpha=0.7, label='Pente initiale k1')
        
        # Point estimé pour k2
        y_pred = y + h * k1
        plt.plot([x + h], [y_pred], 's', color='purple', markersize=8, 
                 label='Point estimé pour k2')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.title("Méthode de Heun - $y\' = \pi \cos(\pi x) y$, $y(0)=0$", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Limites adaptées à la solution
    y_max = max(np.max(y_exact), np.max(y_heun))
    plt.axis([x0, x0 + (n_points-1)*h, -0.2, y_max + 0.2])
    
    plt.tight_layout()
    plt.savefig('heun_graphe.png', dpi=150)
    plt.show()
    
    return x_heun, y_heun

if __name__ == "__main__":
    exemple_heun()