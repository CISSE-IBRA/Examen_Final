import numpy as np
import matplotlib.pyplot as plt

def methode_euler(f, x0, y0, h, n_points):
    """
    Méthode d'Euler explicite
    
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
    
    # Itération d'Euler
    for i in range(n_points - 1):
        x = x_values[i]
        y = y_values[i]
        
        # Calcul de la pente
        pente = f(x, y)
        
        # Mise à jour
        x_values[i+1] = x + h
        y_values[i+1] = y + h * pente
    
    return x_values, y_values

def exemple_euler():
    """Exemple d'utilisation de la méthode d'Euler"""
    # Fonction f(x, y) = dy/dx = 0.1 * x * y
    def f(x, y):
        return 0.1 * x * y
    
    # Conditions initiales
    x0 = 0
    y0 = 1
    h = 0.5  # Pas d'intégration
    n_points = 10  # Nombre de points
    
    # Solution exacte pour comparaison
    def solution_exacte(x):
        return np.exp(0.05 * x**2)
    
    # Application de la méthode d'Euler
    x_euler, y_euler = methode_euler(f, x0, y0, h, n_points)
    
    # Génération de la solution exacte
    x_exact = np.linspace(x0, x0 + (n_points-1)*h, 400)
    y_exact = solution_exacte(x_exact)
    
    # Affichage des résultats
    print("MÉTHODE D'EULER")
    print("=" * 50)
    print(f"Pas h = {h}")
    print(f"Intervalle: [{x0}, {x0 + (n_points-1)*h}]")
    print("\nPoints calculés:")
    for i in range(len(x_euler)):
        print(f"x = {x_euler[i]:.2f}, y_euler = {y_euler[i]:.6f}, "
              f"y_exact = {solution_exacte(x_euler[i]):.6f}, "
              f"erreur = {abs(y_euler[i] - solution_exacte(x_euler[i])):.6f}")
    
    # Tracé du graphique
    plt.figure(figsize=(10, 6))
    
    # Solution exacte
    plt.plot(x_exact, y_exact, 'b-', linewidth=2, label='Solution exacte: $e^{0.05x^2}$')
    
    # Solution Euler
    plt.plot(x_euler, y_euler, 'ro--', linewidth=1.5, markersize=8, 
             label=f"Méthode d'Euler (h={h})")
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.title("Méthode d'Euler - $y\' = 0.1 x y$, $y(0)=1$", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis([x0, x0 + (n_points-1)*h, 0, 2.5])
    
    # Ajout des annotations pour le premier pas
    if len(x_euler) >= 2:
        plt.annotate(f'Pente à x={x_euler[0]}: f({x_euler[0]:.1f},{y_euler[0]:.2f}) = {f(x_euler[0], y_euler[0]):.2f}',
                    xy=(x_euler[0], y_euler[0]), xytext=(x_euler[0]+0.5, y_euler[0]-0.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('euler_graphe.png', dpi=150)
    plt.show()
    
    return x_euler, y_euler

if __name__ == "__main__":
    exemple_euler()