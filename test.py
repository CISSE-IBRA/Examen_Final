import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from scipy import integrate, special
warnings.filterwarnings('ignore')

class IntegrationComparer:
    def __init__(self):
        self.methods = ['gauss_laguerre', 'gauss_legendre', 'gauss_chebyshev', 'simpson', 'spline']
        self.colors = ['r', 'g', 'b', 'm', 'c']
        self.markers = ['o', 's', '^', 'D', 'v']
    
    # 1. Exemple famille Chebyshev : 1/√(1-x²) sur [-1, 1]
    def f_chebyshev(self, x):
        return 1.0 / np.sqrt(1 - x**2 + 1e-15)  # ajout d'un petit epsilon pour éviter division par zéro
    
    # 2. Exemple famille Laguerre : exp(-x) * sin(x) sur [0, ∞]
    def f_laguerre(self, x):
        return np.exp(-x) * np.sin(x)
    
    # 3. Combinaison des deux : exp(-x)/√(1-x²) (adapté pour [0,1])
    def f_combined(self, x):
        return np.exp(-x) / np.sqrt(1 - x**2 + 1e-15)
    
    # 4. Autre quelconque : sin(x²) sur [0, 2]
    def f_arbitrary(self, x):
        return np.sin(x**2)
    
    def gauss_legendre_integral(self, f, a, b, n):
        """Intégration par quadrature de Gauss-Legendre"""
        x, w = np.polynomial.legendre.leggauss(n)
        # Transformation de [-1,1] vers [a,b]
        x_transformed = 0.5 * (b - a) * x + 0.5 * (a + b)
        return 0.5 * (b - a) * np.sum(w * f(x_transformed))
    
    def gauss_laguerre_integral(self, f, n):
        """Intégration par quadrature de Gauss-Laguerre pour [0,∞] avec poids exp(-x)"""
        x, w = np.polynomial.laguerre.laggauss(n)
        return np.sum(w * f(x))
    
    def gauss_chebyshev_integral(self, f, n, a=-1, b=1):
        """Intégration par quadrature de Gauss-Chebyshev de première espèce"""
        # Les nœuds et poids de Chebyshev
        k = np.arange(1, n+1)
        x = np.cos((2*k - 1) * np.pi / (2*n))
        w = np.pi / n * np.ones(n)
        
        # Ajustement pour l'intervalle [a,b]
        x_scaled = 0.5 * (b - a) * x + 0.5 * (a + b)
        w_scaled = 0.5 * (b - a) * w / np.pi * np.sqrt(1 - x**2)
        
        return np.sum(w_scaled * f(x_scaled))
    
    def simpson_integral(self, f, a, b, n):
        """Intégration par méthode de Simpson composite"""
        if n % 2 == 0:
            n += 1  # Simpson nécessite un nombre impair de points
        
        x = np.linspace(a, b, n)
        h = (b - a) / (n - 1)
        y = f(x)
        
        # Formule composite de Simpson
        result = y[0] + y[-1]
        result += 4 * np.sum(y[1:-1:2])
        result += 2 * np.sum(y[2:-2:2])
        
        return result * h / 3
    
    def spline_integral(self, f, a, b, n):
        """Intégration par splines cubiques"""
        x = np.linspace(a, b, n)
        y = f(x)
        
        # Coefficients des splines cubiques
        h = np.diff(x)
        delta = np.diff(y) / h
        
        # Résolution du système tridiagonal pour les dérivées secondes
        n_points = len(x)
        A = np.zeros((n_points, n_points))
        B = np.zeros(n_points)
        
        # Conditions naturelles
        A[0, 0] = 1
        A[-1, -1] = 1
        
        for i in range(1, n_points-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            B[i] = 6 * (delta[i] - delta[i-1])
        
        M = np.linalg.solve(A, B)
        
        # Calcul de l'intégrale
        integral = 0
        for i in range(n_points-1):
            integral += h[i] * (y[i] + y[i+1]) / 2
            integral -= h[i]**3 * (M[i] + M[i+1]) / 24
        
        return integral
    
    def compute_integral(self, f, method, n, a=None, b=None):
        """Calcule l'intégrale avec la méthode spécifiée"""
        start_time = time.time()
        
        try:
            if method == 'gauss_legendre':
                result = self.gauss_legendre_integral(f, a, b, n)
            elif method == 'gauss_laguerre':
                result = self.gauss_laguerre_integral(f, n)
            elif method == 'gauss_chebyshev':
                result = self.gauss_chebyshev_integral(f, n, a, b)
            elif method == 'simpson':
                result = self.simpson_integral(f, a, b, n)
            elif method == 'spline':
                result = self.spline_integral(f, a, b, n)
            else:
                result = None
        except Exception as e:
            result = None
        
        computation_time = time.time() - start_time
        return result, computation_time
    
    def exact_integrals(self):
        """Valeurs exactes des intégrales (quand disponibles)"""
        # Valeurs exactes calculées avec scipy ou analytiquement
        exact_values = {}
        
        # f_chebyshev sur [-1, 1] : ∫ 1/√(1-x²) dx = π
        exact_values['chebyshev'] = np.pi
        
        # f_laguerre sur [0, ∞] : ∫ exp(-x) sin(x) dx = 0.5
        exact_values['laguerre'] = 0.5
        
        # f_combined sur [0, 0.99] (évite la singularité en 1)
        # Valeur numérique de référence avec haute précision
        result, _ = integrate.quad(self.f_combined, 0, 0.99, limit=1000)
        exact_values['combined'] = result
        
        # f_arbitrary sur [0, 2]
        # ∫ sin(x²) dx ≈ 0.804776 (calculé avec scipy)
        result, _ = integrate.quad(self.f_arbitrary, 0, 2)
        exact_values['arbitrary'] = result
        
        return exact_values
    
    def run_comparison(self, n_values):
        """Exécute la comparaison pour toutes les méthodes et exemples"""
        functions = [
            ('chebyshev', self.f_chebyshev, (-0.999, 0.999)),  # Évite les bords
            ('laguerre', self.f_laguerre, (0, None)),  # [0, ∞] pour Laguerre
            ('combined', self.f_combined, (0, 0.99)),  # Évite la singularité
            ('arbitrary', self.f_arbitrary, (0, 2))
        ]
        
        exact_values = self.exact_integrals()
        
        # Stockage des résultats
        errors = {method: {fname: [] for fname, _, _ in functions} for method in self.methods}
        times = {method: {fname: [] for fname, _, _ in functions} for method in self.methods}
        
        for fname, func, (a, b) in functions:
            print(f"\n{'='*60}")
            print(f"Fonction: {fname}")
            print(f"{'='*60}")
            
            for n in n_values:
                print(f"\nn = {n}:")
                
                for method in self.methods:
                    # Pour Laguerre, pas besoin de bornes
                    if method == 'gauss_laguerre' and fname == 'laguerre':
                        result, comp_time = self.compute_integral(func, method, n)
                    elif method == 'gauss_laguerre':
                        continue  # Sauter Laguerre pour les autres fonctions
                    else:
                        result, comp_time = self.compute_integral(func, method, n, a, b)
                    
                    if result is not None:
                        # Calcul de l'erreur
                        if fname in exact_values:
                            error = abs(result - exact_values[fname])
                        else:
                            error = None
                        
                        errors[method][fname].append(error if error is not None else np.nan)
                        times[method][fname].append(comp_time)
                        
                        print(f"  {method:20s}: {result:.8e}, temps: {comp_time:.2e}s, "
                              f"erreur: {error:.2e}" if error is not None else "N/A")
        
        return errors, times
    
    def plot_results(self, n_values, errors, times):
        """Crée les graphiques d'erreur et de temps de calcul"""
        
        functions = ['chebyshev', 'laguerre', 'combined', 'arbitrary']
        
        # Graphique 1: Erreur en fonction de n (log-log)
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
        axes1 = axes1.flatten()
        
        for idx, fname in enumerate(functions):
            ax = axes1[idx]
            for method, color, marker in zip(self.methods, self.colors, self.markers):
                if fname in errors[method] and errors[method][fname]:
                    valid_errors = [err for err in errors[method][fname] if err is not None and not np.isnan(err)]
                    if valid_errors and len(valid_errors) == len(n_values):
                        ax.loglog(n_values, valid_errors, color=color, marker=marker, 
                                 label=method, linewidth=2)
            
            ax.set_xlabel('Nombre de points n', fontsize=10)
            ax.set_ylabel('Erreur absolue', fontsize=10)
            ax.set_title(f'Fonction: {fname}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        
        # Graphique 2: Temps de calcul en fonction de n
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        axes2 = axes2.flatten()
        
        for idx, fname in enumerate(functions):
            ax = axes2[idx]
            for method, color, marker in zip(self.methods, self.colors, self.markers):
                if fname in times[method] and times[method][fname]:
                    ax.semilogy(n_values, times[method][fname], color=color, marker=marker,
                               label=method, linewidth=2)
            
            ax.set_xlabel('Nombre de points n', fontsize=10)
            ax.set_ylabel('Temps de calcul (s)', fontsize=10)
            ax.set_title(f'Fonction: {fname}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        
        # Graphique 3: Comparaison globale des performances
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Erreur moyenne sur toutes les fonctions
        avg_errors = []
        for method in self.methods:
            method_errors = []
            for fname in functions:
                if fname in errors[method]:
                    valid_errors = [err for err in errors[method][fname] 
                                   if err is not None and not np.isnan(err)]
                    if valid_errors:
                        method_errors.append(np.mean(valid_errors))
            if method_errors:
                avg_errors.append(np.mean(method_errors))
            else:
                avg_errors.append(np.nan)
        
        # Temps moyen de calcul
        avg_times = []
        for method in self.methods:
            method_times = []
            for fname in functions:
                if fname in times[method]:
                    method_times.extend(times[method][fname])
            if method_times:
                avg_times.append(np.mean(method_times))
            else:
                avg_times.append(np.nan)
        
        x_pos = np.arange(len(self.methods))
        
        ax1.bar(x_pos, avg_errors, color=self.colors, alpha=0.7)
        ax1.set_xlabel('Méthode')
        ax1.set_ylabel('Erreur moyenne (log)')
        ax1.set_title('Erreur moyenne des différentes méthodes')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', '\n') for m in self.methods])
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(x_pos, avg_times, color=self.colors, alpha=0.7)
        ax2.set_xlabel('Méthode')
        ax2.set_ylabel('Temps moyen (s)')
        ax2.set_title('Temps moyen de calcul des différentes méthodes')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.replace('_', '\n') for m in self.methods])
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig1, fig2, fig3

def main():
    # Initialisation
    comparer = IntegrationComparer()
    
    # Valeurs de n à tester
    n_values = [5, 10, 20, 30, 50, 100]
    
    print("="*60)
    print("COMPARAISON DES MÉTHODES D'INTÉGRATION NUMÉRIQUE")
    print("="*60)
    print("\nMéthodes implémentées:")
    for i, method in enumerate(comparer.methods):
        print(f"  {i+1}. {method}")
    
    print("\nFonctions testées:")
    print("  1. Famille Chebyshev: 1/√(1-x²) sur [-1, 1]")
    print("  2. Famille Laguerre: exp(-x)sin(x) sur [0, ∞]")
    print("  3. Combinaison: exp(-x)/√(1-x²) sur [0, 0.99]")
    print("  4. Autre: sin(x²) sur [0, 2]")
    print("\n" + "="*60)
    
    # Exécution de la comparaison
    errors, times = comparer.run_comparison(n_values)
    
    # Génération des graphiques
    print("\n\nGénération des graphiques...")
    fig1, fig2, fig3 = comparer.plot_results(n_values, errors, times)
    
    # Affichage des résultats
    plt.show()
    
    # Sauvegarde des graphiques
    fig1.savefig('erreurs_integration.png', dpi=300, bbox_inches='tight')
    fig2.savefig('temps_calcul_integration.png', dpi=300, bbox_inches='tight')
    fig3.savefig('performance_globale.png', dpi=300, bbox_inches='tight')
    
    print("\nGraphiques sauvegardés:")
    print("  - erreurs_integration.png")
    print("  - temps_calcul_integration.png")
    print("  - performance_globale.png")
    
    # Affichage des conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS GÉNÉRALES")
    print("="*60)
    print("\n1. Méthodes de Gauss:")
    print("   - Excellente précision pour peu de points")
    print("   - Spécialisées selon la fonction poids")
    print("   - Laguerre idéal pour intégrales sur [0,∞] avec exp(-x)")
    
    print("\n2. Méthode de Simpson:")
    print("   - Simple et robuste")
    print("   - Convergence polynomiale")
    print("   - Bon compromis précision/temps")
    
    print("\n3. Méthode des splines:")
    print("   - Très flexible")
    print("   - Moins précise pour les fonctions singulières")
    print("   - Plus coûteuse en temps de calcul")

if __name__ == "__main__":
    main()