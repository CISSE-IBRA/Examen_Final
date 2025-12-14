import matplotlib.pyplot as plt
import os


def plot_all(results):
    # Crée le dossier results/graphs si il n’existe pas déjà
    os.makedirs('results/graphs', exist_ok=True)

    # results est un dictionnaire contenant :
    # - les valeurs de n testées (Ns)
    # - les erreurs obtenues pour chaque méthode (errs)
    # - les temps d’exécution pour chaque méthode (times)
    #
    # Exemple de structure :
    # results = {
    #   "Exemple1": {
    #       "Ns": [10,20,30,...],
    #       "errs": {"Simpson":[...], "Gauss-Legendre":[...], ...},
    #       "times": {"Simpson":[...], "Gauss-Legendre":[...], ...},
    #   },
    #   ...
    # }

    for name, res in results.items():

        # Récupération des valeurs utiles pour le graphique
        Ns = res['Ns']        # Liste des tailles n
        errs = res['errs']    # Erreurs par méthode
        times = res['times']  # Temps par méthode

        # ---------------------------
        #   GRAPHIQUE DES ERREURS
        # ---------------------------

        plt.figure(figsize=(8,5))  # ouverture d’une figure matplotlib

        # On trace une courbe pour chaque méthode numérique
        for methode, valeurs in errs.items():
            plt.plot(Ns, valeurs, label=methode)

        plt.yscale('log')  # échelle logarithmique pour mieux visualiser les erreurs
        plt.xlabel('n')     # axe des x
        plt.ylabel('Erreur') # axe des y
        plt.title(f'Erreurs - {name}')  # titre incluant le nom du test
        plt.legend()         # légende des méthodes
        plt.grid(True)       # grille pour la lisibilité

        # Enregistrement du graphique dans un fichier PNG
        plt.savefig(f"results/graphs/{name}_erreurs.png")

        # Fermeture de la figure pour éviter des conflits entre graphiques
        plt.close()

        # ---------------------------
        #   GRAPHIQUE DES TEMPS
        # ---------------------------

        plt.figure(figsize=(8,5))

        # Trace une courbe par méthode
        for methode, valeurs in times.items():
            plt.plot(Ns, valeurs, label=methode)

        plt.xlabel('n')
        plt.ylabel('Temps (s)')
        plt.title(f'Temps - {name}')
        plt.legend()
        plt.grid(True)

        plt.savefig(f"results/graphs/{name}_temps.png")
        plt.close()
