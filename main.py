# On importe la fonction run_all depuis le module compute.
# Cette fonction s’occupe d’exécuter toutes les méthodes d'intégration numérique
# (Gauss-Legendre, Gauss-Laguerre, Gauss-Chebyshev, Simpson, Spline)
# sur tous les exemples définis dans le projet.
from Integration_num.compute import run_all

# On importe la fonction plot_all depuis le module plots.
# Cette fonction génère les graphiques (erreurs + temps) pour chaque exemple.
# Dans la version modifiée, tu utiliseras plot_all_interactive à la place.
from Integration_num.plots import plot_all

# Module qui générait les rapports (PDF/Markdown).
# Comme tu n'en veux plus, il reste juste commenté.
# from Integration_num.report import generate_reports


# Point d’entrée principal du programme.
# Ce bloc ne s’exécute QUE si le fichier main.py est lancé directement.
if __name__ == "__main__":

    # ----------------------------------------------------
    # Appel de run_all()
    # ----------------------------------------------------
    # Cette fonction lance toutes les méthodes numériques,
    # calcule les erreurs pour différents n, mesure les temps,
    # et retourne un dictionnaire contenant tous les résultats.
    results = run_all()

    # ----------------------------------------------------
    # Génération des graphiques
    # ----------------------------------------------------
    # plot_all() reçoit les résultats et génère les courbes :
    # - Erreur en fonction de n
    # - Temps en fonction de n
    # Pour chaque exemple, une ou plusieurs figures sont créées.
    plot_all(results)

    # ----------------------------------------------------
    # Génération des rapports (désactivée)
    # ----------------------------------------------------
    # Dans ta version finale, tu as demandé de retirer cette étape.
    # generate_reports(results)

    # ----------------------------------------------------
    # Message final
    # ----------------------------------------------------
    # Indique simplement que le traitement s’est bien terminé.
    print("Traitement terminé. Voir dossier results/")
