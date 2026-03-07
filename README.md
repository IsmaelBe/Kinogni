
# Kinogni

Kinogni - application d'analyse de Sentiments

Application (en local) permettant de rechercher des films, de gérer des avis personnalisés et d'analyser automatiquement leur ton grâce à l'intelligence artificielle.

# Fonctionnalités

    Recherche de films via l'API TMDB.

    Système de gestion des avis (Ajout, Modification, Suppression).

    Analyse automatique des sentiments (modèle BERT).

    Inscription et connexion sécurisées.

# Installation
1. Dépendances

Installez les bibliothèques nécessaires à l'aide du fichier fourni :

    pip install -r requirements.txt
2. Configuration

Créez un fichier .env à la racine du projet avec les variables suivantes :
Extrait de code:

    USER=votre_utilisateur_postgres
    MDP_BDD=votre_mot_de_passe
    API_KEY=votre_cle_api_tmdb

3. Lancement

Pour démarrer le serveur de développement :

    uvicorn app.app:app --reload

L'application sera disponible à l'adresse : http://127.0.0.1:8000


# Structure du projet

Le code est segmenté pour une meilleure lisibilité :

    app/app.py : Routes et configuration FastAPI.

    app/logic.py : Analyse de sentiments BERT, appels API TMDB et sécurité.

    app/schemas.py : Modèles Pydantic pour la validation des entrées.

    app/models.py : Définition des tables de la base de données.

    app/database.py : Connexion à PostgreSQL.

    templates/ : Pages HTML de l'interface.
