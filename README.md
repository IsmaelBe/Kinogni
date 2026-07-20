# Kinogni

**Application locale d'analyse de sentiments sur les avis de films.**

Kinogni permet de rechercher un film, de lire et publier des avis, puis d'analyser
automatiquement le ressenti de ces avis grâce à un modèle d'intelligence artificielle
(**BERT**). L'application combine une API REST (FastAPI), une base de données PostgreSQL,
l'API **TMDB** (pour récupérer films et avis réels) et le modèle BERT (pour classer chaque
avis sur 5 niveaux, de « Très mauvais » à « Très positif »).

> ⚠️ **Projet étudiant, conçu pour tourner en local.** Le modèle BERT et la base de
> données s'exécutent sur votre propre machine ; l'application n'est pas déployée en ligne.

---

## Ce que fait l'application

1. **Recherche de films** — par titre. Si le film n'est pas déjà en base, il est récupéré
   automatiquement depuis l'API TMDB et enregistré.
2. **Gestion des avis** — inscription / connexion (mots de passe hachés, authentification par
   token JWT), puis ajout, modification et suppression de ses propres avis.
3. **Analyse de sentiments** — pour un film, les avis TMDB et les avis locaux sont réunis,
   puis chaque avis est classé par BERT sur 5 niveaux avec un score de confiance. Une
   **synthèse globale** (« Bien reçu », « Avis mitigés », etc.) est calculée.

---

## Prérequis

- **Python 3.13**
- **PostgreSQL** installé et démarré sur `localhost`, avec une base de données nommée `projet`
- Une **clé API TMDB** (gratuite sur https://www.themoviedb.org/settings/api)

---

## Installation

### 1. Dépendances

```bash
pip install -r requirements.txt
```

> Le premier lancement télécharge le modèle BERT (~700 Mo) depuis Hugging Face. Prévoir
> quelques minutes et assez de RAM (le modèle est chargé en mémoire au démarrage).

### 2. Base de données

Créez la base PostgreSQL attendue (les tables, elles, sont créées automatiquement au
premier démarrage) :

```bash
createdb projet
```

### 3. Configuration

Créez un fichier `.env` à la racine du projet avec ces variables :

```
USER=votre_utilisateur_postgres
MDP_BDD=votre_mot_de_passe
API_KEY=votre_cle_api_tmdb
SECRET_KEY=une_longue_chaine_aleatoire_secrete
```

`SECRET_KEY` sert à signer les tokens de connexion (JWT) : elle est **obligatoire**.
Vous pouvez en générer une avec :

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Lancement

```bash
uvicorn app.app:app --reload
```

L'application est alors disponible sur **http://127.0.0.1:8000**.

---

## Structure du projet

Le code est découpé par responsabilité :

| Fichier | Rôle |
|---|---|
| `app/app.py` | Routes et configuration FastAPI (films, avis, rate limiting, CORS). |
| `app/auth.py` | Inscription, connexion, hachage des mots de passe et authentification JWT. |
| `app/logic.py` | Analyse de sentiments BERT, appels à l'API TMDB et synthèse. |
| `app/schemas.py` | Modèles Pydantic pour valider les données reçues. |
| `app/models.py` | Tables de la base de données (SQLAlchemy). |
| `app/database.py` | Connexion à PostgreSQL. |
| `templates/` | Pages HTML de l'interface (accueil, dashboard, confidentialité). |
