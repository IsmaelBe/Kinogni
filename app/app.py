from fastapi import FastAPI, HTTPException, status, Request, Depends #Application FastAPI, Gestion erreurs HTTP
from pydantic import BaseModel # Validation des données pour les POST
from typing import Optional # Typage
from time import sleep # 2 secondes de délai entre les tentatives de connexion de la base de données
import requests # Requête HTTP à l'API TMDB
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Analyse de sentiments
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse # Renvoyer des fichiers HTML
from sqlalchemy.orm import Session
from sqlmodel import SQLModel, Field, create_engine, select
from dotenv import dotenv_values
from app.models import Film, Review
from app.database import get_db

templates = Jinja2Templates(directory="templates")
app = FastAPI()

config = dotenv_values(".env")
database_url = f"postgresql://{config['USER']}:{config['MDP_BDD']}@localhost/projet"

class ReviewMAJ(BaseModel):
    contenu: str


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# On pouvait aussi faire avec Pipeline
"""Reponse = pipeline(
        # Tâche analyse de sentiments
        task="sentiment-analysis",
        # Nom du modèle (Plusieurs langues)
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        # Lancer sur CPU
        device=-1
    )"""

# Renvoie la page d'accueil HTML
@app.get("/",response_class=HTMLResponse)
def accueil(request: Request):
    return templates.TemplateResponse("site.html", {"request": request})


#Recherche film par nom, pour garder les espaces, on le met sous forme de query
#Exemple: http://127.0.0.1:8000/films/liste?nom=The Dark Knight
@app.get("/films/liste")
def obtenir_films(nom: str, db: Session = Depends(get_db)):
    #%s pour éviter les injections SQL
    films = db.query(Film).filter(Film.titre.ilike(f"%{nom}%")).all()

    if films: 
        return {'data': films}
    else:
        # Si le nom de film n'est pas dans la base de données
        films_tmdb = verification_tmdb(nom, db)
        if films_tmdb:
            # Le film est sur l'API TMDB
            return {'data': films_tmdb}
        else:
            # Erreur 404
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Aucun film trouvé dans la base locale ni sur TMDB"
            )

@app.put("/films/reviews/update/{id}")
def update_review(id: str, rev: ReviewMAJ, db: Session = Depends(get_db)):
    review = db.query(Review).filter(Review.review_id == id).first()

    if not review:
        raise HTTPException(status_code=404, detail=f"Aucune review trouvée avec l'id {id}")

    # Mise à jour du contenu et du champ "edited"
    review.contenu = rev.contenu
    review.edited = 1
    db.commit()
    db.refresh(review)

    return {
        "message": "Review mise à jour avec succès",
        "id": id,
        "nouveau_contenu": review.contenu
    }

@app.delete("/films/reviews/delete/{id}")
def delete_review(id: str, db: Session = Depends(get_db)):
    review_cible = db.query(Review).filter(Review.review_id == id).first()

    if not review_cible:
        raise HTTPException(status_code=404, detail=f"Aucune review trouvée avec l'id {id}")
    
    db.delete(review_cible)
    db.commit()
    return {
        "message": "Review mise à jour avec succès (message supprimé)",
    }


# Reviews d'un film précis selon l'identifiant
@app.get("/films/reviews/{film_id}")
def get_reviews(film_id: int, db: Session = Depends(get_db)):
    try:
        # On selectionne le nom de film, date sortie, l'auteur de la review et le contenu
        film = db.query(Film).filter(Film.film_id == film_id).first()

        if not film:
            raise HTTPException(status_code=404, detail="Film introuvable")

        # Vérifie si le film a au moins 1 review
        has_reviews = film.reviews and any(r.auteur for r in film.reviews)

        if has_reviews:
            # Analyse de sentiments avec l'auteur et le contenu
            reviews_locales = [{"author": r.auteur, "content": r.contenu} for r in film.reviews if r.auteur]
            sentiments = sentiment_analysis(reviews_locales)
            return {"data": film.reviews, "Sentiments": sentiments}

        # On regarde sur l'API TMDB les reviews avec une requête GET
        API_KEY = config["API_KEY"]
        url = f"https://api.themoviedb.org/3/movie/{film_id}/reviews"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)

        # Si la requête ne marche pas
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Film introuvable sur TMDB")

        # La liste des reviews
        reviews = response.json()["results"]

        for review in reviews:
            # On insère pour chaque review dans la table review SI l'id est unique
            new_review = Review(
                review_id=review["id"],
                film_id=film_id,
                auteur=review["author"],
                contenu=review["content"]
            )
            db.merge(new_review)
        db.commit()
        # Analyse de sentiment
        sentiments = sentiment_analysis(reviews)
        return {"Avis": reviews, "Sentiments": sentiments}
    # Problème connexion base de donnée ou autre
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")




# Fonction pour interroger l'API TMDB
def verification_tmdb(nom: str, db: Session):

    #Requête GET sur l'API TMDB
    API_KEY = config["API_KEY"]
    endpoint = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": nom}
    response = requests.get(endpoint, params=params)

    # Problème de requête
    if response.status_code != 200:
        return []

    # On prend les résultats
    results = response.json().get("results", [])
    if not results:
        return []

    # On garde seulement les films contenant le mot cherché
    films = [
        {"id": r["id"], "titre": r["title"], "date_sortie": r["release_date"]}
        for r in results
        if nom.lower() in r["title"].lower()
    ]
    # On insère chaque film dans la table si l'id est unique
    for film in films:
        new_film = Film(film_id=film["id"], titre=film["titre"], date_sortie=film["date_sortie"])
        db.merge(new_film)
    db.commit()
    return films

# Fonction analyse de sentiments
ratings = [[1, "Très mauvais"],[2, "Mauvais"],[3, "Mitigé"],[4, "Positif"],[5, "Très positif"]]

def sentiment_analysis(reviews):
    try:
        sentiments = []
        # Pour chaque review (500 charactères max), on analyse la positivité/négativité (de 0 à 1)
        for review in reviews:
            content = review.get("content", "")

            tokens = tokenizer.encode(content[:512], return_tensors='pt')
            outputs = model(tokens)
            logits = outputs.logits
            sentiment = int(torch.argmax(logits)) + 1
            val_precise = float(torch.max(logits))
            # Retourne un tenseur avec une liste de 5 valeurs (très négatif à très positif),
            # Le nombre le plus grand (entre 1 et 5) est le sentiment le plus probable
            for index, val in enumerate(ratings):
                if sentiment == index+1:
                    sentiments.append([val[1], val_precise])
        return sentiments
    except Exception as error:
        raise error


#Voir dictionnaire pythorch
#AVancer compte rendu
#jeudi 11h le 20 partie sentiments + rendre à l'utilisateur
