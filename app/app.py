from fastapi import FastAPI, HTTPException, status, Request, Depends #Application FastAPI, Gestion erreurs HTTP
from pydantic import BaseModel # Validation des données pour les POST
from typing import Optional # Typage
from time import sleep # 2 secondes de délai entre les tentatives de connexion de la base de données
import requests # Requête HTTP à l'API TMDB
import torch
import uuid
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Analyse de sentiments
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse # Renvoyer des fichiers HTML
from sqlalchemy.orm import Session
from sqlmodel import SQLModel, Field, create_engine, select
from dotenv import dotenv_values
from app.models import Film, Review, Users
from app.database import get_db
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext

templates = Jinja2Templates(directory="templates")
app = FastAPI()
# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Autorise toutes les origines (utile en développement)
    allow_credentials=True,
    allow_methods=["*"], # Autorise GET, POST, etc.
    allow_headers=["*"], # Autorise tous les headers
)

config = dotenv_values(".env")
database_url = f"postgresql://{config['USER']}:{config['MDP_BDD']}@localhost/projet"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class ReviewMAJ(BaseModel):
    contenu: str

class UserCreate(BaseModel):
    mail: str
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

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

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Route pour la page d'accueil (Présentation)
@app.get("/", response_class=HTMLResponse)
def accueil(request: Request):
    return templates.TemplateResponse("site.html", {"request": request})

# Route pour le Dashboard (Recherche et Analyse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


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


@app.post("/user/register", status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    # Vérifier si l'utilisateur existe déjà (par mail ou pseudo)
    existing_user = db.query(Users).filter(
        (Users.mail == user_data.mail) | (Users.username == user_data.username)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=400, 
            detail="Cet email ou ce nom d'utilisateur est déjà utilisé."
        )

    # Créer instance modèle SQLAlchemy
    new_user = Users(
        users_id=str(uuid.uuid4()), # Génère un ID unique
        mail=user_data.mail,
        username=user_data.username,
        password=hash_password(user_data.password)
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": "Utilisateur créé avec succès", "user_id": new_user.users_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création : {e}")


@app.post("/user/login")
def login(user_infos: UserLogin, db: Session = Depends(get_db)):
    # Chercher l'utilisateur
    user = db.query(Users).filter(Users.username == user_infos.username).first()

    # Erreur si on le trouve pas
    if not user or not verify_password(user_infos.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect"
        )

    # On renvoie si c'est bon
    return {
        "message": "Connexion réussie",
        "user": {
            "id": user.users_id,
            "username": user.username,
            "mail": user.mail
        }
    }


# Reviews d'un film précis selon l'identifiant
@app.get("/films/reviews/{film_id}")
def get_reviews(film_id: int, db: Session = Depends(get_db)):
    try:
        # On selectionne le nom de film, date sortie, l'auteur de la review et le contenu
        film = db.query(Film).filter(Film.film_id == film_id).first()

        if not film:
            raise HTTPException(status_code=404, detail="Film introuvable")

        # Récupération avis locaux (base de données)
        reviews_locales = film.reviews or []

        # Récupération avis TMDB
        API_KEY = config["API_KEY"]
        url = f"https://api.themoviedb.org/3/movie/{film_id}/reviews"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)
        reviews_tmdb = response.json().get("results", []) if response.status_code == 200 else []
        tous_les_avis = []
        for r in reviews_locales:
            tous_les_avis.append({"auteur": r.auteur, "contenu": r.contenu, "edited": r.edited})
        
        for r in reviews_tmdb:
            if not any(loc["contenu"] == r["content"] for loc in tous_les_avis):
                tous_les_avis.append({"auteur": r["author"], "contenu": r["content"], "edited": 0})

        # Analyse sur la liste
        sentiments = sentiment_analysis(tous_les_avis)

        return {
            "data": tous_les_avis,
            "Sentiments": sentiments
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")


# Interroger l'API TMDB
def verification_tmdb(nom: str, db: Session):

    #Requête GET sur l'API TMDB
    API_KEY = config["API_KEY"]
    endpoint = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": nom}
    response = requests.get(endpoint, params=params)

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

def sentiment_analysis(reviews):
    sentiments = []
    # 0=Très mauvais, 1=Mauvais, 2=Mitigé, 3=Positif, 4=Très positif
    for i, review in enumerate(reviews):
        content = review.get("contenu") or review.get("content") or ""
        
        if not content:
            print(f"Avis {i}: Contenu vide, on saute.")
            sentiments.append(["Inconnu", 0.0])
            continue

        try:
            tokens = tokenizer.encode(content[:512], return_tensors='pt')
            with torch.no_grad():
                outputs = model(tokens)
            
            logits = outputs.logits
            print(logits)
            # L'index brut de BERT (0 à 4)
            prediction = torch.argmax(logits, dim=1).item()
            
            # Calcul de probabilité (0.0 à 1.0)
            prob = torch.softmax(logits, dim=1)[0][prediction].item()
            
            # Résultat final
            if prediction < 0.2:
                label_final = "Très mauvais"
            elif prediction >= 0.2 and prediction < 0.4:
                label_final = "Mauvais"
            elif prediction >= 0.4 and prediction < 0.6:
                label_final = "Mitigé"
            elif prediction >= 0.6 and prediction < 0.8:
                label_final = "Positif"
            else:
                label_final = "Très positif"
            sentiments.append([label_final, prob])
            
        except Exception as e:
            print(f"Avis {i} | ERREUR : {e}")
            sentiments.append(["Erreur", 0.0])
            
    return sentiments
#Voir dictionnaire pythorch
#AVancer compte rendu
#jeudi 11h le 20 partie sentiments + rendre à l'utilisateur


# Fonctionnement précis BERT a partir du vecteur
# transfert learning (utilisation ia déja entrainé)
# Centrer en 0 les valeurs
# Negation dans l'analyse / Sentiment a la fin qui prime
