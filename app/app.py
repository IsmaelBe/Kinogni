from fastapi import FastAPI, HTTPException, status, Request, Depends # Application FastAPI, Gestion erreurs HTTP
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse # Renvoyer des fichiers HTML
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import dotenv_values
import uuid
import requests

# Imports locaux
from app.models import Film, Review, Users
from app.database import get_db
from app.schemas import ReviewMAJ, ReviewCreate, UserCreate, UserLogin
from app.logic import sentiment_analysis, verification_tmdb, hash_password, verify_password, afficher_rapport_terminal

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

# Import variables environnement et connexion base Postgres
config = dotenv_values(".env")

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
        films_tmdb = verification_tmdb(nom, db, config['API_KEY'])
        if films_tmdb:
            # Le film est sur l'API TMDB
            return {'data': films_tmdb}
        else:
            # Erreur 404
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Aucun film trouvé dans la base locale ni sur TMDB"
            )

# Ajoute un avis sur un film précis
@app.post("/films/reviews/add")
def add_review(rev: ReviewCreate, db: Session = Depends(get_db)):
    film_existe = db.query(Film).filter(Film.film_id == rev.film_id).first()
    if not film_existe:
        raise HTTPException(status_code=404, detail="Film absent de la base")

    new_rev = Review(
        review_id=str(uuid.uuid4()),
        film_id=rev.film_id,
        auteur=rev.username, # On utilise le pseudo comme identifiant auteur
        contenu=rev.contenu,
        edited=0
    )
    try:
        db.add(new_rev)
        db.commit()
        db.refresh(new_rev)
        return {"message": "Avis ajouté"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Met à jour un avis existant
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

# Supprime un avis existant
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

# Connexion de l'utilisateur
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
            # On utilise 'auteur' pour comparer avec le username du localStorage
            tous_les_avis.append({"review_id": r.review_id, "auteur": r.auteur, "contenu": r.contenu, "edited": r.edited, "rating": r.rating})
        
        for r in reviews_tmdb:
            if not any(loc["contenu"] == r["content"] for loc in tous_les_avis):
                tous_les_avis.append({"review_id": None, "auteur": r["author"], "contenu": r["content"], "edited": 0})

        # Analyse sur la liste
        sentiments, y_pred, y_true = sentiment_analysis(tous_les_avis)
        afficher_rapport_terminal(y_true,y_pred)
        return {
            "data": tous_les_avis,
            "Sentiments": sentiments
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")
 