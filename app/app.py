from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import dotenv_values
import uuid
import requests
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.models import Film, Review, Users
from app.database import get_db, engine, Base
from app.schemas import ReviewMAJ, ReviewCreate
from app.logic import sentiment_analysis, verification_tmdb, afficher_rapport_terminal, generer_synthese
from app.auth import router as auth_router, get_current_user

# Crée les tables au démarrage si elles n'existent pas encore
Base.metadata.create_all(bind=engine)

limiter = Limiter(key_func=get_remote_address)
templates = Jinja2Templates(directory="templates")
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(auth_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = dotenv_values(".env")

@app.get("/", response_class=HTMLResponse)
def accueil(request: Request):
    return templates.TemplateResponse("site.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/confidentialite", response_class=HTMLResponse)
def confidentialite(request: Request):
    return templates.TemplateResponse("confidentialite.html", {"request": request})

@app.get("/films/liste")
def obtenir_films(nom: str, db: Session = Depends(get_db)):
    films = db.query(Film).filter(Film.titre.ilike(f"%{nom}%")).all()

    if films:
        return {'data': films}
    else:
        films_tmdb = verification_tmdb(nom, db, config['API_KEY'])
        if films_tmdb:
            return {'data': films_tmdb}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Aucun film trouvé dans la base locale ni sur TMDB"
            )

@app.post("/films/reviews/add")
@limiter.limit("2/minute")
def add_review(request: Request, rev: ReviewCreate, db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    film_existe = db.query(Film).filter(Film.film_id == rev.film_id).first()
    if not film_existe:
        raise HTTPException(status_code=404, detail="Film absent de la base")

    new_rev = Review(
        review_id=str(uuid.uuid4()),
        film_id=rev.film_id,
        auteur=current_user.username,
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

@app.put("/films/reviews/update/{id}")
@limiter.limit("2/minute")
def update_review(request: Request, id: str, rev: ReviewMAJ, db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    review = db.query(Review).filter(Review.review_id == id).first()

    if not review:
        raise HTTPException(status_code=404, detail=f"Aucune review trouvée avec l'id {id}")

    if review.auteur != current_user.username:
        raise HTTPException(status_code=403, detail="Vous ne pouvez modifier que vos propres avis")

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
@limiter.limit("2/minute")
def delete_review(request: Request, id: str, db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    review_cible = db.query(Review).filter(Review.review_id == id).first()

    if not review_cible:
        raise HTTPException(status_code=404, detail=f"Aucune review trouvée avec l'id {id}")

    if review_cible.auteur != current_user.username:
        raise HTTPException(status_code=403, detail="Vous ne pouvez supprimer que vos propres avis")

    db.delete(review_cible)
    db.commit()
    return {
        "message": "Review mise à jour avec succès (message supprimé)",
    }

@app.get("/films/reviews/{film_id}")
@limiter.limit("20/minute")
def get_reviews(request: Request, film_id: int, db: Session = Depends(get_db)):
    try:
        film = db.query(Film).filter(Film.film_id == film_id).first()

        if not film:
            raise HTTPException(status_code=404, detail="Film introuvable")

        reviews_locales = film.reviews or []

        API_KEY = config["API_KEY"]
        url = f"https://api.themoviedb.org/3/movie/{film_id}/reviews"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params, timeout=5)
        reviews_tmdb = response.json().get("results", []) if response.status_code == 200 else []
        tous_les_avis = []
        for r in reviews_locales:
            tous_les_avis.append({"review_id": r.review_id, "auteur": r.auteur, "contenu": r.contenu, "edited": r.edited, "rating": r.rating})

        for r in reviews_tmdb:
            if not any(loc["contenu"] == r["content"] for loc in tous_les_avis):
                tous_les_avis.append({"review_id": None, "auteur": r["author"], "contenu": r["content"], "edited": 0})

        sentiments, y_pred, y_true = sentiment_analysis(tous_les_avis)
        afficher_rapport_terminal(y_true, y_pred)
        return {
            "data": tous_les_avis,
            "Sentiments": sentiments,
            "synthese": generer_synthese(sentiments)
        }
    except HTTPException:
        # On laisse passer les erreurs HTTP volontaires (404) sans les masquer en 500
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")
