from pydantic import BaseModel

# Ensemble de classe pour les requêtes POST/PUT
class ReviewMAJ(BaseModel):
    contenu: str

class ReviewCreate(BaseModel):
    film_id: int
    user_id: str
    username: str
    contenu: str

class UserCreate(BaseModel):
    mail: str
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str