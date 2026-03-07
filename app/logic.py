import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Analyse de sentiments
from passlib.context import CryptContext
from app.models import Film

# Initialisation BERT
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Initialisation Sécurité
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

# Analyse de sentiments
def sentiment_analysis(reviews):
    sentiments = []
    # 0=Très mauvais, 1=Mauvais, 2=Mitigé, 3=Positif, 4=Très positif
    # Mapping pour transformer l'index de BERT en label textuel
    labels_map = {
        0: "Très mauvais",
        1: "Mauvais",
        2: "Mitigé",
        3: "Positif",
        4: "Très positif"
    }

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
            # L'index BERT (0 à 4)
            prediction = torch.argmax(logits, dim=1).item()
            
            # Calcul probabilité (0.0 à 1.0)
            prob = torch.softmax(logits, dim=1)[0][prediction].item()
            
            # Résultat final
            print(prob)
            
            # On récupère le label 
            label_final = labels_map[prediction]
            
            sentiments.append([label_final, prob])
            
        except Exception as e:
            print(f"Avis {i} | ERREUR : {e}")
            sentiments.append(["Erreur", 0.0])
            
    return sentiments

# Vérifie si le film pas trouvé est dans la base TMDB est l'ajoute à la base de données 
def verification_tmdb(nom, db, API_KEY):
    #Requête GET sur l'API TMDB
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
        {"id": r["id"], "titre": r["title"], "date_sortie": r.get("release_date", None)}
        for r in results
        if nom.lower() in r["title"].lower()
    ]
    # On insère chaque film dans la table si l'id est unique
    for film in films:
        new_film = Film(film_id=film["id"], titre=film["titre"], date_sortie=film["date_sortie"])
        db.merge(new_film)
    db.commit()
    return films

#Voir dictionnaire pythorch
#AVancer compte rendu
#jeudi 11h le 20 partie sentiments + rendre à l'utilisateur

# Fonctionnement précis BERT a partir du vecteur
# transfert learning (utilisation ia déja entrainé)
# Centrer en 0 les valeurs
# Negation dans l'analyse / Sentiment a la fin qui prime