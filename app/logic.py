import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.models import Film
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Chargé une seule fois au démarrage
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def sentiment_analysis(reviews):
    sentiments = []
    y_pred = []
    y_true = []
    # Index BERT: label textuel (0=Très mauvais ... 4=Très positif)
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
            inputs = tokenizer(content, max_length=512, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            prob = torch.softmax(logits, dim=1)[0][prediction].item()
            label_final = labels_map[prediction]

            sentiments.append([label_final, prob])

            if review.get("rating") is not None:
                y_pred.append(prediction)
                y_true.append(review.get("rating"))
                print("Valeur réelle: ", y_true[-1])
                print(f"Valeur prédite: {y_pred[-1]}\n")

        except Exception as e:
            print(f"Avis {i} | ERREUR : {e}")
            sentiments.append(["Erreur", 0.0])

    return sentiments, y_pred, y_true

def verification_tmdb(nom, db, API_KEY):
    endpoint = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": nom}
    response = requests.get(endpoint, params=params, timeout=5)

    if response.status_code != 200:
        return []

    results = response.json().get("results", [])
    if not results:
        return []

    # Seulement les films contenant le mot cherché
    films = [
        {"id": r["id"], "titre": r["title"], "date_sortie": r.get("release_date", None)}
        for r in results
        if nom.lower() in r["title"].lower()
    ]
    for film in films:
        new_film = Film(film_id=film["id"], titre=film["titre"], date_sortie=film["date_sortie"])
        db.merge(new_film)
    db.commit()
    return films

def generer_synthese(sentiments: list) -> dict:
    positifs = sum(1 for label, _ in sentiments if label in ("Positif", "Très positif"))
    negatifs = sum(1 for label, _ in sentiments if label in ("Mauvais", "Très mauvais"))
    neutres  = sum(1 for label, _ in sentiments if label == "Mitigé")
    total = positifs + negatifs + neutres

    if total == 0:
        verdict = "Aucun avis analysé"
    else:
        ratio = positifs / total
        if ratio >= 0.7:
            verdict = "Très bien reçu"
        elif ratio >= 0.5:
            verdict = "Bien reçu"
        elif negatifs / total >= 0.7:
            verdict = "Très mal reçu"
        elif negatifs / total >= 0.5:
            verdict = "Mal reçu"
        else:
            verdict = "Avis mitigés"

    return {"verdict": verdict, "positifs": positifs, "negatifs": negatifs, "neutres": neutres, "total": total}


def afficher_rapport_terminal(y_true, y_pred):
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return

    print("Valeurs réelles: ", y_true, "\n Valeurs prédites: ", y_pred)
    labels = ["Très mauvais", "Mauvais", "Mitigé", "Positif", "Très positif"]

    print("\n")
    print("Analyse IA sur des valeurs préremplies")
    print("\n")

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.2%}")

    print("\n Report classification")
    print(classification_report(y_true, y_pred, target_names=labels, labels=[0,1,2,3,4], zero_division=0))

    print("Matrice de confusion :")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    print(cm)

    print("\nLignes = vraies notes, les colonnes = prédictions")
