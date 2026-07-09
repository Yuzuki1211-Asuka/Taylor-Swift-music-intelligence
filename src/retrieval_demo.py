from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Zero-API retrieval demo.
# Run after build_assets.py: python src/retrieval_demo.py

profiles = pd.read_csv("data/song_profiles.csv")
texts = profiles["profile_text"].fillna("").tolist()

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

def search(query: str, top_k: int = 5):
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, X).ravel()
    idx = scores.argsort()[::-1][:top_k]
    return profiles.iloc[idx].assign(score=scores[idx])[
        ["score", "name", "album_base", "era", "mood_tags", "profile_text"]
    ]

if __name__ == "__main__":
    while True:
        query = input("Ask about songs, mood, album, or audio features > ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        print(search(query, top_k=5).to_string(index=False))
        print()
