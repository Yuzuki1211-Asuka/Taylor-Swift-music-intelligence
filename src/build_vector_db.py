from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "song_profiles.csv"
DB_DIR = ROOT / "vector_db"

EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "taylor_song_profiles"


def main():
    df = pd.read_csv(DATA_PATH)

    if "profile_text" not in df.columns:
        raise ValueError("data/song_profiles.csv 里需要有 profile_text 字段。")

    texts = df["profile_text"].fillna("").tolist()

    ids = []
    metadatas = []

    for i, row in df.iterrows():
        ids.append(f"song_{i}")
        metadatas.append({
            "name": str(row.get("name", "")),
            "album": str(row.get("album", "")),
            "era": str(row.get("era", "")),
            "cluster_name": str(row.get("cluster_name", "")),
        })

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding song profiles...")
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print("Building Chroma vector DB...")
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Done. Indexed {len(texts)} song profiles into {DB_DIR}")


if __name__ == "__main__":
    main()
