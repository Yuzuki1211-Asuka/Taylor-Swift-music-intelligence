from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from qwen_local import qwen_chat

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "song_profiles.csv"

_df = None
_vectorizer = None
_matrix = None


def load_index():
    global _df, _vectorizer, _matrix

    if _df is None:
        _df = pd.read_csv(DATA_PATH)

        if "profile_text" not in _df.columns:
            raise ValueError("data/song_profiles.csv 里需要有 profile_text 字段。")

        texts = _df["profile_text"].fillna("").astype(str).tolist()

        _vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000,
        )

        _matrix = _vectorizer.fit_transform(texts)

    return _df, _vectorizer, _matrix


def retrieve(query: str, top_k: int = 5):
    df, vectorizer, matrix = load_index()

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]

    top_indices = scores.argsort()[::-1][:top_k]

    items = []
    for idx in top_indices:
        row = df.iloc[idx]
        items.append({
            "score": float(scores[idx]),
            "metadata": {
                "name": str(row.get("name", "")),
                "album": str(row.get("album", "")),
                "era": str(row.get("era", "")),
                "cluster_name": str(row.get("cluster_name", "")),
            },
            "document": str(row.get("profile_text", "")),
        })

    return items


def answer_with_rag(query: str, top_k: int = 5):
    retrieved = retrieve(query, top_k=top_k)

    context = "\n\n---\n\n".join(
        [f"[{i + 1}] {item['document']}" for i, item in enumerate(retrieved)]
    )

    system_prompt = """
你是一个基于 Taylor Swift Spotify 数据集的音乐数据分析助手。
你只能基于给定的检索上下文回答。
如果上下文不足以回答，请明确说“当前数据不足以判断”。
不要编造不存在的歌曲、专辑、数值或结论。
回答要结构化，优先给出歌曲名、专辑、音频特征依据和推荐理由。
""".strip()

    user_prompt = f"""
用户问题：
{query}

检索上下文：
{context}

请基于检索上下文回答。
""".strip()

    answer = qwen_chat(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=700,
    )

    return {
        "answer": answer,
        "retrieved": retrieved,
    }
