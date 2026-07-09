from pathlib import Path
import re
import pandas as pd
from qwen_local import qwen_chat

ROOT = Path(__file__).resolve().parents[1]

PROFILE_PATH = ROOT / "data" / "song_profiles.csv"
CLEAN_PATH = ROOT / "data" / "taylor_swift_spotify_clean.csv"

_df = None


def contains_any(text, keywords):
    text = str(text).lower()
    return any(k.lower() in text for k in keywords)


def normalize_song_name(name):
    """
    用于普通推荐去重。
    同一首歌出现在 standard / deluxe / live / acoustic / TV 等版本中时，
    默认归一成同一个 canonical_name。
    """
    name = str(name).lower().strip()

    remove_patterns = [
        r"\(taylor's version\)",
        r"\(from the vault\)",
        r"\(deluxe.*?\)",
        r"\(acoustic.*?\)",
        r"\(live.*?\)",
        r"\(piano.*?\)",
        r"\(remix.*?\)",
        r"\(radio edit\)",
        r"\s+-\s+live.*$",
        r"\s+-\s+acoustic.*$",
        r"\s+-\s+piano.*$",
        r"\s+-\s+remix.*$",
    ]

    for pattern in remove_patterns:
        name = re.sub(pattern, "", name)

    name = re.sub(r"\s+", " ", name)
    name = name.strip(" -")
    return name


def infer_version_type(row):
    name = str(row.get("name", "")).lower()
    album = str(row.get("album", "")).lower()
    combined = name + " " + album

    if "taylor's version" in combined:
        return "taylor_version"
    if "from the vault" in combined:
        return "vault"
    if "live" in combined:
        return "live"
    if "acoustic" in combined:
        return "acoustic"
    if "remix" in combined:
        return "remix"
    if "anthology" in combined:
        return "anthology"
    if "deluxe" in combined:
        return "deluxe"

    return "standard"


def version_priority(version_type):
    """
    数字越小，默认推荐时越优先保留。
    """
    priority = {
        "standard": 0,
        "taylor_version": 1,
        "deluxe": 2,
        "anthology": 3,
        "vault": 4,
        "live": 5,
        "acoustic": 6,
        "remix": 7,
    }
    return priority.get(str(version_type), 9)


def extract_number_from_profile(text, field):
    if not isinstance(text, str):
        return None

    pattern = rf"{field}\s*=\s*([0-9.]+)"
    match = re.search(pattern, text)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None


def build_profile_text(row):
    return (
        f"Song: {row.get('name', '')}\n"
        f"Album: {row.get('album', '')}\n"
        f"Era: {row.get('era', '')}\n"
        f"Version type: {row.get('version_type', '')}\n"
        f"Popularity: {row.get('popularity', '')}\n"
        f"Audio profile: acousticness={row.get('acousticness', '')}, "
        f"danceability={row.get('danceability', '')}, "
        f"energy={row.get('energy', '')}, "
        f"valence={row.get('valence', '')}, "
        f"tempo={row.get('tempo', '')}, "
        f"loudness={row.get('loudness', '')}\n"
        f"Mood tags: {row.get('mood_tags', '')}\n"
        f"Cluster: {row.get('cluster_name', '')}"
    )


def load_data():
    global _df

    if _df is not None:
        return _df.copy()

    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)
    elif PROFILE_PATH.exists():
        df = pd.read_csv(PROFILE_PATH)
    else:
        raise FileNotFoundError("没有找到 data/song_profiles.csv 或 data/taylor_swift_spotify_clean.csv")

    if PROFILE_PATH.exists():
        profiles = pd.read_csv(PROFILE_PATH)

        if "profile_text" in profiles.columns:
            keep_cols = [c for c in ["name", "album", "profile_text"] if c in profiles.columns]

            if (
                "name" in df.columns
                and "album" in df.columns
                and "name" in profiles.columns
                and "album" in profiles.columns
            ):
                df = df.merge(
                    profiles[keep_cols],
                    on=["name", "album"],
                    how="left",
                    suffixes=("", "_profile"),
                )

                if "profile_text_profile" in df.columns and "profile_text" not in df.columns:
                    df["profile_text"] = df["profile_text_profile"]

            elif "profile_text" not in df.columns:
                df["profile_text"] = profiles["profile_text"]

    if "profile_text" not in df.columns:
        df["profile_text"] = ""

    numeric_cols = [
        "acousticness",
        "energy",
        "valence",
        "danceability",
        "popularity",
        "tempo",
        "loudness",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = df["profile_text"].apply(lambda x: extract_number_from_profile(x, col))

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["name", "album", "album_base", "era", "cluster_name", "mood_tags"]:
        if col not in df.columns:
            df[col] = ""

    if "version_type" not in df.columns:
        df["version_type"] = df.apply(infer_version_type, axis=1)

    df["canonical_name"] = df["name"].apply(normalize_song_name)
    df["version_priority"] = df["version_type"].apply(version_priority)

    # 更新 profile_text，使上下文里也包含版本类型
    empty_profile = df["profile_text"].fillna("").str.len() == 0
    df.loc[empty_profile, "profile_text"] = df[empty_profile].apply(build_profile_text, axis=1)

    _df = df
    return _df.copy()


def apply_filters(df: pd.DataFrame, query: str):
    q = query.lower()

    if "taylor's version" in q or "重录" in q:
        df = df[df["version_type"] == "taylor_version"]

    if "deluxe" in q or "豪华版" in q:
        df = df[df["version_type"] == "deluxe"]

    if "live" in q or "现场" in q:
        df = df[df["version_type"] == "live"]

    if "acoustic version" in q or "原声版" in q:
        df = df[df["version_type"] == "acoustic"]

    # 不要把 "Taylor Swift 歌" 误判成首专过滤。
    albums = [
        "folklore",
        "evermore",
        "midnights",
        "red",
        "1989",
        "reputation",
        "lover",
        "speak now",
        "fearless",
        "the tortured poets department",
        "ttpd",
    ]

    for album in albums:
        if album in q:
            target = "the tortured poets department" if album == "ttpd" else album

            album_col = df["album"].fillna("").str.lower()
            base_col = df["album_base"].fillna("").str.lower()

            mask = album_col.str.contains(target, regex=False) | base_col.str.contains(target, regex=False)

            if mask.sum() > 0:
                df = df[mask]

    return df


def get_numeric_score(df: pd.DataFrame, query: str):
    q = query.lower()
    score = pd.Series(0.0, index=df.index)

    if contains_any(q, ["acoustic", "acousticness", "原声", "安静", "深夜", "night"]):
        score += 0.30 * df["acousticness"]

    if contains_any(q, ["low energy", "low-energy", "energy 低", "energy低", "低能量", "平静", "calm", "quiet", "深夜"]):
        score += 0.30 * (1 - df["energy"])

    if contains_any(q, ["low valence", "valence 低", "valence低", "melancholic", "sad", "忧郁", "难过", "伤感", "emo", "深夜"]):
        score += 0.50 * (1 - df["valence"])

    if contains_any(q, ["high energy", "high-energy", "高能量", "运动", "workout", "energetic"]):
        score += 0.35 * df["energy"]

    if contains_any(q, ["dance", "danceable", "跳舞", "律动"]):
        score += 0.30 * df["danceability"]

    if contains_any(q, ["popular", "popularity", "热门", "流行"]):
        max_pop = df["popularity"].max()
        if max_pop > 0:
            score += 0.15 * (df["popularity"] / max_pop)

    if score.sum() == 0:
        max_pop = df["popularity"].max()
        if max_pop > 0:
            score += df["popularity"] / max_pop

    return score


def should_keep_versions_separate(query: str):
    q = query.lower()

    return contains_any(
        q,
        [
            "version",
            "taylor's version",
            "deluxe",
            "live",
            "acoustic version",
            "remix",
            "vault",
            "anthology",
            "版本",
            "豪华版",
            "现场",
            "重录",
            "原声版",
            "混音",
        ],
    )


def retrieve(query: str, top_k: int = 5):
    df = load_data()
    df = apply_filters(df, query)

    if df.empty:
        return []

    df = df.assign(retrieval_score=get_numeric_score(df, query))

    q = query.lower()

    # 明确的数值条件做硬过滤
    if contains_any(q, ["low energy", "low-energy", "energy 低", "energy低", "低能量", "深夜", "安静"]):
        df = df[df["energy"] <= 0.55]

    if contains_any(q, ["low valence", "valence 低", "valence低", "melancholic", "sad", "忧郁", "伤感", "emo", "深夜"]):
        df = df[df["valence"] <= 0.45]

    if contains_any(q, ["acoustic", "acousticness", "原声"]):
        df = df[df["acousticness"] >= 0.15]

    # 过滤太严格时回退到排序
    if df.empty:
        df = load_data()
        df = apply_filters(df, query)
        df = df.assign(retrieval_score=get_numeric_score(df, query))

    # 默认推荐：同一首歌多版本去重。
    # 版本问题：Deluxe / Live / TV / Acoustic 等保留多个版本。
    keep_versions = should_keep_versions_separate(query)

    if keep_versions:
        df = df.sort_values("retrieval_score", ascending=False)
    else:
        # 对普通推荐，给标准版/TV 轻微优先级，避免 Deluxe/Live 重复刷屏
        df = df.copy()
        df["final_score"] = df["retrieval_score"] - df["version_priority"] * 0.003

        df = df.sort_values(
            ["canonical_name", "final_score"],
            ascending=[True, False],
        )

        df = df.drop_duplicates(subset=["canonical_name"], keep="first")

        df = df.sort_values("final_score", ascending=False)

    df = df.head(top_k)

    items = []
    for _, row in df.iterrows():
        doc = str(row.get("profile_text", ""))

        if "Version type:" not in doc:
            doc += f"\nVersion type: {row.get('version_type', '')}"

        items.append({
            "score": float(row.get("retrieval_score", 0)),
            "metadata": {
                "name": str(row.get("name", "")),
                "canonical_name": str(row.get("canonical_name", "")),
                "album": str(row.get("album", "")),
                "album_base": str(row.get("album_base", "")),
                "era": str(row.get("era", "")),
                "version_type": str(row.get("version_type", "")),
                "cluster_name": str(row.get("cluster_name", "")),
                "acousticness": float(row.get("acousticness", 0)),
                "energy": float(row.get("energy", 0)),
                "valence": float(row.get("valence", 0)),
                "danceability": float(row.get("danceability", 0)),
                "popularity": float(row.get("popularity", 0)),
            },
            "document": doc,
        })

    return items


def answer_with_rag(query: str, top_k: int = 5):
    retrieved = retrieve(query, top_k=top_k)

    if not retrieved:
        return {
            "answer": "当前数据不足以判断。",
            "retrieved": [],
        }

    context = "\n\n---\n\n".join(
        [f"[{i + 1}] {item['document']}" for i, item in enumerate(retrieved)]
    )

    system_prompt = """
你是一个基于 Taylor Swift Spotify 数据集的音乐数据分析助手。
你只能基于给定的检索上下文回答。
不要编造不存在的歌曲、专辑、数值或结论。
回答要结构化。
如果是推荐问题，需要列出歌曲名、专辑、版本类型、关键音频特征和推荐理由。
如果某首歌不完全符合条件，也要明确说明。
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
