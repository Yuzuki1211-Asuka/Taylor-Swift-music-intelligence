from pathlib import Path
import re
import pandas as pd
from qwen_local import qwen_chat

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "taylor_swift_spotify_clean.csv"

FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "popularity",
]

ALBUM_ALIASES = {
    "ttpd": "the tortured poets department",
    "the tortured poets department": "the tortured poets department",
    "folklore": "folklore",
    "evermore": "evermore",
    "midnights": "midnights",
    "lover": "lover",
    "reputation": "reputation",
    "1989": "1989",
    "red": "red",
    "speak now": "speak now",
    "fearless": "fearless",
}


def normalize_song_name(name):
    """
    用于 song-level analytics 去重。
    把同一首歌的不同版本归一到 canonical song name。
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


def should_keep_versions_separate(query: str):
    q = query.lower()

    keywords = [
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
        "对比",
        "比较",
    ]

    return any(k in q for k in keywords)


def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["name", "album", "album_base", "era"]:
        if col not in df.columns:
            df[col] = ""

    if "version_type" not in df.columns:
        df["version_type"] = df.apply(infer_version_type, axis=1)

    df["canonical_name"] = df["name"].apply(normalize_song_name)
    df["version_priority"] = df["version_type"].apply(version_priority)

    return df


def detect_feature(query: str):
    q = query.lower()

    feature_aliases = {
        "acousticness": ["acousticness", "acoustic", "原声", "安静"],
        "danceability": ["danceability", "danceable", "dance", "跳舞", "律动"],
        "energy": ["energy", "高能量", "低能量", "能量"],
        "valence": ["valence", "忧郁", "开心", "愉悦", "情绪"],
        "tempo": ["tempo", "bpm", "节奏", "速度"],
        "loudness": ["loudness", "响度"],
        "popularity": ["popularity", "popular", "热门", "流行度"],
    }

    for feature, aliases in feature_aliases.items():
        if any(a in q for a in aliases):
            return feature

    return "valence"


def detect_direction(query: str):
    q = query.lower()

    low_words = ["最低", "低", "least", "lowest", "忧郁", "安静"]
    high_words = ["最高", "高", "most", "highest", "热门", "强"]

    if any(w in q for w in low_words):
        return "low"

    if any(w in q for w in high_words):
        return "high"

    return "high"


def detect_group(query: str):
    q = query.lower()

    if "era" in q or "时期" in q or "阶段" in q:
        return "era"

    if "专辑" in q or "album" in q:
        return "album_base"

    return None


def detect_albums(query: str):
    q = query.lower()
    found = []

    for key, canonical in ALBUM_ALIASES.items():
        if key in q:
            found.append(canonical)

    # 去重保序
    result = []
    for x in found:
        if x not in result:
            result.append(x)

    return result


def filter_album(df, canonical_album):
    album_col = df["album"].fillna("").str.lower()
    base_col = df["album_base"].fillna("").str.lower()

    if canonical_album == "the tortured poets department":
        return df[
            album_col.str.contains("tortured poets", regex=False)
            | base_col.str.contains("tortured poets", regex=False)
        ]

    return df[
        album_col.str.contains(canonical_album, regex=False)
        | base_col.str.contains(canonical_album, regex=False)
    ]


def make_album_comparison(df, albums):
    rows = []

    for album in albums:
        sub = filter_album(df, album)

        if sub.empty:
            continue

        row = {
            "album": album,
            "song_count": len(sub),
        }

        for f in FEATURES:
            if f in sub.columns:
                row[f"mean_{f}"] = round(float(sub[f].mean()), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def make_group_analysis(df, group_col, feature, direction):
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            song_count=("name", "count"),
            mean_value=(feature, "mean"),
            median_value=(feature, "median"),
        )
        .reset_index()
    )

    grouped["mean_value"] = grouped["mean_value"].round(4)
    grouped["median_value"] = grouped["median_value"].round(4)

    ascending = direction == "low"
    grouped = grouped.sort_values("mean_value", ascending=ascending)

    return grouped.head(10)


def make_song_ranking(df, feature, direction, keep_versions=False):
    ascending = direction == "low"

    df = df.copy()

    # 先按目标特征排序；同分时用 version_priority 辅助选择更常规版本
    df = df.sort_values(
        [feature, "version_priority"],
        ascending=[ascending, True],
    )

    # 默认 song-level ranking 去重，避免同一首歌不同专辑版本重复刷屏
    if not keep_versions:
        df = df.drop_duplicates(subset=["canonical_name"], keep="first")

    cols = [
        "name",
        "canonical_name",
        "album",
        "album_base",
        "version_type",
        "era",
        "acousticness",
        "energy",
        "valence",
        "danceability",
        "popularity",
    ]

    available = [c for c in cols if c in df.columns]

    result = df[available].head(15).copy()

    for c in FEATURES:
        if c in result.columns:
            result[c] = result[c].round(4)

    return result


def build_analysis(query: str):
    df = load_data()

    feature = detect_feature(query)
    direction = detect_direction(query)
    group_col = detect_group(query)
    albums = detect_albums(query)

    if len(albums) >= 2:
        table = make_album_comparison(df, albums)
        analysis_type = "album_comparison"

    elif len(albums) == 1:
        sub = filter_album(df, albums[0])
        table = make_song_ranking(
            sub,
            feature,
            direction,
            keep_versions=should_keep_versions_separate(query),
        )
        analysis_type = "single_album_song_ranking"

    elif group_col is not None:
        table = make_group_analysis(df, group_col, feature, direction)
        analysis_type = "group_analysis"

    else:
        table = make_song_ranking(
            df,
            feature,
            direction,
            keep_versions=should_keep_versions_separate(query),
        )
        analysis_type = "song_ranking"

    return {
        "analysis_type": analysis_type,
        "feature": feature,
        "direction": direction,
        "table": table,
    }


def answer_analytics(query: str):
    result = build_analysis(query)
    table = result["table"]

    table_text = table.to_string(index=False)

    system_prompt = """
你是一个严谨的音乐数据分析助手。
你必须基于 pandas 计算结果回答，不要编造表格之外的数值。
回答要说明：
1. 计算对象
2. 排名或对比结果
3. 关键数值
4. 简短解释
如果结果不足，请说明数据不足。
""".strip()

    user_prompt = f"""
用户问题：
{query}

分析类型：
{result["analysis_type"]}

特征：
{result["feature"]}

排序方向：
{result["direction"]}

pandas 计算结果：
{table_text}

请基于这些结果回答。
""".strip()

    answer = qwen_chat(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=700,
    )

    return {
        "answer": answer,
        "table": table,
        "analysis_type": result["analysis_type"],
        "feature": result["feature"],
        "direction": result["direction"],
    }
