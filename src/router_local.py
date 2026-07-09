from rag_hybrid_local import answer_with_rag
from analytics_local import answer_analytics


def get_route_explanation(query: str):
    q = query.lower()

    analytics_keywords = [
        "哪个", "哪些", "最高", "最低", "平均", "均值", "中位数",
        "比较", "对比", "差异", "排名", "top", "highest", "lowest",
        "mean", "average", "median", "最多", "最少",
    ]

    recommendation_keywords = [
        "推荐", "适合", "听", "playlist", "深夜", "运动", "emo",
        "安静", "开心", "伤感", "workout", "recommend",
    ]

    version_keywords = [
        "deluxe", "live", "taylor's version", "anthology",
        "豪华版", "现场", "重录", "版本",
    ]

    feature_keywords = [
        "valence", "energy", "acousticness", "danceability",
        "popularity", "tempo", "流行度", "能量", "情绪",
    ]

    album_keywords = [
        "ttpd", "folklore", "evermore", "midnights", "lover",
        "reputation", "red", "fearless", "speak now", "1989",
        "debut", "taylor swift",
    ]

    matched_recommendation = [k for k in recommendation_keywords if k in q]
    matched_analytics = [k for k in analytics_keywords if k in q]
    matched_version = [k for k in version_keywords if k in q]
    matched_features = [k for k in feature_keywords if k in q]
    matched_albums = [k for k in album_keywords if k in q]

    signals = {
        "recommendation": matched_recommendation,
        "analytics": matched_analytics,
        "version": matched_version,
        "features": matched_features,
        "albums": matched_albums,
    }

    if matched_recommendation:
        return {
            "intent": "rag",
            "reason": f"问题中包含推荐/场景/情绪相关信号：{', '.join(matched_recommendation)}",
            "pipeline": "Hybrid structured retrieval → Version-aware deduplication → Local Qwen explanation",
            "signals": signals,
        }

    if matched_analytics:
        return {
            "intent": "analytics",
            "reason": f"问题中包含统计/排名/对比相关信号：{', '.join(matched_analytics)}",
            "pipeline": "Pandas computation → Result table → Local Qwen explanation",
            "signals": signals,
        }

    if matched_version:
        return {
            "intent": "analytics",
            "reason": f"问题中包含版本分析相关信号：{', '.join(matched_version)}",
            "pipeline": "Version-aware pandas analysis → Result table → Local Qwen explanation",
            "signals": signals,
        }

    return {
        "intent": "rag",
        "reason": "未检测到明确统计/排名/对比信号，默认作为推荐或语义检索问题处理。",
        "pipeline": "Hybrid structured retrieval → Local Qwen explanation",
        "signals": signals,
    }


def build_reliability(intent: str, route: dict):
    signals = route.get("signals", {})

    if intent == "analytics":
        return {
            "level": "High",
            "reason": "This answer is based on deterministic pandas computation over the local dataset.",
        }

    feature_count = len(signals.get("features", []))
    album_count = len(signals.get("albums", []))
    version_count = len(signals.get("version", []))

    if feature_count > 0 and album_count > 0:
        return {
            "level": "Medium-High",
            "reason": "The recommendation is grounded by explicit album and audio-feature constraints.",
        }

    if feature_count > 0 or album_count > 0 or version_count > 0:
        return {
            "level": "Medium",
            "reason": "The recommendation uses partial structured constraints and retrieved song evidence.",
        }

    return {
        "level": "Medium",
        "reason": "The recommendation is based on semantic intent and heuristic audio-feature matching.",
    }


def build_trace(query: str, intent: str, route: dict, top_k: int):
    if intent == "analytics":
        return [
            {
                "step": "1. Query Input",
                "detail": query,
            },
            {
                "step": "2. Intent Detection",
                "detail": "Detected as analytics because the query asks for statistics, ranking, comparison, or aggregation.",
            },
            {
                "step": "3. Data Processing",
                "detail": "The system uses pandas to compute results directly from the local Spotify dataset.",
            },
            {
                "step": "4. Evidence Source",
                "detail": "The answer is grounded in the computed result table, not generated from memory.",
            },
            {
                "step": "5. Language Generation",
                "detail": "Local Qwen2.5-3B-Instruct explains the computed results in natural language.",
            },
        ]

    return [
        {
            "step": "1. Query Input",
            "detail": query,
        },
        {
            "step": "2. Intent Detection",
            "detail": "Detected as recommendation / mood / scene query and routed to Hybrid RAG.",
        },
        {
            "step": "3. Structured Retrieval",
            "detail": f"The system applies album, version, mood, and audio-feature signals, then retrieves top-{top_k} candidates.",
        },
        {
            "step": "4. Version-aware Deduplication",
            "detail": "Canonical song names and version types are handled to avoid repeated duplicate versions unless the query asks for versions.",
        },
        {
            "step": "5. Language Generation",
            "detail": "Local Qwen2.5-3B-Instruct explains the retrieved evidence instead of answering from memory.",
        },
    ]


def detect_intent(query: str):
    return get_route_explanation(query)["intent"]


def answer_auto(query: str, top_k: int = 8):
    route = get_route_explanation(query)
    intent = route["intent"]

    reliability = build_reliability(intent, route)
    trace = build_trace(query, intent, route, top_k)

    if intent == "analytics":
        result = answer_analytics(query)
        return {
            "intent": "analytics",
            "answer": result["answer"],
            "table": result["table"],
            "route_reason": route["reason"],
            "pipeline": route["pipeline"],
            "signals": route["signals"],
            "reliability": reliability,
            "trace": trace,
            "raw": result,
        }

    result = answer_with_rag(query, top_k=top_k)
    return {
        "intent": "rag",
        "answer": result["answer"],
        "retrieved": result["retrieved"],
        "route_reason": route["reason"],
        "pipeline": route["pipeline"],
        "signals": route["signals"],
        "reliability": reliability,
        "trace": trace,
        "raw": result,
    }
