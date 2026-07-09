from rag_hybrid_local import retrieve
from analytics_local import build_analysis

def check_rag_ttpd_low_valence():
    q = "从 TTPD 里推荐几首 valence 低、energy 低的歌"
    items = retrieve(q, top_k=5)

    checks = []
    for item in items:
        m = item["metadata"]
        checks.append({
            "name": m["name"],
            "album": m["album"],
            "is_ttpd": "TORTURED POETS" in m["album"].upper(),
            "energy_ok": m["energy"] <= 0.55,
            "valence_ok": m["valence"] <= 0.45,
        })

    return checks


def check_rag_deluxe():
    q = "推荐几首 deluxe 版本里 valence 低的歌"
    items = retrieve(q, top_k=5)

    return [
        {
            "name": item["metadata"]["name"],
            "version_type": item["metadata"]["version_type"],
            "is_deluxe": item["metadata"]["version_type"] == "deluxe",
        }
        for item in items
    ]


def check_analytics_popularity_dedup():
    q = "哪些歌 popularity 最高？"
    result = build_analysis(q)
    table = result["table"]

    duplicated = table["canonical_name"].duplicated().any() if "canonical_name" in table.columns else None

    return {
        "has_canonical_name": "canonical_name" in table.columns,
        "has_duplicates": bool(duplicated),
        "top_song": table.iloc[0]["name"] if len(table) else None,
        "top_popularity": table.iloc[0]["popularity"] if len(table) else None,
    }


def check_analytics_album_valence():
    q = "哪个专辑平均 valence 最低？"
    result = build_analysis(q)
    table = result["table"]

    return {
        "analysis_type": result["analysis_type"],
        "feature": result["feature"],
        "direction": result["direction"],
        "top_album": table.iloc[0].iloc[0] if len(table) else None,
        "top_mean_value": table.iloc[0]["mean_value"] if "mean_value" in table.columns and len(table) else None,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("RAG: TTPD low valence / low energy")
    for row in check_rag_ttpd_low_valence():
        print(row)

    print("\n" + "=" * 80)
    print("RAG: deluxe filter")
    for row in check_rag_deluxe():
        print(row)

    print("\n" + "=" * 80)
    print("Analytics: popularity dedup")
    print(check_analytics_popularity_dedup())

    print("\n" + "=" * 80)
    print("Analytics: album valence")
    print(check_analytics_album_valence())
