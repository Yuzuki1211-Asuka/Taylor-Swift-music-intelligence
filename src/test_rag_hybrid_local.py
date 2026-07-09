from rag_hybrid_local import answer_with_rag

queries = [
    "根据数据集，推荐几首 acousticness 高、energy 低、valence 低，适合深夜听的 Taylor Swift 歌",
    "根据数据集，从 TTPD 里推荐几首 valence 低、energy 低、适合深夜听的歌",
    "根据数据集，推荐几首 deluxe 版本里 valence 低的歌",
]

for query in queries:
    print("=" * 100)
    print("Query:", query)

    result = answer_with_rag(query, top_k=8)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved:")
    for item in result["retrieved"]:
        m = item["metadata"]
        print(
            f"- {m['name']} | {m['album']} | "
            f"version={m['version_type']} | "
            f"canonical={m['canonical_name']} | "
            f"acousticness={m['acousticness']:.3f}, "
            f"energy={m['energy']:.3f}, "
            f"valence={m['valence']:.3f}, "
            f"score={item['score']:.4f}"
        )
    print()
