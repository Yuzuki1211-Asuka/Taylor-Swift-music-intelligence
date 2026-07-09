from analytics_local import answer_analytics

queries = [
    "哪个专辑平均 valence 最低？",
    "哪个 era 的 energy 最高？",
    "TTPD 和 folklore 的 acousticness 有什么差异？",
    "从 TTPD 里找 valence 最低的歌",
    "哪些歌 popularity 最高？",
]

for q in queries:
    print("=" * 100)
    print("Query:", q)
    result = answer_analytics(q)

    print("\nAnswer:")
    print(result["answer"])

    print("\nTable:")
    print(result["table"].to_string(index=False))
    print()
