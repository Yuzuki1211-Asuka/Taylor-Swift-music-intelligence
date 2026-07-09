from router_local import answer_auto

queries = [
    "从 TTPD 里推荐几首 valence 低、energy 低的歌",
    "哪个专辑平均 valence 最低？",
    "哪些歌 popularity 最高？",
    "推荐几首适合运动的 high energy 歌",
    "TTPD 和 folklore 的 acousticness 有什么差异？",
]

for q in queries:
    print("=" * 100)
    print("Query:", q)

    result = answer_auto(q)

    print("Detected intent:", result["intent"])
    print("Answer:")
    print(result["answer"])

    if result["intent"] == "analytics":
        print("\nTable:")
        print(result["table"].to_string(index=False))
    else:
        print("\nRetrieved:")
        for item in result["retrieved"]:
            m = item["metadata"]
            print("-", m.get("name"), "|", m.get("album"), "|", m.get("version_type"))

    print()
