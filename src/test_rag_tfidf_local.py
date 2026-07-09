from rag_tfidf_local import answer_with_rag

query = "根据数据集，推荐几首 acousticness 高、energy 低、valence 低，适合深夜听的 Taylor Swift 歌"

result = answer_with_rag(query, top_k=5)

print("Answer:")
print(result["answer"])

print("\nRetrieved:")
for item in result["retrieved"]:
    print("-", item["metadata"])
    print("  score:", item["score"])
    print("  text:", item["document"][:400].replace("\n", " "))
    print()
