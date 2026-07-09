import pandas as pd
from router_local import detect_intent, answer_auto

df = pd.read_csv("eval_questions.csv")

rows = []

for _, row in df.iterrows():
    q = row["question"]
    expected = row["expected_intent"]

    detected = detect_intent(q)
    passed = detected == expected

    rows.append({
        "question": q,
        "expected_intent": expected,
        "detected_intent": detected,
        "pass": passed,
    })

result = pd.DataFrame(rows)

print(result.to_string(index=False))
print()
print("Router accuracy:", result["pass"].mean())
