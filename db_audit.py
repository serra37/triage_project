import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from database.chroma_client import get_chroma_client
from collections import Counter

db = get_chroma_client()
all_data = db.get()

diseases = [m.get("disease", "unknown").lower() for m in all_data["metadatas"]]
print(f"Toplam chunk: {len(diseases)}\n")

counter = Counter(diseases)
print("=== Hastalık bazında chunk sayısı ===")
for disease, count in counter.most_common(60):
    print(f"  {count:4d}  {disease}")

print("\n=== Başarısız hastalıklar DB'de var mı? ===")
targets = ["migraine", "asthma", "appendicitis", "acid reflux",
           "arthritis", "allergy", "viral fever", "heart attack"]

for disease in targets:
    results = db.similarity_search(disease, k=3)
    print(f"\n--- {disease} ---")
    for r in results:
        print(f"  disease: {r.metadata.get('disease')} | {r.page_content[:80]}")