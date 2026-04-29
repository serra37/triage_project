"""
Retrieval Başarı Ölçüm Scripti
- ChromaDB'den otomatik test seti oluşturur
- Precision@3, Hit Rate ve MRR hesaplar
"""

import os
import sys
import json
import random
import time
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from database.chroma_client import get_chroma_client, search_and_rerank

# ─────────────────────────────────────────
# 1. ChromaDB'den test seti oluştur
# ─────────────────────────────────────────
def build_test_set(n=30):
    """
    DB'deki symptoms chunk'larından rastgele n tane seç.
    Her biri için:
      - query: chunk içeriğinden ilk 10 kelime
      - relevant_doc_id: o chunk'ın doc_id'si (disease bazlı)
      - disease: hastalık adı
    """
    db = get_chroma_client()
    data = db.get()

    symptom_chunks = [
        {
            "id": data["ids"][i],
            "content": data["documents"][i],
            "metadata": data["metadatas"][i]
        }
        for i in range(len(data["ids"]))
        if data["metadatas"][i].get("aspect") == "symptoms"
        and data["metadatas"][i].get("disease", "unknown") != "unknown"
        and len(data["documents"][i].strip()) > 20
    ]

    sampled = random.sample(symptom_chunks, min(n, len(symptom_chunks)))

    test_set = []
    for chunk in sampled:
        disease = chunk["metadata"].get("disease", "")
        # Query olarak chunk içeriğinin ilk 10 kelimesini kullan
        query_words = chunk["content"].strip().split()[:10]
        query = " ".join(query_words)

        test_set.append({
            "query": query,
            "disease": disease,
            "expected_doc_id": chunk["metadata"].get("doc_id", ""),
            "expected_content_snippet": chunk["content"][:80]
        })

    return test_set


# ─────────────────────────────────────────
# 2. Retrieval'ı çalıştır ve değerlendir
# ─────────────────────────────────────────
def evaluate(test_set):
    hits = 0          # en az 1 doğru chunk geldiyse
    precision_scores = []
    mrr_scores = []

    for i, item in enumerate(test_set):
        query = item["query"]
        disease = item["disease"].lower()

        retrieved_docs = search_and_rerank(query, k=10, final_k=3)
        time.sleep(7)  # Cohere trial: 10 istek/dakika sınırı

        # Her retrieve edilen chunk'ta hastalık adı geçiyor mu?
        relevant_retrieved = []
        for doc in retrieved_docs:
            doc_disease = doc.metadata.get("disease", "").lower()
            doc_content = doc.page_content.lower()
            if disease in doc_disease or disease in doc_content:
                relevant_retrieved.append(doc)

        # Precision@3
        precision = len(relevant_retrieved) / 3
        precision_scores.append(precision)

        # Hit Rate
        if len(relevant_retrieved) > 0:
            hits += 1

        # MRR — ilk doğru chunk kaçıncı sırada?
        mrr = 0
        for rank, doc in enumerate(retrieved_docs, start=1):
            doc_disease = doc.metadata.get("disease", "").lower()
            doc_content = doc.page_content.lower()
            if disease in doc_disease or disease in doc_content:
                mrr = 1 / rank
                break
        mrr_scores.append(mrr)

        print(f"[{i+1}/{len(test_set)}] Query: {query[:50]}...")
        print(f"  Hastalık: {disease}")
        print(f"  Precision@3: {precision:.2f} | MRR: {mrr:.2f} | Hit: {'✅' if mrr > 0 else '❌'}")
        print()

    print("=" * 50)
    print(f"TOPLAM TEST: {len(test_set)}")
    print(f"Hit Rate:    {hits / len(test_set):.2%}  ({hits}/{len(test_set)})")
    print(f"Precision@3: {sum(precision_scores) / len(precision_scores):.2%}")
    print(f"MRR:         {sum(mrr_scores) / len(mrr_scores):.3f}")
    print("=" * 50)


if __name__ == "__main__":
    print("Test seti oluşturuluyor...")
    test_set = build_test_set(n=30)
    print(f"{len(test_set)} test örneği oluşturuldu.\n")

    print("Retrieval değerlendirmesi başlıyor...\n")
    evaluate(test_set)