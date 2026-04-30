"""
Manual Retrieval Evaluation — Baseline
Query expansion yok.
Metrikler: Recall@5 / Hit Rate, Precision@5, MRR
"""

import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from database.chroma_client import search_and_rerank

K = 20
FINAL_K = 5

test_set = [
    {"query": "chest pain radiating to left arm, sweating, shortness of breath",
     "expected_diseases": ["heart attack", "heart disease", "cardiac"]},

    {"query": "irregular heartbeat, palpitations, dizziness",
     "expected_diseases": ["atrial fibrillation", "arrhythmia", "irregular heart beat"]},

    {"query": "high blood pressure headache blurry vision",
     "expected_diseases": ["hypertension", "high blood pressure"]},

    {"query": "severe throbbing headache one side, nausea, light sensitivity",
     "expected_diseases": ["migraine"]},

    {"query": "dizziness spinning sensation when lying down, nausea",
     "expected_diseases": ["bppv", "vertigo", "cervical spondylosis"]},

    {"query": "sudden weakness one side of body, slurred speech, confusion",
     "expected_diseases": ["stroke", "mini strokes"]},

    {"query": "persistent cough, fever, chest pain when breathing",
     "expected_diseases": ["pneumonia", "lung infection", "bronchitis"]},

    {"query": "shortness of breath, wheezing, chest tightness",
     "expected_diseases": ["asthma", "copd", "allergic asthma"]},

    {"query": "severe abdominal pain lower right side, fever, nausea",
     "expected_diseases": ["appendicitis"]},

    {"query": "burning sensation stomach after eating, acid taste in mouth",
     "expected_diseases": ["acid reflux", "gastritis", "gastroesophageal reflux"]},

    {"query": "blood in stool, abdominal cramps, diarrhea",
     "expected_diseases": ["irritable bowel", "bowel cancer", "gastroenteritis"]},

    {"query": "excessive thirst, frequent urination, fatigue, blurry vision",
     "expected_diseases": ["diabetes", "diabetes mellitus"]},

    {"query": "sudden sweating, shaking, confusion, feeling faint",
     "expected_diseases": ["diabetes", "hypoglycemia", "hipoglisemi"]},

    {"query": "burning pain when urinating, frequent urination, cloudy urine",
     "expected_diseases": ["urinary tract infection", "urinary infection", "idrar yolu enfeksiyonu"]},

    {"query": "severe flank pain radiating to groin, blood in urine",
     "expected_diseases": ["kidney stones", "renal calculus"]},

    {"query": "lower back pain radiating down leg, numbness in foot",
     "expected_diseases": ["sciatica", "herniated disc", "disc herniation"]},

    {"query": "joint swelling, morning stiffness, pain in multiple joints",
     "expected_diseases": ["arthritis", "psoriatic arthritis", "artrit"]},

    {"query": "itchy red rash spreading on skin, hives",
     "expected_diseases": ["allergy", "allergic reaction", "urticaria"]},

    {"query": "painful swollen area on skin, pus, warmth",
     "expected_diseases": ["abscess", "cellulitis", "infection"]},

    {"query": "weight gain, fatigue, feeling cold, hair loss, constipation",
     "expected_diseases": ["hypothyroidism", "thyroid dysfunction"]},

    {"query": "high fever, body aches, sore throat, runny nose",
     "expected_diseases": ["viral fever", "viral illness", "upper respiratory infection"]},

    {"query": "yellow skin, yellow eyes, dark urine, fatigue",
     "expected_diseases": ["hepatitis", "jaundice", "liver disease"]},

    {"query": "racing heart, shortness of breath, feeling of doom, trembling",
     "expected_diseases": ["anxiety", "panic attack", "panik atak"]},

    {"query": "nasal congestion, facial pain, yellow discharge from nose",
     "expected_diseases": ["sinusitis", "sinus infection", "nasal congestion"]},

    {"query": "pelvic pain, abnormal discharge, painful intercourse",
     "expected_diseases": ["pelvic inflammatory disease", "pid", "endometriosis"]},

    {"query": "göğsüm ağrıyor sol koluma vuruyor nefes alamıyorum",
     "expected_diseases": ["heart attack", "heart disease", "kalp"]},

    {"query": "idrar yaparken yanıyor sık sık tuvalete gidiyorum",
     "expected_diseases": ["urinary tract infection", "idrar yolu enfeksiyonu"]},

    {"query": "başım dönüyor mide bulantım var ayağa kalkınca kötüleşiyor",
     "expected_diseases": ["bppv", "vertigo", "cervical spondylosis"]},

    {"query": "çok su içiyorum sık idrara çıkıyorum gözlerim bulanık",
     "expected_diseases": ["diabetes", "diabetes mellitus"]},

    {"query": "sırt ağrısı bacağıma vuruyor ayağım uyuşuyor",
     "expected_diseases": ["sciatica", "herniated disc", "disc herniation"]},
]


ALIAS_MAP = {
    "heart attack": ["myocardial infarction", "cardiac", "angina", "heart disease"],
    "asthma": ["bronchial", "wheezing", "breathlessness", "spasm"],
    "acid reflux": ["gerd", "gastroesophageal reflux", "gastritis", "hiatus"],
    "arthritis": ["rheumatoid", "joint inflammation", "costochondritis", "lupus"],
    "allergy": ["urticaria", "allergen", "allergic"],
    "viral fever": ["viral infection", "viral illness"],
    "migraine": ["cluster headache", "ischemia of brain", "headache"],
    "appendicitis": ["abdomen pain", "acidity", "pancreatitis"],
    "diabetes": ["hypoglycemia", "hipoglisemi", "glucose", "insulin"],
    "bppv": ["vertigo", "dizziness", "cervical spondylosis"],
    "urinary tract infection": ["idrar yolu", "urinary infection", "cystitis"],
    "sciatica": ["disc herniation", "herniated", "nerve compression"],
}

def is_relevant(doc, expected_diseases):
    doc_disease = doc.metadata.get("disease", "").lower()
    doc_content = doc.page_content.lower()

    expanded = list(expected_diseases)
    for exp in expected_diseases:
        if exp in ALIAS_MAP:
            expanded.extend(ALIAS_MAP[exp])

    return any(
        term in doc_disease or term in doc_content
        for term in expanded
    )


def evaluate(test_set):
    hits = 0
    precision_scores = []
    recall_scores = []
    mrr_scores = []

    for i, item in enumerate(test_set, start=1):
        query = item["query"]
        expected = [e.lower() for e in item["expected_diseases"]]

        retrieved_docs = search_and_rerank(query, k=K, final_k=FINAL_K)
        time.sleep(7)

        relevant_flags = [is_relevant(doc, expected) for doc in retrieved_docs]
        relevant_count = sum(relevant_flags)

        precision = relevant_count / len(retrieved_docs) if retrieved_docs else 0
        hit = relevant_count > 0
        recall = 1.0 if hit else 0.0

        mrr = 0
        for rank, is_rel in enumerate(relevant_flags, start=1):
            if is_rel:
                mrr = 1 / rank
                break

        if hit:
            hits += 1

        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)

        print(f"[{i}/{len(test_set)}] Query: {query[:55]}...")
        print(f"  Beklenen: {expected[0]}")
        print(f"  Precision@{FINAL_K}: {precision:.2f} | Recall@{FINAL_K}: {recall:.2f} | MRR: {mrr:.2f} | Hit: {'✅' if hit else '❌'}")
        print()

    total = len(test_set)

    print("=" * 50)
    print(f"TOPLAM TEST: {total}")
    print(f"Recall@{FINAL_K} / Hit Rate: {sum(recall_scores) / total:.2%}  ({hits}/{total})")
    print(f"Precision@{FINAL_K}:         {sum(precision_scores) / total:.2%}")
    print(f"MRR:                         {sum(mrr_scores) / total:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    print(f"Manuel test seti: {len(test_set)} soru\n")
    print("Baseline retrieval değerlendirmesi başlıyor...\n")
    evaluate(test_set)