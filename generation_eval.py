"""
Generation Evaluation
clinical_node çıktısını 3 metrikle değerlendirir:

  1. Answer Correctness  – Beklenen hastalık yanıtta geçiyor mu? (0.0 – 1.0)
  2. Faithfulness        – Model RAG context'inde olmayan bir hastalık uyduruyor mu? (0.0 – 1.0)
  3. Answer Relevancy    – Yanıt gerçekten soruya mı odaklı? (0.0 – 1.0)

RAGAS'ın yaptığını manuel olarak LLM-as-judge ile uyguluyoruz:
  - Faithfulness      → yanıttaki her claim context'te var mı?
  - Answer Relevancy  → yanıttan tersine soru üretilip embedding benzerliğiyle ölçülür
"""

import os
import sys
import time
import json
import re

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from agents.nodes import rag_node, clinical_node, get_llm

# ── Test seti ────────────────────────────────────────────────────────────────
# Her case için:
#   patient_complaint   : hastanın ilk yazdığı şikayet (is_clarified=True varsayılır)
#   extracted_symptoms  : intent_node çıkışını simüle eden semptom listesi
#   expected_disease    : yanıtta mutlaka geçmesi beklenen hastalık adı/eşdeğerleri

test_set = [
    {
        "patient_complaint": "göğüs ağrısı sol kola yayılıyor, terleme ve nefes darlığı var",
        "extracted_symptoms": ["chest pain", "radiating to left arm", "sweating", "shortness of breath"],
        "expected_disease": ["kalp krizi", "miyokard", "kardiyak", "koroner", "kalp"],
    },
    {
        "patient_complaint": "düzensiz kalp atışı, çarpıntı hissediyorum, baş dönmesi var",
        "extracted_symptoms": ["irregular heartbeat", "palpitations", "dizziness"],
        "expected_disease": ["atriyal fibrilasyon", "aritmi", "kalp ritim", "kalp"],
    },
    {
        "patient_complaint": "yüksek tansiyon, baş ağrısı ve gözlerim bulanıyor",
        "extracted_symptoms": ["high blood pressure", "headache", "blurry vision"],
        "expected_disease": ["hipertansiyon", "yüksek tansiyon", "tansiyon"],
    },
    {
        "patient_complaint": "tek taraflı zonklayan baş ağrısı, bulantı ve ışığa duyarlılık",
        "extracted_symptoms": ["severe throbbing headache", "nausea", "light sensitivity"],
        "expected_disease": ["migren"],
    },
    {
        "patient_complaint": "vücudumun sol tarafında ani güçsüzlük, konuşmakta zorlanıyorum",
        "extracted_symptoms": ["sudden weakness", "one side of body", "slurred speech"],
        "expected_disease": ["inme", "felç", "serebrovasküler", "stroke"],
    },
    {
        "patient_complaint": "ateşim var, öksürük geçmiyor ve nefes alırken göğsüm ağrıyor",
        "extracted_symptoms": ["persistent cough", "fever", "chest pain when breathing"],
        "expected_disease": ["pnömoni", "zatürre", "akciğer enfeksiyonu", "bronşit"],
    },
    {
        "patient_complaint": "karın sağ alt tarafında şiddetli ağrı, ateş ve bulantı var",
        "extracted_symptoms": ["severe abdominal pain", "lower right", "fever", "nausea"],
        "expected_disease": ["apandisit", "pelvik inflamatuar", "PID", "over kisti", "ektopik gebelik", "kasık fıtığı"],
    },
    {
        "patient_complaint": "çok su içiyorum, sık idrara çıkıyorum, gözlerim bulanık ve yorgunum",
        "extracted_symptoms": ["excessive thirst", "frequent urination", "fatigue", "blurry vision"],
        "expected_disease": ["diyabet", "şeker hastalığı", "diabetes"],
    },
    {
        "patient_complaint": "idrar yaparken yanma, sık sık tuvalete gidiyorum ve idrarım bulanık",
        "extracted_symptoms": ["burning urination", "frequent urination", "cloudy urine"],
        "expected_disease": ["idrar yolu enfeksiyonu", "sistit", "üriner"],
    },
    {
        "patient_complaint": "belde başlayıp bacağıma yayılan ağrı var, ayağımda uyuşma hissediyorum",
        "extracted_symptoms": ["lower back pain", "radiating to leg", "numbness in foot"],
        "expected_disease": ["siyatik", "disk hernisi", "bel fıtığı", "lomber"],
    },
    {
        "patient_complaint": "ciltte yayılan kaşıntılı kızarıklık ve kurdeşen var",
        "extracted_symptoms": ["itchy red rash", "hives", "spreading"],
        "expected_disease": ["alerji", "ürtiker", "alerjik reaksiyon"],
    },
    {
        "patient_complaint": "kilo alıyorum, sürekli yorgunum, üşüyorum, saçlarım dökülüyor",
        "extracted_symptoms": ["weight gain", "fatigue", "feeling cold", "hair loss"],
        "expected_disease": ["hipotiroidizm", "tiroid", "guatr"],
    },
    {
        "patient_complaint": "kalp çarpıntısı, nefes darlığı, titreme ve korku hissediyorum",
        "extracted_symptoms": ["racing heart", "shortness of breath", "trembling", "feeling of doom"],
        "expected_disease": ["panik atak", "anksiyete", "panik bozukluk"],
    },
    {
        "patient_complaint": "yüzümde ağrı var, sarı renkli burun akıntısı ve burnumu tıkandı",
        "extracted_symptoms": ["nasal congestion", "facial pain", "yellow discharge"],
        "expected_disease": ["sinüzit", "sinüs enfeksiyonu"],
    },
    {
        "patient_complaint": "hafif öksürük ve hafif ateş var, genel olarak yorgunum sadece",
        "extracted_symptoms": ["mild cough", "low grade fever", "fatigue"],
        "expected_disease": ["viral enfeksiyon", "soğuk algınlığı", "grip", "üst solunum yolu"],
    },
]


# ── State builder ────────────────────────────────────────────────────────────

def build_state(item: dict) -> dict:
    return {
        "chat_history": [HumanMessage(content=item["patient_complaint"])],
        "patient_complaint": item["patient_complaint"],
        "is_clarified": True,
        "clarification_question": "",
        "medical_context": "",
        "final_decision": "",
        "extracted_symptoms": item["extracted_symptoms"],
        "next_node": "",
    }


# ── JSON temizleyici ─────────────────────────────────────────────────────────
# LLM bazen JSON'un etrafına açıklama metni ekleyebilir.
# Regex ile sadece {...} bloğunu çekiyoruz.

def clean_json(raw: str) -> str:
    raw = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    return match.group(0) if match else raw


# ── Metrik 1: Answer Correctness ─────────────────────────────────────────────
# İki aşamalı:
#   1. String match  → hızlı, API maliyeti yok
#   2. LLM doğrulama → string match başarısız olursa devreye girer
#      ("miyokard enfarktüsü" == "kalp krizi" gibi eşdeğerleri yakalar)

CORRECTNESS_PROMPT = """Bir tıbbi triyaj sisteminin hastaya verdiği yanıtı değerlendir.

Beklenen hastalık: {expected}
Sistem yanıtı: {response}

Soru: Bu yanıt, beklenen hastalığı tıbbi olarak doğruluyor mu?
Eşanlamlı terimler, teknik adlar veya ilgili açıklamalar da kabul edilir.
(Örnek: "miyokard enfarktüsü" → "kalp krizi" için EVET)

Sadece JSON döndür:
{{"confirmed": true/false, "reason": "<kısa açıklama>"}}"""

def answer_correctness(response: str, expected_diseases: list[str], use_llm_fallback: bool = True):
    """
    Aşama 1 — string match.
    Aşama 2 — string match başarısızsa LLM ile semantik doğrulama.
    Döner: (hit_binary, hit_ratio, llm_used)
    """
    resp_lower = response.lower()
    hits = sum(1 for d in expected_diseases if d.lower() in resp_lower)
    hit_binary = 1.0 if hits > 0 else 0.0
    hit_ratio  = hits / len(expected_diseases) if expected_diseases else 0.0
    llm_used   = False

    # String match başarısızsa LLM'e sor
    if hit_binary == 0.0 and use_llm_fallback:
        llm = get_llm()
        primary_disease = expected_diseases[0]
        prompt = CORRECTNESS_PROMPT.format(expected=primary_disease, response=response)
        raw = llm.invoke([
            SystemMessage(content="Sen bir tıbbi AI kalite değerlendirme uzmanısın."),
            HumanMessage(content=prompt)
        ])
        try:
            result = json.loads(clean_json(raw.content))
            if result.get("confirmed", False):
                hit_binary = 1.0
                hit_ratio  = max(hit_ratio, 1.0 / len(expected_diseases))
                llm_used   = True
                print(f"  [LLM Correctness] Semantik eşleşme: {result.get('reason', '')}")
        except Exception as e:
            print(f"  [LLM Correctness parse hatası]: {e}")

    return hit_binary, hit_ratio, llm_used


# ── Metrik 2: Faithfulness ───────────────────────────────────────────────────
# Yanıttaki tıbbi claimler RAG context'inden geliyor mu?
# RAGAS yaklaşımı: yanıtı cümlelere böl → her cümle için context'te destek var mı?

FAITHFULNESS_PROMPT = """Sana bir RAG sisteminin retrieved context'i ve bu context'e dayanarak üretilmiş bir yanıt verilecek.

Görevin: Yanıttaki her tıbbi iddiayı (claim) incele.
- Eğer iddia context'ten destekleniyorsa veya makul bir tıbbi gerçekse: desteklendi
- Eğer iddia context'te YOK ve uydurulmuş bir hastalık/ilaç/bilgi içeriyorsa: desteklenmedi

Desteklenen claim sayısını ve toplam claim sayısını say.
Sadece JSON döndür (başka hiçbir şey yazma):
{{"supported_claims": <int>, "total_claims": <int>, "faithfulness": <float 0-1>}}

--- CONTEXT ---
{context}

--- YANIT ---
{response}
"""

def faithfulness(context: str, response: str) -> float:
    llm = get_llm()
    prompt = FAITHFULNESS_PROMPT.format(
        context=context[:3000],
        response=response
    )
    raw = llm.invoke([
        SystemMessage(content="Sen bir tıbbi RAG sistemi kalite değerlendirme uzmanısın."),
        HumanMessage(content=prompt)
    ])
    try:
        result = json.loads(clean_json(raw.content))
        return float(result.get("faithfulness", 0.0))
    except Exception as e:
        print(f"  [Faithfulness parse hatası]: {e}")
        return 0.0


# ── Metrik 3: Answer Relevancy ───────────────────────────────────────────────
# RAGAS yaklaşımı: yanıttan N adet soru üret → bu soruları orijinal soruyla
# embedding benzerliğiyle karşılaştır → ortalama cosine similarity = relevancy

REVERSE_QUESTION_PROMPT = """
Sana bir tıbbi asistanın yanıtı verilecek. 
Bu yanıtın DOĞRUDAN cevapladığı, sadece hastanın şikayetlerini ve semptomlarını içeren 3 çok kısa soru üret. 

Kurallar:
- Sorular sadece semptom odaklı olsun (Örn: "Göğüs ağrısı ve terleme ne anlama gelir?").
- Yanıttaki "doktora gidin" veya "geçmiş olsun" gibi kısımları görmezden gel, sadece tıbbi çıkarıma odaklan.
- Sadece JSON döndür:
{{"questions": ["soru1", "soru2", "soru3"]}}

--- YANIT ---
{response}
"""

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    import math
    dot   = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def answer_relevancy(complaint: str, response: str) -> float:
    llm        = get_llm()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Adım 1: Yanıttan tersine sorular üret
    prompt = REVERSE_QUESTION_PROMPT.format(response=response)
    raw = llm.invoke([
        SystemMessage(content="Sen bir tıbbi AI kalite değerlendirme uzmanısın."),
        HumanMessage(content=prompt)
    ])
    try:
        generated_questions = json.loads(clean_json(raw.content)).get("questions", [])
    except Exception as e:
        print(f"  [Answer Relevancy soru üretme hatası]: {e}")
        return 0.0

    if not generated_questions:
        return 0.0

    # Adım 2: Orijinal şikayet ile üretilen sorular arasındaki cosine similarity
    original_emb    = embeddings.embed_query(complaint)
    generated_embs  = embeddings.embed_documents(generated_questions)

    similarities = [cosine_similarity(original_emb, gen_emb) for gen_emb in generated_embs]
    return sum(similarities) / len(similarities)


# ── Ana değerlendirme döngüsü ────────────────────────────────────────────────

def evaluate(test_set: list[dict]):
    correctness_hits   = []   # binary (0/1)
    correctness_ratios = []   # partial credit
    faithfulness_scores     = []
    relevancy_scores        = []

    for i, item in enumerate(test_set, start=1):
        complaint = item["patient_complaint"]
        print(f"\n[{i}/{len(test_set)}] {complaint[:65]}...")

        # RAG + Clinical node çalıştır
        state           = build_state(item)
        rag_result      = rag_node(state)
        context         = rag_result["medical_context"]
        state["medical_context"] = context

        clinical_result = clinical_node(state)
        response        = clinical_result["final_decision"]

        print(f"  Yanıt (ilk 180 kr): {response[:180].replace(chr(10), ' ')}...")

        # 1. Answer Correctness (string match + LLM fallback)
        hit_binary, hit_ratio, llm_used = answer_correctness(response, item["expected_disease"])
        correctness_hits.append(hit_binary)
        correctness_ratios.append(hit_ratio)

        # 2. Faithfulness  (LLM çağrısı)
        time.sleep(2)
        faith_score = faithfulness(context, response)
        faithfulness_scores.append(faith_score)

        # 3. Answer Relevancy  (LLM + embedding çağrısı)
        time.sleep(2)
        rel_score = answer_relevancy(complaint, response)
        relevancy_scores.append(rel_score)

        print(f"  Answer Correctness : {'✅' if hit_binary else '❌'}  (ratio: {hit_ratio:.0%})"
              f"{'  [LLM]' if llm_used else ''}  | Beklenen: {item['expected_disease'][0]}")
        print(f"  Faithfulness       : {faith_score:.2f}")
        print(f"  Answer Relevancy   : {rel_score:.2f}")

    # ── Özet ────────────────────────────────────────────────────────────────
    total = len(test_set)
    print("\n" + "=" * 55)
    print(f"TOPLAM TEST                   : {total}")
    print(f"Answer Correctness (Hit Rate) : {sum(correctness_hits)/total:.2%}  ({int(sum(correctness_hits))}/{total})")
    print(f"Answer Correctness (Ort Oran) : {sum(correctness_ratios)/total:.2%}")
    print(f"Faithfulness       (ort)      : {sum(faithfulness_scores)/total:.3f}")
    print(f"Answer Relevancy   (ort)      : {sum(relevancy_scores)/total:.3f}")
    print("=" * 55)


if __name__ == "__main__":
    print(f"Generation test seti: {len(test_set)} soru\n")
    print("Generation değerlendirmesi başlıyor...\n")
    evaluate(test_set)