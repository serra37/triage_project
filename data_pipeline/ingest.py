import os
import sys
import time
import json
from dotenv import load_dotenv

# Ana dizini yola ekle (import hatalarını önlemek için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from datasets import load_dataset
from langchain_core.documents import Document
from database.chroma_client import get_chroma_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def chunk_with_llm(text, source, base_id):
    """LLM kullanarak metni belirtiler, nedenler ve tedavi olarak JSON'a ayırıp Document listesi döner."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sys_msg = SystemMessage(content="""
        Verilen tıbbi metinden şu bilgileri ayrı ayrı çıkartıp SADECE ve SADECE aşağıdaki JSON formatında döndür:
        {
            "disease": "Eğer metinde hastalık adı geçiyorsa buraya yaz, geçmiyorsa boş bırak",
            "symptoms": "Hastalığın belirtilerini yaz, eğer yoksa boş bırak",
            "causes": "Hastalığın nedenlerini veya risk faktörlerini yaz, yoksa boş bırak",
            "treatment": "Hastalığın tedavisini, ne yapılması gerektiğini yaz, yoksa boş bırak"
        }
        Eğer metinde ilgili kategori için hiçbir bilgi yoksa boş bırak (""). Açıklama, giriş veya ek metin ekleme, doğrudan geçerli bir JSON döndür.
    """)
    human_msg = HumanMessage(content=text)
    
    docs = []
    try:
        response = llm.invoke([sys_msg, human_msg])
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        disease = result.get("disease", "")
        
        for aspect, content_text in zip(["symptoms", "causes", "treatment"], 
                                   [result.get("symptoms"), result.get("causes"), result.get("treatment")]):
            if content_text and content_text.strip():
                docs.append(Document(
                    page_content=content_text.strip(),
                    metadata={
                        "source": source,
                        "aspect": aspect,
                        "disease": disease if disease else "unknown",
                        "doc_id": f"{base_id}_{aspect}"
                    }
                ))
    except Exception as e:
        print(f"JSON Parse Hatası Ingestion'da: {e}")
        # Hata durumunda fallback olarak tek chunk ekle
        docs.append(Document(
            page_content=text,
            metadata={"source": source, "aspect": "general", "disease": "unknown", "doc_id": f"{base_id}_general"}
        ))
    
    return docs

def process_medquad_row(row):
    """MedQuAD verisini Soru-Cevap olarak hazırlar."""
    question = row.get('question', '').strip()
    answer = row.get('answer', '').strip()
    question_id = row.get("question_id", row.get("id", "unknown"))
    text = f"Soru: {question}\nCevap: {answer}"
    return chunk_with_llm(text, "MedQuAD", str(question_id))

def process_healthcaremagic_row(row, idx):
    """HealthCareMagic verisini temizler ve asıl şikayete odaklanır."""
    question = row.get('input', '').strip()
    answer = row.get('output', '').strip()
    dirty_prefix = "If you are a doctor, please answer the medical questions based on the patient's description."
    
    if not question:
        instruction = row.get('instruction', '').strip()
        question = instruction.replace(dirty_prefix, "").strip()
    else:
        question = question.replace(dirty_prefix, "").strip()

    text = f"Şikayet: {question}\nDoktor Yanıtı: {answer}"
    return chunk_with_llm(text, "HealthCareMagic", f"hc_{idx}")

def ingest_data(sample_size=1000):
    """Sadece verilen sample_size kadar veri alarak LLM tabanlı vektör veritabanı oluşturur."""
    print(f"[TEST MODU] AKTİF: Her verisetinden sadece ilk {sample_size} kayıt alınıyor...")
    docs = []
    
    # 1. MedQuAD Yükleme
    try:
        print("[+] MedQuAD verileri indiriliyor ve LLM ile chunklanıyor...")
        medquad = load_dataset("lavita/MedQuAD", split=f"train[:{sample_size}]")
        for i, row in enumerate(medquad):
            docs.extend(process_medquad_row(row))
            if (i+1) % 10 == 0:
                print(f"   MedQuAD {i+1}/{sample_size} işlendi...")
    except Exception as e:
        print(f"MedQuAD yükleme hatası: {e}")

    # 2. HealthCareMagic Yükleme
    try:
        print("[+] HealthCareMagic verileri indiriliyor ve LLM ile chunklanıyor...")
        hc_magic = load_dataset("wangrongsheng/HealthCareMagic-100k-en", split=f"train[:{sample_size}]")
        for i, row in enumerate(hc_magic):
            docs.extend(process_healthcaremagic_row(row, i))
            if (i+1) % 10 == 0:
                print(f"   HealthCareMagic {i+1}/{sample_size} işlendi...")
    except Exception as e:
        print(f"HealthCareMagic yükleme hatası: {e}")

    if not docs:
        print("[HATA] Veritabanına eklenecek döküman bulunamadı.")
        return

    print(f"[BASARILI] Toplam {len(docs)} chunk oluşturuldu. ChromaDB'ye OpenAI modeli ile yazılıyor...")
    db = get_chroma_client()
    
    batch_size = 50 
    for i in range(0, len(docs), batch_size):
        try:
            batch = docs[i:i + batch_size]
            db.add_documents(batch)
            print(f"[*] İlerleme: {i + len(batch)}/{len(docs)} eklendi.")
            time.sleep(0.5) 
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower():
                print("[UYARI] Hız sınırına takıldık, 5 saniye mola veriliyor...")
                time.sleep(5)
                db.add_documents(batch)
            else:
                print(f"[UYARI] Hata: {e}")
                time.sleep(5)
    
    print(f"\n[BITTI] İşlem başarıyla tamamlandı! Kütüphanen artık taze {len(docs)} chunk ile dolu.")

if __name__ == "__main__":
    ingest_data(sample_size=1000)