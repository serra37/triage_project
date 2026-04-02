import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from datasets import load_dataset
from langchain_core.documents import Document
from database.chroma_client import get_chroma_client

def process_medquad_row(row):
    text = f"Soru: {row['question']}\nCevap: {row['answer']}"
    return Document(page_content=text, metadata={"source": "MedQuAD"})

def process_healthcaremagic_row(row):
    input_text = row.get("input", "")
    full_query = f"{row.get('instruction', '')} {input_text}".strip()
    text = f"Şikayet/Soru: {full_query}\nYanıt: {row.get('output', '')}"
    return Document(page_content=text, metadata={"source": "HealthCareMagic"})

def ingest_data(sample_size=500):
    print("Veri setleri indiriliyor ve işleniyor (sadece ilk", sample_size, "kayıt).")
    
    docs = []
    
    # 1. MedQuAD 
    try:
        medquad = load_dataset("lavita/MedQuAD", split=f"train[:{sample_size}]")
        for row in medquad:
            docs.append(process_medquad_row(row))
        print(f"MedQuAD: {len(docs)} belge hazırlandı.")
    except Exception as e:
        print(f"MedQuAD indirme hatası: {e}")
        
    # 2. HealthCareMagic 
    try:
        hc_magic = load_dataset("wangrongsheng/HealthCareMagic-100k-en", split=f"train[:{sample_size}]")
        docs_before = len(docs)
        for row in hc_magic:
            docs.append(process_healthcaremagic_row(row))
        print(f"HealthCareMagic: {len(docs) - docs_before} belge hazırlandı.")
    except Exception as e:
        print(f"HealthCareMagic indirme hatası: {e}")
        
    if not docs:
        print("İşlenecek belge bulunamadı.")
        return

    print("Dokümanlar ChromaDB'ye ekleniyor")
    db = get_chroma_client()
    # Batch size
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        db.add_documents(batch)
        print(f"[{i}-{i+len(batch)}]/{len(docs)} eklendi.")
    
    print("Veri indirme ve indeksleme işlemi başarıyla tamamlandı.")

if __name__ == "__main__":
    # Geliştirme ortamı için limitli şekilde çalıştır (500 MedQuAD, 500 HC Magic)
    ingest_data(sample_size=500)
 
