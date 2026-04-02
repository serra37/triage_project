import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph import triage_app

# API anahtarını yükle
load_dotenv()

def main():
    print("=== Tıbbi Triyaj Asistanı ===")
    print("Lütfen şikayetinizi yazın (Çıkmak için 'q' tuşuna basın).\n")
    
    state = {
        "chat_history": [],
        "patient_complaint": "",
        "is_clarified": False,
        "clarification_question": "",
        "medical_context": "",
        "final_decision": "",
        "next_node": ""
    }
    
    while True:
        user_input = input("Siz: ")
        
        if user_input.lower() == 'q':
            print("Asistandan çıkılıyor...")
            break
            
        if not user_input.strip():
            continue
            
        # İlk giriş mi?
        if not state["patient_complaint"]:
            state["patient_complaint"] = user_input
            
        state["chat_history"].append(HumanMessage(content=user_input))
        
        print("\n*Asistan düşünüyor...*")
        try:
            result = triage_app.invoke(state)
        except Exception as e:
            print(f"\n[HATA]: Lütfen .env dosyasında OPENAI_API_KEY olduğundan emin olun!\nDetay: {e}")
            break
        
        # Durumu güncelle
        state["is_clarified"] = result.get("is_clarified", False)
        state["medical_context"] = result.get("medical_context", "")
        state["final_decision"] = result.get("final_decision", "")
        
        # Henüz netleşmediyse soruyu ekrana yaz ve döngüye devam et
        if not result.get("is_clarified") and not result.get("final_decision"):
            question = result.get("clarification_question", "Lütfen şikayetinizle ilgili daha fazla detay verin.")
            print(f"Asistan: {question}\n")
            state["chat_history"].append(AIMessage(content=question))
            
        # Karar verildiyse cevabı yaz ve bitir
        elif result.get("final_decision"):
            print("\n=== SİSTEMİN DEĞERLENDİRMESİ VE YÖNLENDİRMESİ ===")
            print(result["final_decision"])
            break

if __name__ == "__main__":
    main()

