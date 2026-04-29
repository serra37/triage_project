import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from database.chroma_client import search_and_rerank
from agents.state import TriageState
import json


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

#yönetici agent
def supervisor_node(state: TriageState):
    """Sistemin akışını yönetir ve sıradaki düğümü belirler."""
    if not state.get("is_clarified", False):
        if state.get("clarification_question"):
            return {"next_node": "end"}
        return {"next_node": "intent"}
    if not state.get("medical_context"):
        return {"next_node": "rag"}
    if not state.get("final_decision"):
        return {"next_node": "clinical"}
    return {"next_node": "end"}

def intent_node(state: TriageState):
    """Kullanıcının şikayetini analiz edip eksiği var mı diye bakar. Varsa soru üretir."""
    complaint = state.get("patient_complaint", "")      #kullanıcnın şikayetini çeker
    history = state.get("chat_history", [])     #diyalogları liste olarak tutar
    
    # Sohbet geçmişini formata dönüştür
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history[-5:]]) # Son 5 mesajı modele verecek
    
    sys_msg = SystemMessage(content="""
    Sen insanların hastaneye veya acil servise gitmeden ÖNCE danıştıkları uzman bir yapay zeka sağlık chatbotusun.
    Görevin hastanın şikayetini dinleyip, durumları hakkında doğru bilgilendirme ve yönlendirme yapabilmek için eksik kalan bilgileri (ne zaman başladı, şiddeti nedir, kronik hastalık var mı vb.) sormaktır.
    Bilgi yeterliliğine ulaşana kadar hastaya teşhis söyleme, sadece empati kurarak sorunu derinleştiren tek bir soru sor.
    Eğer hastanın verdiği şikayetler durumu değerlendirmek için yeterliyse JSON formatında 'is_clarified': true döndür ve hastanın belirttiği tüm semptomları İngilizce tıp terimleri veya anahtar kelimeler şeklinde bir liste olarak 'extracted_symptoms' anahtarında döndür (Örn: ["chest pain", "shortness of breath", "nausea"]).
    
    Eğer ek bir soru sorman gerekiyorsa JSON formatında 'is_clarified': false ve 'question': "Soracağın netleştirici soru" şeklinde döndür.
    Sadece ve sadece JSON formatında yanıt ver, başka bir metin içerme.
    """)
    
    user_msg_content = f"Şikayet: {complaint}\nSohbet Geçmişi: {history_str}"   #o an ki konuşmaları tek bir metin haline getirerk modele verecek
    human_msg = HumanMessage(content=user_msg_content)
    

    response = get_llm().invoke([sys_msg, human_msg])   #hem sistem promptunu hem de hastanın şikayetlerini modele gönderiyoruz
    
    try:
        
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        is_clarified = result.get("is_clarified", False)
        question = result.get("question", "")
        extracted_symptoms = result.get("extracted_symptoms", [])
        
        return {
            "is_clarified": is_clarified,
            "clarification_question": question,
            "extracted_symptoms": extracted_symptoms
        }
    except Exception as e:
        print(f"JSON Parse Hatası Intent Düğümünde: {e}")
        # Hata anında en azından bir standart soru sor
        return {
            "is_clarified": False,
            "clarification_question": "Şikayetinizle ilgili başka belirtebileceğiniz bir detay var mı?",
            "extracted_symptoms": []
        }

def rag_node(state: TriageState):
    """Şikayet netleştiğinde RAG üzerinden tıbbi vaka/literatür tarar."""
    extracted_symptoms = state.get("extracted_symptoms", [])
    if extracted_symptoms:
        query = " ".join(extracted_symptoms)
    else:
        query = state.get("patient_complaint", "")

    print("EXTRACTED:", extracted_symptoms)
    print("FINAL QUERY:", query)
    
    docs = search_and_rerank(query, k=20, final_k=5)
    
    context = "\n---\n".join([doc.page_content for doc in docs])  
    
    return {"medical_context": context}     

def clinical_node(state: TriageState):
    """Tüm bilgileri sentezleyip hastayı doğru polikliniğe yönlendirir."""
    complaint = state.get("patient_complaint", "")
    history = state.get("chat_history", [])
    context = state.get("medical_context", "")
    
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history if msg.type in ['human', 'ai']])
    
    sys_msg = SystemMessage(content="""
    Sen insanların hastaneye gitmeden önce danıştıkları, yetkin ve empatik bir yapay zeka sağlık danışmanısın.
    Kullanıcının beyan ettiği şikayetleri ve RAG sisteminden çekilen benzer tıbbi vakaları analiz ederek hastayı olası rahatsızlığı hakkında kısaca ve güven verici bir dille BİLGİLENDİR. 
    Kesin bir teşhis koymaktan kaçın, ancak mevcut belirtilerin hangi tıbbi durumlara işaret edebileceğini açıkla.
    Bu bilgilendirmenin MAKSADI hastayı doğru yönlendirmektir, bu yüzden bilgilendirmenin sonucunda hastaya ne yapması gerektiğini açıkça belirt.
    (Örn: "Durumunuz acil servislik görünüyor, lütfen hemen başvurun", "Yakın zamanda bir Dahiliye doktorundan randevu alabilirsiniz", "Bu belirtiler evde istirahatle geçebilir" vb.)
    """)
    
    user_prompt = f"""
    Hastanın Ana Şikayeti: {complaint}
    Ek Görüşmeler (Semptom detayları):
    {history_str}
    
    Benzer Tıbbi Vaka Referansları (Literatür):
    {context}
    
    Lütfen nihai tıbbi yönlendirme kararını yazın:
    """
    
    human_msg = HumanMessage(content=user_prompt)
    response = get_llm().invoke([sys_msg, human_msg])   #Burada elimizdeki tüm o sentezlenmiş bilgileri (şikayet, mülakat, RAG verisi) LLM'e (GPT'ye) paketleyip gönderiyoruz. LLM bunu okuyor ve "Sizin durumunuz Dahiliye randevusu gerektiriyor..." gibi uzun bir metin oluşturuyor.
    
    return {"final_decision": response.content}

