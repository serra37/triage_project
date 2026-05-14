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

    # Netlestirilmis mi?
    if not state.get("is_clarified", False):
        # Soru sorulmussa kullanicidan cevap bekle
        if state.get("clarification_question"):
            return {"next_node": "end"}
        # Soru sorulmamissa intent'e git
        return {"next_node": "intent"}

    # Netlestirildiyse RAG'a git
    if not state.get("medical_context"):
        return {"next_node": "rag"}

    # RAG tamsa clinical'a git
    if not state.get("final_decision"):
        return {"next_node": "clinical"}

    # Her sey tamsa bitir
    return {"next_node": "end"}

def intent_node(state: TriageState):
    """Kullanıcının şikayetini analiz edip eksiği var mı diye bakar. Varsa soru üretir."""
    complaint = state.get("patient_complaint", "")      #kullanıcnın şikayetini çeker
    history = state.get("chat_history", [])     #diyalogları liste olarak tutar
    user_response_count = max(
        sum(1 for msg in history if getattr(msg, "type", "") == "human") - 1,
        0
    )

    # Sohbet geçmişini formata dönüştür
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])

    sys_msg = SystemMessage(content="""
    Sen insanların hastaneye veya acil servise gitmeden ÖNCE danıştıkları uzman bir yapay zeka sağlık chatbotusun.
    Görevin hastanın şikayetini dinleyip, durumları hakkında doğru bilgilendirme ve yönlendirme yapabilmek için eksik kalan bilgileri (ne zaman başladı, şiddeti nedir, kronik hastalık var mı vb.) sormaktır.
    Eğer hastanın verdiği şikayetler durumu değerlendirmek için yeterliyse JSON formatında 'is_clarified': true döndür ve hastanın belirttiği tüm semptomları İngilizce tıp terimleri veya anahtar kelimeler şeklinde bir liste olarak 'extracted_symptoms' anahtarında döndür (Örn: ["chest pain", "shortness of breath", "nausea"]).

    Strict rules:
    1. If the chat_history contains ANY user response after the initial complaint, immediately return is_clarified: true.
    2. Never ask the same question twice.
    3. Only ask ONE clarifying question when the very first message has insufficient info.
    4. If the user has answered at least one follow-up question, always return is_clarified: true.

    Eğer ek bir soru sorman gerekiyorsa JSON formatında 'is_clarified': false ve 'question': "Soracağın netleştirici soru" şeklinde döndür.
    Sadece ve sadece JSON formatında yanıt ver, başka bir metin içerme.
    """)
    
    user_msg_content = f"""
Ana Şikayet: {complaint}

Konuşma Geçmişi ({user_response_count} kullanıcı cevabı):
{history_str}

KURAL: Eğer kullanıcı en az 1 cevap verdiyse (user_response_count >= 1),
kesinlikle is_clarified: true döndür. Aynı soruyu tekrar sorma.
"""
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

