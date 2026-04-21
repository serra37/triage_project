from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class TriageState(TypedDict):
     #listeye yeni mesaj geldiğinde geçmişi silme, üzerşne ekle
    chat_history: Annotated[Sequence[BaseMessage], operator.add]   
    
    # hastanın ana şikayeti ve konuşmalar
    patient_complaint: str
    
    # kullanıcının ilk yazdığı ana şikayeti odağı kaybetmemek için burada sabit tutuyoruz burası false olduğu sürece niyet ajanı hastaya soru sormaya devam eder ve true döndüğünde ise rag ajanına yönlendiriyo akışı
    is_clarified: bool
    
    # istenebilecek ek soruların geçici olarak tutulduğu yer
    clarification_question: str
    
    # ragdan gelen tıbbi literatür 
    medical_context: str
    
    # son karar 
    final_decision: str
    
    # anlaşılan semptomlar (İngilizce keyword listesi)
    extracted_symptoms: list[str]
    
    # bir sonraki çalışacak düğümü yönlendirmek için
    next_node: str


#düğümler arasındaki akışın (trafiğin) nereye gideceğine buradaki verilere bakarak karar veriyoruz.