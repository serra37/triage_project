from langgraph.graph import StateGraph, START, END
from agents.state import TriageState
from agents.nodes import supervisor_node, intent_node, rag_node, clinical_node

def route_from_supervisor(state: TriageState):
    """Supervisor'ın kararına göre ('next_node') grafikte bir sonraki ajanı belirler."""
    return state.get("next_node", "end")

workflow = StateGraph(TriageState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("intent", intent_node)    # Niyet ajani
workflow.add_node("rag", rag_node)          # RAG ajani
workflow.add_node("clinical", clinical_node)# Klinik karar ajani

# Başlangıçta tüm trafiği yönetici ajan devralır
workflow.add_edge(START, "supervisor")

# Yöneticinin verdiği 'next_node' cevabına göre agent ataması yap
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor, 
    {
        "intent": "intent",
        "rag": "rag",
        "clinical": "clinical",
        "end": END
    }
)

# Ajanlar işlerini bitirince geri rapor vermek için doğrudan yöneticiye (supervisor) döner
workflow.add_edge("intent", "supervisor")
workflow.add_edge("rag", "supervisor")
workflow.add_edge("clinical", "supervisor")

# Grafı derle
triage_app = workflow.compile()
