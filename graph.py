from langgraph.graph import StateGraph, START, END
from agents.state import TriageState
from agents.nodes import intent_node, rag_node, clinical_node

def route_after_intent(state: TriageState):
    """Niyet analizinin sonucuna göre sonrakı adımı belirler."""
    if state.get("is_clarified"):
        return "rag"
    else:
        return END

workflow = StateGraph(TriageState)

workflow.add_node("intent", intent_node)
workflow.add_node("rag", rag_node)
workflow.add_node("clinical", clinical_node)

workflow.add_edge(START, "intent")
workflow.add_conditional_edges(
    "intent", 
    route_after_intent, 
    {"rag": "rag", END: END}
)
workflow.add_edge("rag", "clinical")
workflow.add_edge("clinical", END)

# Grafı derle
triage_app = workflow.compile()
