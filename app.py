import streamlit as st
from graph import triage_app
from agents.risk_model_agent import predict_risk
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(
    page_title="Sağlık Risk Değerlendirme Asistanı",
    page_icon="🏥",
    layout="wide"
)


def reset_graph_state():
    return {
        "chat_history": [], "patient_complaint": "",
        "is_clarified": False, "clarification_question": "",
        "medical_context": "", "final_decision": "",
        "extracted_symptoms": [], "next_node": ""
    }

with st.sidebar:
    st.markdown("### 👤 Hasta Profili")
    age = st.number_input("Yaş", 0, 120, 25)
    sex = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
    totchron = 1 if st.checkbox("Kronik hastalığım var") else 0
    st.divider()
    if st.button("🔄 Sohbeti Sıfırla", use_container_width=True):
        st.session_state.messages = []
        st.session_state.graph_state = reset_graph_state()
        st.rerun()

st.title("🏥 Akıllı Triyaj Asistanı")
st.caption("Yapay zeka destekli ön değerlendirme sistemi")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph_state" not in st.session_state:
    st.session_state.graph_state = reset_graph_state()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("risk_info"):
            risk = msg["risk_info"]
            level = risk.get("risk_level", 0)
            color = "red" if level == 2 else "orange" if level == 1 else "green"
            label = risk.get("risk_label_tr", "")
            conf = int(risk.get("confidence", 0) * 100)
            shap = ", ".join(risk.get("shap_explanation", []))
            rec = risk.get("recommendation", "")
            st.markdown(f":{color}[**Risk: {label} (%{conf} güven)**]")
            if shap:
                st.caption(f"Etkili faktörler: {shap}")
            if rec:
                st.info(rec)

if prompt := st.chat_input("Şikayetinizi buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analiz ediliyor..."):
            current_state = st.session_state.graph_state
            current_state["chat_history"].append(HumanMessage(content=prompt))
            if not current_state["patient_complaint"]:
                current_state["patient_complaint"] = prompt
            else:
                current_state["clarification_question"] = ""

            try:
                result = triage_app.invoke(current_state)
            except Exception as e:
                st.error(f"Sistem hatası: {e}")
                st.stop()

            original_complaint = current_state["patient_complaint"]
            chat_history = current_state["chat_history"]
            current_state.update(result)
            current_state["patient_complaint"] = original_complaint
            current_state["chat_history"] = chat_history
            st.session_state.graph_state = current_state

            ai_response = (
                current_state.get("final_decision") or
                current_state.get("clarification_question") or
                "Şikayetinizi aldım, devam edebiliriz."
            )
            st.markdown(ai_response)
            current_state["chat_history"].append(AIMessage(content=ai_response))

            risk_data = None
            if current_state.get("final_decision"):
                symptoms = current_state.get("extracted_symptoms", [])
                symptom_to_rfv = {
                    "chest pain": 105,
                    "shortness of breath": 153,
                    "palpitation": 154,
                    "headache": 121,
                    "abdominal pain": 141,
                    "nausea": 144,
                    "vomiting": 144,
                    "fever": 101,
                    "dizziness": 122,
                    "back pain": 192,
                    "cough": 153,
                    "fatigue": 101,
                    "chest tightness": 105,
                }

                active_rfv = 100
                if symptoms:
                    first_symptom = symptoms[0].lower().strip()
                    active_rfv = symptom_to_rfv.get(first_symptom, 100)

                patient_features = {
                    "age": age,
                    "sex": 1 if sex == "Kadın" else 0,
                    "rfv1_group": active_rfv,
                    "arrems": 0,
                    "totchron": totchron
                }

                try:
                    risk_data = predict_risk(patient_features)
                except Exception as e:
                    st.warning(f"Risk analizi çalıştırılamadı: {e}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "risk_info": risk_data
            })

            if current_state.get("final_decision"):
                st.session_state.graph_state = reset_graph_state()

st.divider()
st.warning("⚠️ Bu sistem tıbbi tanı koymaz. Acil durumda lütfen **112**'yi arayın.")
