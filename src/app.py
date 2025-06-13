import streamlit as st
from chatbot.core import CubaChatbot
from chatbot.gap_detector import GapDetector
from crawlers.dynamic_crawler import DynamicCrawler
from agents.retriever_agent import RetrieverAgent
from agents.generator_agent import GeneratorAgent
from agents.gap_detector_agent import GapDetectorAgent
from agents.updater_agent import UpdaterAgent
from agents.agent_manager import AgentManager
from agents.guide_agent import GuideAgent
from agents.planner_agent import TravelPlannerAgent
import time

st.title("Asistente Turístico de Cuba 🇨🇺")

# Inicialización de componentes
chatbot = CubaChatbot()
if not chatbot.vector_db.get_documents():
    print("\nCargando datos iniciales...\n")
    try:
        chatbot.vector_db.reload_data()
        if not chatbot.vector_db.get_documents():
            st.error("Error: No se pudieron cargar los datos iniciales")
            st.stop()
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.stop()

detector = GapDetector(chatbot.vector_db)
updater = DynamicCrawler()
guide_agent = GuideAgent(chatbot.vector_db)
planner_agent = TravelPlannerAgent(chatbot.vector_db)

# Inicialización de agentes
retriever_agent = RetrieverAgent(chatbot.vector_db)
generator_agent = GeneratorAgent(guide_agent, planner_agent)
gap_detector_agent = GapDetectorAgent(detector)
updater_agent = UpdaterAgent(updater)

# Inicialización del manager
manager = AgentManager([
    retriever_agent,
    generator_agent,
    gap_detector_agent,
    updater_agent
])

# Estado de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = []
if "update_triggered" not in st.session_state:
    st.session_state.update_triggered = False

# Mostrar historial de mensajes
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Procesar input del usuario
if prompt := st.chat_input("Pregunta sobre lugares turísticos"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Recuperar contexto (opcional, si lo usas)
    retrieval_task = {"type": "retrieve", "query": prompt}
    context = manager.dispatch(retrieval_task, {})

    # Generar respuesta
    generate_task = {"type": "generate", "prompt": prompt}
    response = manager.dispatch(generate_task, context)

    # Verificar si necesita actualización
    detect_task = {"type": "detect_gap", "prompt": prompt, "response": response}
    needs_update = manager.dispatch(detect_task, context)

    if needs_update:
        with st.status("🔄 Actualizando información...", expanded=True) as status:
            # Identificar fuentes a actualizar
            sources = detector.identify_outdated_sources(prompt)
            update_task = {"type": "update_sources", "sources": sources}
            manager.dispatch(update_task, context)
            chatbot.vector_db.reload_data()
            # Regenerar respuesta
            response = manager.dispatch(generate_task, context)
            status.update(label="✅ Actualización completada", state="complete")

    # Agregar respuesta final
    if hasattr(response, "choices"):
        response_text = " ".join([choice.message.content for choice in response.choices])
    else:
        response_text = str(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()