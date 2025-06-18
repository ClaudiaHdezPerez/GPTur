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
from agents.gastronomy_agent import GastronomyAgent
from agents.historic_agent import HistoricAgent
from agents.lodging_agent import LodgingAgent
from agents.nightlife_agent import NightlifeAgent
from pathlib import Path

# Configuración de la página
logo_path = Path(__file__).parent / "logo" / "GPTur.png"
st.set_page_config(
    page_title="GPTur",
    page_icon=str(logo_path)
)

st.title("GPTur - Asistente Turístico de Cuba")

# Inicialización de componentes
if "chatbot" not in st.session_state:
    st.session_state.chatbot = CubaChatbot()
    
if not st.session_state.chatbot.vector_db.get_documents():
    print("\nCargando datos iniciales...\n")
    try:
        st.session_state.chatbot.vector_db.reload_data()
        if not st.session_state.chatbot.vector_db.get_documents():
            st.error("Error: No se pudieron cargar los datos iniciales")
            st.stop()
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.stop()

detector = GapDetector(st.session_state.chatbot.vector_db)
updater = DynamicCrawler()

# Inicialización de agentes principales
guide_agent = GuideAgent(st.session_state.chatbot.vector_db)
planner_agent = TravelPlannerAgent(st.session_state.chatbot.vector_db)

# Inicialización de agentes especializados
historic_agent = HistoricAgent("HistoricAgent", st.session_state.chatbot.vector_db)
gastronomy_agent = GastronomyAgent("GastronomyAgent", st.session_state.chatbot.vector_db)
lodging_agent = LodgingAgent("LodgingAgent", st.session_state.chatbot.vector_db)
nightlife_agent = NightlifeAgent("NightlifeAgent", st.session_state.chatbot.vector_db)

# Configurar agentes especializados en el planner
planner_agent.set_specialized_agents(
    historic=historic_agent,
    gastronomy=gastronomy_agent,
    lodging=lodging_agent,
    nightlife=nightlife_agent
)

# Inicialización de agentes restantes
retriever_agent = RetrieverAgent(st.session_state.chatbot.vector_db)
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
            st.session_state.chatbot.vector_db.update_index()
            # Regenerar respuesta
            response = manager.dispatch(generate_task, context)
            status.update(label="✅ Actualización completada", state="complete")

    # Agregar respuesta final
    if hasattr(response, "choices"):
        response_text = " ".join([choice.message.content for choice in response.choices])
    else:
        response_text = str(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()