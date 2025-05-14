import streamlit as st
from chatbot.core import CubaChatbot
from chatbot.gap_detector import GapDetector
from crawlers.dynamic_crawler import DynamicCrawler  # Asumiendo que tienes esta clase
import time

st.title("Asistente Turístico de Cuba 🇨🇺")

# Inicialización de componentes
chatbot = CubaChatbot()
detector = GapDetector(chatbot.vector_db)
updater = DynamicCrawler()  # Clase que maneja el re-crawling

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
    # Agregar mensaje de usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generar respuesta inicial
    response = chatbot.generate_response(prompt)
    
    # Verificar si necesita actualización
    if detector.check_accuracy(prompt, response):
        with st.status("🔄 Actualizando información...", expanded=True) as status:
            # 1. Identificar fuentes a actualizar
            sources = detector.identify_outdated_sources(prompt)
            
            # 2. Ejecutar crawler dinámico
            updater.update_sources(sources)
            
            # 3. Re-indexar la base de datos vectorial
            chatbot.vector_db.reload_data()
            
            # 4. Regenerar respuesta con nuevos datos
            response = chatbot.generate_response(prompt)
            
            status.update(label="✅ Actualización completada", state="complete")
    
    # Agregar respuesta final
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()