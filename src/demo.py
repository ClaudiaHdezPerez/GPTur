import streamlit as st
from chatbot import CubaChatbot

# Configuración de la página
st.set_page_config(
    page_title="Asistente de Turismo en Cuba",
    page_icon="🇨🇺",
    layout="centered"
)

# Inicializa el chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = CubaChatbot()

# Sidebar
st.sidebar.title("Configuración")
st.sidebar.markdown("### Personaliza tu experiencia:")
user_name = st.sidebar.text_input("Tu nombre", value="Viajer@")

# Interfaz principal
st.title("🇨🇺 Asistente de Turismo Cubano")
st.write("Pregúntame sobre lugares históricos, playas, o recomendaciones de viaje!")

# Historial de chat
for message in st.session_state.chatbot.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input del usuario
user_input = st.chat_input(f"Hola {user_name}, ¿qué quieres saber?")

if user_input:
    # Muestra input del usuario
    with st.chat_message("user"):
        st.write(user_input)
    
    # Genera respuesta
    with st.chat_message("assistant"):
        response = st.session_state.chatbot.generate_response(user_input)
        st.write(response)
    
    # Guarda en historial
    st.session_state.chatbot.add_to_history("user", user_input)
    st.session_state.chatbot.add_to_history("assistant", response)