from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import streamlit as st
from data.destinations import cuba_destinations

class CubaChatbot:
    def __init__(self, model="open-mixtral-8x7b"):
        self.model = model
        self.client = MistralClient(api_key="XEV0fCx3MqiG9HqVkGc4Hy5qyD3WwPHr")
        self.history = []
        
        # Prompt del sistema para contextualizar al chatbot
        self.system_prompt = """
        Eres un experto en turismo de Cuba. Responde preguntas sobre lugares, historia y recomendaciones.
        Usa solo la siguiente base de datos. Si no sabes algo, di 'No tengo información sobre eso'.
        
        Base de datos:
        {context}
        """.format(context=str(cuba_destinations))

    def generate_response(self, user_query):
        # Formatea el contexto para Mistral
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_query)
        ]
        
        # Genera la respuesta
        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0.3  # Para respuestas más precisas
        )
        
        return response.choices[0].message.content

    def add_to_history(self, role, content):
        self.history.append({"role": role, "content": content})