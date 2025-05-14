from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain.retrievers import BM25Retriever
import streamlit as st  # Si usas Streamlit para secrets

class GapDetector:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.bm25_retriever = BM25Retriever.from_documents(self.vector_db.get())
        self.client = MistralClient(api_key=st.secrets.mistral_api_key)  # O desde .env
    
    def check_accuracy(self, query, response):
        # Paso 1: Buscar documentos relevantes
        relevant_docs = self.bm25_retriever.get_relevant_documents(query)
        
        # Paso 2: Crear prompt estructurado para Mistral
        verification_prompt = f"""
        Eres un validador de información turística. Evalúa si la respuesta dada necesita actualización 
        basándote en el contexto proporcionado. Responde **solo con una palabra**: 

        [RESPUESTA ACTUAL]
        {response}

        [CONTEXTO ACTUALIZADO]
        {relevant_docs[:2]}

        ¿La respuesta está desactualizada o es incompleta? Responde: ACTUALIZAR o OK.
        """
        
        # Paso 3: Llamar a Mistral AI
        messages = [
            ChatMessage(role="user", content=verification_prompt)
        ]
        
        try:
            api_response = self.client.chat(
                model="mistral-small",
                messages=messages,
                temperature=0.0  # Máxima precisión
            )
            verification_result = api_response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"Error al validar: {e}")
            verification_result = "OK"  # Default seguro
            
        # Paso 4: Devolver decisión
        return "ACTUALIZAR" in verification_result
    
    def identify_outdated_sources(self, query):
        relevant_docs = self.vector_db.similarity_search(query, k=2)
        return [doc.metadata['source'] for doc in relevant_docs]