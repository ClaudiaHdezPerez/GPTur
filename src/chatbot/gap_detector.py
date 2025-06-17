from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_community.retrievers import BM25Retriever
import streamlit as st  # Si usas Streamlit para secrets
import json
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup

class GapDetector:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.bm25_retriever = BM25Retriever.from_documents(self.vector_db.get_documents())
        self.client = MistralClient(api_key="XEV0fCx3MqiG9HqVkGc4Hy5qyD3WwPHr")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _load_json_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def _save_json_file(self, filepath, data):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, ensure_ascii=False, indent=4, fp=f)

    def _fetch_webpage_info(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            return {
                "url": url,
                "title": soup.title.string if soup.title else url,
                "description": soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else "",
                "content": soup.get_text(strip=True),
                "attractions": [],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "source": url,
                    "crawl_date": datetime.now().isoformat(),
                    "language": "es"
                }
            }
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

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

    def identify_outdated_sources(self, query, city=None):
        # Si no se proporciona ciudad, intentar extraerla del query usando Mistral
        if city is None:
            city_prompt = f"""
            Extrae el nombre de la ciudad de Cuba mencionada en este texto. 
            Si no hay ninguna ciudad específica mencionada, responde con 'Cuba'.
            Responde SOLO con el nombre, nada más.
            
            Texto: {query}
            """
            messages = [ChatMessage(role="user", content=city_prompt)]
            try:
                api_response = self.client.chat(
                    model="mistral-small",
                    messages=messages,
                    temperature=0.0
                )
                city = api_response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error al extraer ciudad: {e}")
                city = "Cuba"

        # Obtener documentos relevantes
        relevant_docs = self.vector_db.similarity_search(query, k=2)
        docs_content = "\n".join([doc.page_content for doc in relevant_docs])

        # Crear prompt para Mistral AI solicitando enlaces actualizados
        prompt = f"""
        Basado en esta información turística sobre {city}:
        {docs_content}

        Proporciona 2-3 enlaces a fuentes oficiales o sitios web confiables con información actualizada.
        Responde en formato JSON exactamente así:
        [
            {{"url": "URL1", "title": "Breve título descriptivo"}},
            {{"url": "URL2", "title": "Breve título descriptivo"}}
        ]
        """

        messages = [ChatMessage(role="user", content=prompt)]

        try:
            # Obtener recomendaciones de Mistral
            api_response = self.client.chat(
                model="mistral-small",
                messages=messages,
                temperature=0.7
            )

            new_sources = json.loads(api_response.choices[0].message.content.strip())

            # Cargar archivos JSON existentes
            sources_file = os.path.join(self.base_dir, 'data', 'sources.json')
            normalized_file = os.path.join(self.base_dir, 'data', 'processed', 'normalized_data.json')

            existing_sources = self._load_json_file(sources_file)
            normalized_data = self._load_json_file(normalized_file)

            # Procesar cada nueva fuente
            for source in new_sources:
                # Verificar si la URL ya existe
                if not any(existing['url'] == source['url'] for existing in existing_sources):
                    # Agregar a sources.json
                    existing_sources.append(source)

                    # Obtener información detallada y agregar a normalized_data.json
                    webpage_info = self._fetch_webpage_info(source['url'])
                    if webpage_info:
                        webpage_info['city'] = city.lower()
                        normalized_data.append(webpage_info)

            # Guardar los archivos actualizados
            self._save_json_file(sources_file, existing_sources)
            self._save_json_file(normalized_file, normalized_data)

            return new_sources

        except Exception as e:
            print(f"Error al buscar fuentes actualizadas: {e}")
            return []