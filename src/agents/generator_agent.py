from .base_agent import BaseAgent
import streamlit as st

class GeneratorAgent(BaseAgent):
    def __init__(self, guide_agent, planner_agent):
        self.guide_agent = guide_agent
        self.planner_agent = planner_agent

    def _convert_docs_to_string(self, documents):
        # Concatenate page_content from all documents
        combined_content = ""
        for doc in documents:
            try:
                # Parse the page_content which is a JSON string
                content_json = eval(doc.page_content)
                page_content = content_json.get("page_content", "")
                if page_content:
                    combined_content += page_content + " "
            except:
                # If there's an error parsing, skip this document
                continue
        return combined_content.strip()

    def can_handle(self, task):
        return task.get("type") == "generate"

    def _extract_travel_params(self, prompt):
        # Usar LLM para extraer parámetros
        system_prompt = """Extrae los siguientes parámetros del texto del usuario:
        - días (número entre 1-15, default 5)
        - destino (ciudad o país, default Cuba)
        - presupuesto (número en USD, default 50)
        Responde en formato JSON como: {"dias": X, "destino": "Y", "presupuesto": Z}"""
        
        response = self.guide_agent.client.chat(
            model="mistral-small",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        try:
            params = eval(response.choices[0].message.content)
            return {
                "dias": params.get("dias", 5),
                "destino": params.get("destino", "Cuba"),
                "presupuesto": params.get("presupuesto", 50),
                "intereses": prompt
            }
        except:
            print("no pudo obtener requisitos del itinerario")
            return {
                "dias": 5,
                "destino": "Cuba", 
                "presupuesto": 50,
                "intereses": prompt
            }
            
    def handle(self, task, context):
        prompt = task.get("prompt")
        
        # Convert context documents to string if context is a list of documents
        context_text = self._convert_docs_to_string(context) if isinstance(context, list) else str(context)
        
        # Detect user intention with MistralAI client
        system_prompt = """Analiza la intención del usuario y clasifícala en una de estas categorías:
        - PLANNING: Si el usuario quiere planear un viaje, crear un itinerario o programar actividades
        - INFO: Si el usuario busca información general, recomendaciones o respuestas sobre lugares
        Responde únicamente con la categoría: PLANNING o INFO"""
        
        intent_response = self.guide_agent.client.chat(
            model="mistral-small",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        intent = intent_response.choices[0].message.content.strip()
        
        print(f"Detected intent: {intent}")
        print(f"Context for handling: {context_text}")
        
        # If the intention is planner, use the planner agent
        if intent == "PLANNING":
            preferences = self._extract_travel_params(prompt)
            print(f"\npreferencias del usuario: {preferences}\n")
            return self.planner_agent.create_itinerary(preferences)
        
        # By default, use the guide agent
        return self.guide_agent.action((prompt, context_text))