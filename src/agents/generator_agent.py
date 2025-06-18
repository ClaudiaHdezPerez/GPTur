import json
import logging
from .base_agent import BaseAgent
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@staticmethod
def _convert_docs_to_string(documents):
    # Concatenate page_content from all documents
    combined_content = ""
    for doc in documents:
        try:
            # Parse the page_content which is a JSON string
            content_json = json.loads(doc.page_content)
            page_content = content_json.get("page_content", "")
            if page_content:
                combined_content += page_content + " "
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing document JSON: {e}")
            # Try to use the raw page_content if JSON parsing fails
            if hasattr(doc, 'page_content'):
                combined_content += str(doc.page_content) + " "
    return combined_content.strip()


class GeneratorAgent(BaseAgent):
    def __init__(self, guide_agent, planner_agent):
        self.guide_agent = guide_agent
        self.planner_agent = planner_agent

    def can_handle(self, task):
        return task.get("type") == "generate"

    def _extract_travel_params(self, prompt):
        # Usar LLM para extraer parámetros
        system_prompt = """Extrae los siguientes parámetros del texto del usuario y responde SOLO con un JSON válido que contenga exactamente estos campos:
        - días (número entre 1-15, default 5)
        - destino (ciudad o país, default Cuba)
        - presupuesto (número en USD, default 50)
        
        Formato de respuesta (exactamente así):
        {"dias": X, "destino": "Y", "presupuesto": Z}"""
        
        response = self.guide_agent.client.chat(
            model="mistral-small",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            content = response.choices[0].message.content.strip()
            # Log the raw response for debugging
            logger.info(f"LLM Response: {content}")
            
            # Try to find and extract just the JSON part if there's extra text
            try:
                # Find the first { and last }
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    params = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON object found", content, 0)
            except json.JSONDecodeError:
                # If that fails, try parsing the whole response
                params = json.loads(content)
                
            return {
                "dias": params.get("dias", 5),
                "destino": params.get("destino", "Cuba"),
                "presupuesto": params.get("presupuesto", 50),
                "intereses": prompt
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            logger.error(f"Raw content: {response.choices[0].message.content}")
            return {                
                "dias": 5,
                "destino": "Cuba", 
                "presupuesto": 300,
                "intereses": prompt
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {                
                "dias": 5,
                "destino": "Cuba", 
                "presupuesto": 50,
                "intereses": prompt
            }
            
    def handle(self, task, context):
        prompt = task.get("prompt")
        
        # Convert context documents to string if context is a list of documents
        context_text = _convert_docs_to_string(context) if isinstance(context, list) else str(context)
        
        # Detect user intention with MistralAI client
        system_prompt = """Analiza la intención del usuario y clasifícala en una de estas categorías:
        - PLANNING: Si el usuario quiere planear un viaje o crear un itinerario
        - INFO: Si el usuario busca información general, recomendaciones o respuestas sobre lugares
        Si tienes dudas sobre la intención, responde con INFO.
        Responde únicamente con la categoría: PLANNING o INFO"""
        
        intent_response = self.guide_agent.client.chat(
            model="mistral-small",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        intent = intent_response.choices[0].message.content.strip()
        
        # If the intention is planner, use the planner agent
        if intent == "PLANNING":
            preferences = self._extract_travel_params(prompt)
            return self.planner_agent.action(preferences)
        
        # By default, use the guide agent
        return self.guide_agent.action((prompt, context_text))