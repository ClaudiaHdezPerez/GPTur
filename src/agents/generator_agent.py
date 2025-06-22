import json
import logging
from .base_agent import BaseAgent
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@staticmethod
def _convert_docs_to_string(documents):
    """
    Convert a list of documents into a single concatenated string.

    Args:
        documents (list): List of document objects containing page_content

    Returns:
        str: Combined string of all document contents
    """
    combined_content = ""
    for doc in documents:
        try:
            content_json = json.loads(doc.page_content)
            page_content = content_json.get("page_content", "")
            if page_content:
                combined_content += page_content + " "
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing document JSON: {e}")
            if hasattr(doc, 'page_content'):
                combined_content += str(doc.page_content) + " "
    return combined_content.strip()


class GeneratorAgent(BaseAgent):
    """
    An agent responsible for generating responses and handling different types of user queries.
    Coordinates between guide and planner agents based on user intent.
    """

    def __init__(self, guide_agent, planner_agent):
        """
        Initialize the generator agent with its required dependencies.

        Args:
            guide_agent: Agent handling general tourist information queries
            planner_agent: Agent handling travel planning queries
        """
        self.guide_agent = guide_agent
        self.planner_agent = planner_agent

    def can_handle(self, task):
        """
        Check if the agent can handle the given task.

        Args:
            task (dict): The task to be evaluated

        Returns:
            bool: True if the task is of type 'generate', False otherwise
        """
        return task.get("type") == "generate"

    def _extract_travel_params(self, prompt):
        """
        Extract travel parameters from user prompt using LLM.

        Args:
            prompt (str): User's input text

        Returns:
            dict: Extracted parameters including days, destination, budget and interests
                 with default values if extraction fails
        """
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
            logger.info(f"LLM Response: {content}")
            
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    params = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON object found", content, 0)
            except json.JSONDecodeError:
                params = json.loads(content)
                
            return {
                "dias": params.get("dias", 5),
                "destino": params.get("destino", "Cuba"),
                "presupuesto": params.get("presupuesto", 100),
                "intereses": prompt
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            logger.error(f"Raw content: {response.choices[0].message.content}")
            return {                
                "dias": 5,
                "destino": "Cuba", 
                "presupuesto": 100,
                "intereses": prompt
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {                
                "dias": 5,
                "destino": "Cuba", 
                "presupuesto": 100,
                "intereses": prompt
            }
            
    def handle(self, task, context):
        """
        Process the task and generate appropriate response using either guide or planner agent.

        Args:
            task (dict): The task containing the prompt to process
            context (dict): Additional context information

        Returns:
            str: Generated response from either guide or planner agent
        """
        print("\nGeneratorAgent.handle")
        prompt = task.get("prompt")
        
        context_text = _convert_docs_to_string(context) if isinstance(context, list) else str(context)
        
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
        
        if intent == "PLANNING":
            preferences = self._extract_travel_params(prompt)
            return self.planner_agent.action(preferences)
        
        return self.guide_agent.action((prompt, context_text))