from .base_agent import BaseAgent
import streamlit as st

class GeneratorAgent(BaseAgent):
    def __init__(self, guide_agent, planner_agent):
        self.guide_agent = guide_agent
        self.planner_agent = planner_agent

    def can_handle(self, task):
        return task.get("type") == "generate"

    def handle(self, task, context):
        prompt = task.get("prompt")
        self.guide_agent.update_beliefs(prompt, st.session_state.messages)
    
        # Deliberar y actuar
        self.guide_agent.deliberate()
        response = self.guide_agent.act()
        
        # Si se detecta solicitud de itinerario
        if "itinerario" in prompt.lower() or "planear" in prompt.lower():
            preferences = {
                "destino": "Cuba",  # Podrías extraer esto del prompt
                "dias": 5,  # Valor por defecto
                "intereses": prompt
            }
            response = self.planner_agent.create_itinerary(preferences)

        return response