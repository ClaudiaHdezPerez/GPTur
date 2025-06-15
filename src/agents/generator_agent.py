from .base_agent import BaseAgent
import streamlit as st

class GeneratorAgent(BaseAgent):
    def __init__(self, guide_agent, planner_agent):
        self.guide_agent = guide_agent
        self.planner_agent = planner_agent

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
            return {
                "dias": 5,
                "destino": "Cuba", 
                "presupuesto": 50,
                "intereses": prompt
            }

    def handle(self, task, context):
        prompt = task.get("prompt")
        self.guide_agent.update_beliefs(prompt, context)
    
        # Deliberar y actuar
        self.guide_agent.deliberate()
        response = self.guide_agent.act()
        
        # Si se detecta solicitud de itinerario
        if "itinerario" in prompt.lower() or "planear" in prompt.lower():
            preferences = self._extract_travel_params(prompt)
            response = self.planner_agent.create_itinerary(preferences)

        return response