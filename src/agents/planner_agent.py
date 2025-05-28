from .bdi_agent import BDIAgent

class TravelPlannerAgent(BDIAgent):
    def __init__(self, vector_db):
        super().__init__("PlanificadorViajes", vector_db)
        self.desires = [
            "crear_itinerario",
            "optimizar_recursos"
        ]

    def deliberate(self):
        if "nuevo_usuario" in self.beliefs:
            self.intentions = ["solicitar_preferencias"]
        elif "preferencias_usuario" in self.beliefs:
            self.intentions = ["generar_itinerario"]

    def plan(self):
        new_intentions = []
        for intention in self.intentions:
            if intention == "generar_itinerario":
                new_intentions.extend([
                    "consultar_lugares_relevantes",
                    "calcular_ruta_optima",
                    "verificar_disponibilidad"
                ])
        self.intentions = new_intentions

    def create_itinerary(self, preferences):
        context = self.vector_db.similarity_search(
            f"lugares turísticos en {preferences['destino']} para {preferences['intereses']}"
        )
        response = self.client.chat(
            model="mistral-medium",
            messages=[{
                "role": "user",
                "content": f"Crea un itinerario de {preferences['dias']} días basado en: {context}"
            }]
        )
        return response.choices[0].message.content