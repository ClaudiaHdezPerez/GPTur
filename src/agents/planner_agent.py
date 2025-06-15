from .bdi_agent import BDIAgent
import math
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Union

@dataclass
class Place:
    name: str
    city: str
    cost: float
    rating: float
    type: str
    description: str = ""

class TravelPlannerAgent(BDIAgent):
    def __init__(self, vector_db):
        super().__init__("PlanificadorViajes", vector_db)
        self.desires = [
            "crear_itinerario",
            "optimizar_recursos"
        ]
        self.historic_agent = None
        self.gastronomy_agent = None
        self.lodging_agent = None
        self.nightlife_agent = None

    def set_specialized_agents(self, historic, gastronomy, lodging, nightlife):
        self.historic_agent = historic
        self.gastronomy_agent = gastronomy
        self.lodging_agent = lodging
        self.nightlife_agent = nightlife

    def _get_places_from_agents(self, destination: str) -> Dict[str, List[Place]]:
        places = {
            "gastronomicos": [],
            "nocturnos": [],
            "alojamientos": []
        }
        
        # Obtener lugares gastronómicos
        gastro_response = self.gastronomy_agent.get_recommendations(destination)
        for place in self._parse_agent_response(gastro_response):
            places["gastronomicos"].append(Place(
                name=place["name"],
                city=destination,
                cost=place.get("cost", 20),
                rating=place.get("rating", 7),
                type="restaurant",
                description=place.get("description", "")
            ))

        # Obtener lugares nocturnos
        night_response = self.nightlife_agent.get_recommendations(destination)
        for place in self._parse_agent_response(night_response):
            places["nocturnos"].append(Place(
                name=place["name"],
                city=destination,
                cost=place.get("cost", 30),
                rating=place.get("rating", 7),
                type="nightlife",
                description=place.get("description", "")
            ))

        # Obtener alojamientos
        lodging_response = self.lodging_agent.get_recommendations(destination)
        for place in self._parse_agent_response(lodging_response):
            places["alojamientos"].append(Place(
                name=place["name"],
                city=destination,
                cost=place.get("cost", 50),
                rating=place.get("rating", 7),
                type="lodging",
                description=place.get("description", "")
            ))

        return places

    def _parse_cost(self, cost_str: str) -> float:
        """Extrae y procesa el costo de un string, devolviendo un número"""
        if isinstance(cost_str, (int, float)):
            return float(cost_str)
            
        # Si es un string, extraer números
        numbers = re.findall(r'\d+', str(cost_str))
        if not numbers:
            return 25.0  # valor por defecto si no se encuentra ningún número
            
        # Si hay un rango (ej: "15-25"), tomar el promedio
        if len(numbers) >= 2:
            return (float(numbers[0]) + float(numbers[1])) / 2
            
        # Si hay un solo número
        return float(numbers[0])

    def _parse_agent_response(self, response: str) -> List[Dict[str, Any]]:
        """Convierte la respuesta del agente en una lista de diccionarios"""
        try:
            system_prompt = """Extrae la información de los lugares mencionados en el texto.
            Por cada lugar devuelve un diccionario con:
            - name: nombre del lugar
            - cost: costo estimado (como string con el formato "$X-Y USD" o "$X USD")
            - rating: valoración de 1-10 (como número)
            - description: breve descripción
            Responde como lista de diccionarios."""
            
            parsed = self.client.chat(
                model="mistral-small",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": response}
                ]
            )
            result = eval(parsed.choices[0].message.content)
            
            # Procesar los costos a números
            for item in result:
                if "cost" in item:
                    item["cost"] = self._parse_cost(item["cost"])
                if "rating" in item:
                    item["rating"] = float(item["rating"])
            
            return result
        except:
            # Valores por defecto si hay error
            return [
                {"name": "Lugar Genérico", "cost": 25.0, "rating": 7.0, "description": "Lugar típico"}
            ]

    def simulated_annealing_csp(self, days: int, places: Dict[str, List[Place]], 
                budget_per_day: float, destination: str, max_iter: int = 1000):
    
        def generate_initial_solution():
            solution = []
            for _ in range(days):
                day = {
                    "desayuno": random.choice(places["gastronomicos"]),
                    "almuerzo": random.choice(places["gastronomicos"]),
                    "cena": random.choice(places["gastronomicos"]),
                    "noche": random.choice(places["nocturnos"]),
                    "alojamiento": random.choice(places["alojamientos"])
                }
                solution.append(day)
            return solution

        def calculate_total_rating(sol):
            return sum(
                day["desayuno"].rating +
                day["almuerzo"].rating +
                day["cena"].rating +
                day["noche"].rating +
                day["alojamiento"].rating
                for day in sol
            )

        def generate_neighbor(sol):
            day = random.randint(0, len(sol)-1)
            activity = random.choice(["desayuno", "almuerzo", "cena", "noche", "alojamiento"])
            
            new_sol = [d.copy() for d in sol]
            if activity in ["desayuno", "almuerzo", "cena"]:
                new_sol[day][activity] = random.choice(places["gastronomicos"])
            elif activity == "noche":
                new_sol[day][activity] = random.choice(places["nocturnos"])
            else:
                new_sol[day][activity] = random.choice(places["alojamientos"])
            
            return new_sol

        def is_valid_solution(sol):
            for day in sol:
                total_cost = (
                    day["desayuno"].cost +
                    day["almuerzo"].cost +
                    day["cena"].cost +
                    day["noche"].cost +
                    day["alojamiento"].cost
                )
                if total_cost > budget_per_day:
                    return False
            return True

        # Inicialización
        current_sol = generate_initial_solution()
        while not is_valid_solution(current_sol):
            current_sol = generate_initial_solution()
            
        best_sol = current_sol.copy()
        best_rating = calculate_total_rating(current_sol)
        
        # Parámetros de recocido
        T = 100.0
        T_min = 0.1
        alpha = 0.99
        
        while T > T_min:
            for _ in range(max_iter):
                neighbor_sol = generate_neighbor(current_sol)
                if not is_valid_solution(neighbor_sol):
                    continue
                    
                current_rating = calculate_total_rating(current_sol)
                neighbor_rating = calculate_total_rating(neighbor_sol)
                
                delta = neighbor_rating - current_rating
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_sol = neighbor_sol
                    if neighbor_rating > best_rating:
                        best_sol = neighbor_sol.copy()
                        best_rating = neighbor_rating
            
            T *= alpha
        
        return best_sol

    def _format_itinerary(self, solution: List[Dict[str, Place]]) -> str:
        itinerary = "🌟 Itinerario de Viaje 🌟\n\n"
        
        for i, day in enumerate(solution, 1):
            itinerary += f"📅 Día {i}:\n"
            itinerary += f"🌅 Desayuno: {day['desayuno'].name} - ${day['desayuno'].cost}\n"
            itinerary += f"🍽️ Almuerzo: {day['almuerzo'].name} - ${day['almuerzo'].cost}\n"
            itinerary += f"🌙 Cena: {day['cena'].name} - ${day['cena'].cost}\n"
            itinerary += f"🎭 Actividad Nocturna: {day['noche'].name} - ${day['noche'].cost}\n"
            itinerary += f"🏨 Alojamiento: {day['alojamiento'].name} - ${day['alojamiento'].cost}\n"
            
            total_day = (day['desayuno'].cost + day['almuerzo'].cost + 
                        day['cena'].cost + day['noche'].cost + 
                        day['alojamiento'].cost)
            itinerary += f"💰 Total del día: ${total_day}\n\n"
            
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": """Como experto en turismo, genera una descripción detallada y atractiva 
                del itinerario proporcionado. de forma rápida"""},
                {"role": "user", "content": f"Por favor, genera un itinerario atractivo basado en esta información: {itinerary}"}
            ]
        )
        
        formatted_itinerary = response.choices[0].message.content
        
        return formatted_itinerary

    def create_itinerary(self, preferences: Dict[str, Any]) -> str:
        destination = preferences.get("destino", "Cuba")
        days = preferences.get("dias", 5)
        budget = preferences.get("presupuesto", 50)
        
        # Obtener lugares de los agentes especializados
        places = self._get_places_from_agents(destination)
        
        # Ejecutar CSP con recocido simulado
        solution = self.simulated_annealing_csp(
            days=days,
            places=places,
            budget_per_day=budget,
            destination=destination
        )
        
        return self._format_itinerary(solution)