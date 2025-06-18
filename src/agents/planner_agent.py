from .bdi_agent import BDIAgent
import math
import random
import re
import time
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict, Any, Union, final

@dataclass
class StochasticPrice:
    base_price: float
    std_dev: float = None
    
    def __post_init__(self):
        if self.std_dev is None:
            self.std_dev = self.base_price * 0.2
    
    def sample(self) -> float:
        return max(1.0, norm.rvs(loc=self.base_price, scale=self.std_dev))

@dataclass
class Place:
    name: str
    city: str
    cost: Union[float, StochasticPrice]
    final_cost: float
    rating: float
    type: str
    description: str = ""
    
    def get_cost(self) -> float:
        if isinstance(self.cost, StochasticPrice):
            self.final_cost = self.cost.sample()
        else:
            self.final_cost = self.cost
            
        return self.final_cost

class TravelPlannerAgent(BDIAgent):
    def __init__(self, vector_db):
        super().__init__("PlanificadorViajes", vector_db)
        
        # Definir deseos base del agente
        self.desires = [
            "crear_itinerario"
        ]
        
        # Definir planes disponibles
        self.plans = {
            "crear_itinerario": {
                "objetivo": "crear_itinerario",
                "precondiciones": ["destino_valido", "dias_validos", "presupuesto_valido"],
                "acciones": ["crear_itinerario"]
            }
        }
        
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
        
        percept = {"destination": destination}
        
        # Obtener lugares gastronómicos
        gastro_response = self.gastronomy_agent.action(percept)
        for place in self._parse_agent_response(gastro_response):
            places["gastronomicos"].append(Place(
                name=place["name"],
                city=destination,
                cost=StochasticPrice(base_price=place.get("cost", 20)),
                final_cost=0.0,
                rating=place.get("rating", 7),
                type="restaurant",
                description=place.get("description", "")
            ))

        # Obtener lugares nocturnos
        night_response = self.nightlife_agent.action(percept)
        for place in self._parse_agent_response(night_response):
            places["nocturnos"].append(Place(
                name=place["name"],
                city=destination,
                cost=StochasticPrice(base_price=place.get("cost", 30)),
                final_cost=0.0,
                rating=place.get("rating", 7),
                type="nightlife",
                description=place.get("description", "")
            ))

        # Obtener alojamientos
        lodging_response = self.lodging_agent.action(percept)
        for place in self._parse_agent_response(lodging_response):
            places["alojamientos"].append(Place(
                name=place["name"],
                city=destination,
                cost=StochasticPrice(base_price=place.get("cost", 50)),
                final_cost=0.0,
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
                budget_per_day: float, destination: str, max_iter: int = 1000,
                max_time: float = 120):  # 120 segundos = 2 minutos

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
                    day["desayuno"].get_cost() +
                    day["almuerzo"].get_cost() +
                    day["cena"].get_cost() +
                    day["noche"].get_cost() +
                    day["alojamiento"].get_cost()
                )
                if total_cost > budget_per_day:
                    return False
            return True

        # Inicialización
        print("Iniciando recocido simulado para planificar viaje...")
        current_sol = generate_initial_solution()
        while not is_valid_solution(current_sol):
            current_sol = generate_initial_solution()
            
        best_sol = current_sol.copy()
        best_rating = calculate_total_rating(current_sol)
        
        # Parámetros de recocido
        T = 100.0
        T_min = 0.1
        alpha = 0.99
        
        start_time = time.time()  # Registrar tiempo de inicio
        while T > T_min:
            for _ in range(max_iter):
                # Verificar si se excedió el tiempo límite
                if time.time() - start_time > max_time:
                    return best_sol  # Devolver la mejor solución encontrada hasta ahora
                    
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
            desayuno_cost = day['desayuno'].final_cost
            almuerzo_cost = day['almuerzo'].final_cost
            cena_cost = day['cena'].final_cost
            noche_cost = day['noche'].final_cost
            alojamiento_cost = day['alojamiento'].final_cost
            
            print(desayuno_cost, almuerzo_cost, cena_cost, noche_cost, alojamiento_cost)
            
            itinerary += f"🌅 Desayuno: {day['desayuno'].name} - ${desayuno_cost:.2f}\n"
            itinerary += f"🍽️ Almuerzo: {day['almuerzo'].name} - ${almuerzo_cost:.2f}\n"
            itinerary += f"🌙 Cena: {day['cena'].name} - ${cena_cost:.2f}\n"
            itinerary += f"🎭 Actividad Nocturna: {day['noche'].name} - ${noche_cost:.2f}\n"
            itinerary += f"🏨 Alojamiento: {day['alojamiento'].name} - ${alojamiento_cost:.2f}\n"
            
            total_day = (desayuno_cost + almuerzo_cost + 
                        cena_cost + noche_cost + 
                        alojamiento_cost)
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

        print("dias:", days)
        print("dest:", destination)
        print("budget:", budget)
        print("places:", places)
        
        # Ejecutar CSP con recocido simulado
        solution = self.simulated_annealing_csp(
            days=days,
            places=places,
            budget_per_day=budget,
            destination=destination
        )
        
        return self._format_itinerary(solution)

    def _is_achievable(self, plan) -> bool:
        """Verifica si un plan es alcanzable según las precondiciones"""
        for precondition in plan["precondiciones"]:
            if precondition == "destino_valido":
                if "destino" not in self.beliefs or not self.beliefs["destino"]:
                    return False
            elif precondition == "dias_validos":
                if "dias" not in self.beliefs or self.beliefs["dias"] <= 0:
                    return False
            elif precondition == "presupuesto_valido":
                if "presupuesto" not in self.beliefs or self.beliefs["presupuesto"] <= 0:
                    return False
        return True
    
    def _is_compatible(self, option) -> bool:
        return True
    
    def _is_plan_relevant(self, plan) -> bool:
        return True
        
    def _evaluate_intention(self, option) -> bool:
        """Evalúa si una opción debe convertirse en intención"""
        # Evaluar basado en el estado actual y recursos disponibles
        if option["objetivo"] == "crear_itinerario":
            return "destino" in self.beliefs
        return True
        
    def _get_next_action(self, intention):
        """Obtiene la siguiente acción para una intención"""
        if not intention.get("acciones"):
            return None
        return intention["acciones"][0]
        
    def _perform_action(self, action):
        """Ejecuta una acción específica"""
        if action == "crear_itinerario":
            return self.create_itinerary(self.beliefs)
        return None