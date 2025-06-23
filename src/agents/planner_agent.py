from .bdi_agent import BDIAgent
import math
import random
import re
import time
import numpy as np
from scipy.stats import truncnorm
from dataclasses import dataclass
from typing import List, Dict, Any, Union

@dataclass
class StochasticPrice:
    """
    A class representing a price with stochastic variation.
    Used for simulating real-world price fluctuations in travel planning.
    """
    base_price: float
    std_dev: float = None
    lower: float = None
    upper: float = None
    
    def __post_init__(self):
        if self.std_dev is None:
            self.std_dev = self.base_price * 0.2
            self.lower = (self.base_price * 0.5 - self.base_price) / self.std_dev
            self.upper = (self.base_price * 2 - self.base_price) / self.std_dev
    
    def sample(self) -> float:
        """
        Generate a random price sample based on normal distribution.

        Returns:
            float: A random price value, minimum 1.0
        """
        return truncnorm.rvs(self.lower, self.upper, loc=self.base_price, scale=self.std_dev)

@dataclass
class Place:
    """
    A class representing a tourist destination or venue with its attributes.
    Includes name, location, cost, rating and other relevant information.
    """
    name: str
    city: str
    cost: Union[float, StochasticPrice]
    final_cost: float
    rating: float
    type: str
    description: str = ""
    
    def get_cost(self) -> float:
        """
        Calculate the final cost of the place, either static or stochastic.

        Returns:
            float: The final calculated cost
        """
        if isinstance(self.cost, StochasticPrice):
            self.final_cost = self.cost.sample()
        else:
            self.final_cost = self.cost
            
        return self.final_cost

class TravelPlannerAgent(BDIAgent):
    def __init__(self, vector_db):
        super().__init__("PlanificadorViajes", vector_db)
        
        self.desires = [
            "crear_itinerario"
        ]
        
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
        """
        Set up specialized agents for different aspects of travel planning.

        Args:
            historic: Agent for historical sites and attractions
            gastronomy: Agent for restaurants and dining
            lodging: Agent for accommodations
            nightlife: Agent for nightlife and entertainment
        """
        self.historic_agent = historic
        self.gastronomy_agent = gastronomy
        self.lodging_agent = lodging
        self.nightlife_agent = nightlife

    def _get_places_from_agents(self, destination: str) -> Dict[str, List[Place]]:
        """
        Collect place recommendations from all specialized agents.

        Args:
            destination (str): Target destination

        Returns:
            Dict[str, List[Place]]: Places categorized by type (gastronomy, nightlife, lodging)
        """
        places = {
            "gastronomicos": [],
            "nocturnos": [],
            "alojamientos": []
        }
        
        percept = {"destination": destination}
        
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
        """
        Extract and process cost value from a string representation.

        Args:
            cost_str (str): String containing cost information

        Returns:
            float: Processed cost value
        """
        if isinstance(cost_str, (int, float)):
            return float(cost_str)
            
        numbers = re.findall(r'\d+', str(cost_str))
        if not numbers:
            return 25.0
            
        if len(numbers) >= 2:
            return (float(numbers[0]) + float(numbers[1])) / 2
            
        return float(numbers[0])

    def _parse_agent_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse agent responses into structured data.

        Args:
            response (str): Raw response from an agent

        Returns:
            List[Dict[str, Any]]: List of parsed place information
        """
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
                    {"role": "user", "content": response[:130000]}
                ]
            )
            result = eval(parsed.choices[0].message.content)
            
            for item in result:
                if "cost" in item:
                    item["cost"] = self._parse_cost(item["cost"])
                if "rating" in item:
                    item["rating"] = float(item["rating"])
            
            return result
        except:
            return [
                {"name": "Lugar Genérico", "cost": 25.0, "rating": 7.0, "description": "Lugar típico"}
            ]
    
    def get_price_means(self, sol, n):
        means = []
        for day in sol:
            day_costs = { key: 0 for key in day }
            for _ in range(n):
                day_costs["desayuno"] += day["desayuno"].get_cost()
                day_costs["almuerzo"] += day["almuerzo"].get_cost()
                day_costs["cena"] += day["cena"].get_cost()
                day_costs["noche"] += day["noche"].get_cost()
                day_costs["alojamiento"] += day["alojamiento"].get_cost()
            
            day_costs = { key: day_costs[key] / 30 for key in day_costs }
            means.append(sum(day_costs.values()))
        
        return means

    def evaluate(self, sol, n=30):
        ratings = [
            day["desayuno"].rating +
            day["almuerzo"].rating +
            day["cena"].rating +
            day["noche"].rating +
            day["alojamiento"].rating
            for day in sol
        ]
        
        price_means = self.get_price_means(sol, n)
        
        return sum([ratings[i] / price_means[i] for i in range(len(sol))])
    
    def is_valid_solution(self, sol, budget_per_day):
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
            
    def simulated_annealing_csp(
        self, days: int, places: Dict[str, List[Place]], 
        budget_per_day: float, destination: str, max_iter: int = 1000,
        max_time: float = 180, T: float = 100.0, alpha: float = 0.99, 
        T_min: float = 0.1
    ):
        """
        Generate optimized travel itinerary using simulated annealing algorithm.

        Args:
            days (int): Number of days for the itinerary
            places (Dict[str, List[Place]]): Available places by category
            budget_per_day (float): Maximum daily budget
            destination (str): Travel destination
            max_iter (int, optional): Maximum iterations per temperature. Defaults to 1000
            max_time (float, optional): Maximum execution time in seconds. Defaults to 120

        Returns:
            List[Dict[str, Place]]: Optimized itinerary for the entire trip
        """

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

        def _is_valid_solution(sol):
            return self.is_valid_solution(sol, budget_per_day)

        print("Iniciando recocido simulado para planificar viaje...")
        current_sol = generate_initial_solution()
        start_time = time.time()
        while not _is_valid_solution(current_sol):
            if (time.time() - start_time) > 300:
                return None
            current_sol = generate_initial_solution()
            
        best_sol = current_sol.copy()
        best_rating = self.evaluate(current_sol)
        
        start_time = time.time()
        while T > T_min:
            for _ in range(max_iter):
                current_time = time.time() - start_time
                if current_time > max_time:
                    return best_sol
                    
                neighbor_sol = generate_neighbor(current_sol)
                if not _is_valid_solution(neighbor_sol):
                    continue
                    
                current_rating = self.evaluate(current_sol)
                neighbor_rating = self.evaluate(neighbor_sol)
                
                delta = neighbor_rating - current_rating
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_sol = neighbor_sol
                    if neighbor_rating > best_rating:
                        best_sol = neighbor_sol.copy()
                        best_rating = neighbor_rating
            
            T *= alpha
    
        return best_sol

    def _format_itinerary(self, solution: List[Dict[str, Place]]) -> str:
        """
        Format the generated itinerary into a user-friendly string.

        Args:
            solution (List[Dict[str, Place]]): The optimized itinerary

        Returns:
            str: Formatted itinerary with detailed descriptions
        """
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
                {"role": "system", "content": """Como experto en turismo, genera una descripción detallada, atractiva y en buen formato
                del itinerario proporcionado. de forma rápida"""},
                {"role": "user", "content": f"Por favor, genera un itinerario atractivo basado en esta información: {itinerary}"}
            ]
        )
        
        formatted_itinerary = response.choices[0].message.content
        
        return formatted_itinerary

    def create_itinerary(self, preferences: Dict[str, Any]) -> str:
        """
        Create a complete travel itinerary based on user preferences.

        Args:
            preferences (Dict[str, Any]): User preferences including destination, days, and budget

        Returns:
            str: Complete formatted itinerary
        """
        destination = preferences.get("destino", "Cuba")
        days = preferences.get("dias", 5)
        budget = preferences.get("presupuesto", 100)
                
        places = self._get_places_from_agents(destination)

        print("dias:", days)
        print("dest:", destination)
        print("budget:", budget)
        print("places:", places)
        
        solution = self.simulated_annealing_csp(
            days=days,
            places=places,
            budget_per_day=budget,
            destination=destination
        )
        
        if not solution:
            return "Lo siento, no se pudo generar un itinerario con las condiciones dadas."
        
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