from .bdi_agent import BDIAgent
from .blackboard import Blackboard
from datetime import datetime, time

class NightlifeAgent(BDIAgent):
    def __init__(self, name, vector_db=None):
        super().__init__(name, vector_db)
        self.specialization = "nightlife"
        self.blackboard = Blackboard()
        
        self.beliefs = {
            "venues": {},
            "venue_types": [
                "bar",
                "club",
                "live_music",
                "dance_hall",
                "cultural_center",
                "cafe"
            ],
            "music_types": [
                "traditional",
                "salsa",
                "jazz",
                "contemporary",
                "mixed"
            ],
            "price_ranges": ["economic", "moderate", "luxury"],
            "operating_hours": {
                "standard": {
                    "open": time(20, 0), 
                    "close": time(2, 0) 
                },
                "late": {
                    "open": time(22, 0),
                    "close": time(6, 0)
                }
            },
            "current_query": None,
            "destination": None
        }
        
        self.desires = [
            "process_user_query",
            "recommend_nightlife"
        ]
        
        self.plans = {
            "process_user_query": {
                "objetivo": "search_venues",
                "precondiciones": ["has_query"],
                "acciones": ["search_nightlife"]
            },
            "recommend_nightlife": {
                "objetivo": "provide_recommendations",
                "precondiciones": ["has_destination"],
                "acciones": ["get_recommendations"]
            }
        }
        
    def _is_plan_relevant(self, plan) -> bool:
        """
        Check if a plan is relevant for the current state.

        Args:
            plan (dict): The plan to evaluate

        Returns:
            bool: True if the plan's objective matches current beliefs, False otherwise
        """
        if plan["objetivo"] == "search_venues":
            return "current_query" in self.beliefs and self.beliefs["current_query"] is not None
        elif plan["objetivo"] == "provide_recommendations":
            return "destination" in self.beliefs and self.beliefs["destination"] is not None
        return False

    def _check_precondition(self, precondition) -> bool:
        """
        Verify if a specific precondition is met.

        Args:
            precondition (str): The precondition to check

        Returns:
            bool: True if precondition is met, False otherwise
        """
        if precondition == "has_query":
            return "current_query" in self.beliefs and self.beliefs["current_query"] is not None
        elif precondition == "has_destination":
            return "destination" in self.beliefs and self.beliefs["destination"] is not None
        return False

    def _is_achievable(self, plan) -> bool:
        """Verifica si un plan es alcanzable según las precondiciones"""
        return all(self._check_precondition(pre) for pre in plan["precondiciones"])

    def _is_compatible(self, plan) -> bool:
        """Verifica si un plan es compatible con las intenciones actuales"""
        return True

    def _get_next_action(self, intention) -> str:
        """Determina la siguiente acción para una intención"""
        if not intention.get("acciones"):
            return None
        return intention["acciones"][0]

    def _perform_action(self, action):
        """Ejecuta una acción específica"""
        if action == "search_nightlife":
            query = self.beliefs["current_query"]
            return self.search_nightlife(query)
            
        elif action == "get_recommendations":
            return self.get_recommendations(self.beliefs["destination"])

    def search_nightlife(self, query, relevant_docs=None):
        """
        Search for nightlife venues matching the provided query.

        Args:
            query (str): The search query string
            relevant_docs (list, optional): Pre-filtered relevant documents

        Returns:
            str: Formatted string containing nightlife venue search results
        """
        if relevant_docs is None:
            relevant_docs = self.vector_db.similarity_search(
                self._build_nightlife_query(query)
            )
            
        processed_results = []
        for doc in relevant_docs:
            if any(venue in doc.page_content.lower() for venue in self.beliefs["venue_types"]):
                processed_results.append(doc.page_content)
                
        classified_venues = self._classify_venues(processed_results)
        
        system_prompt = """Extract the destination/location from this query. Return ONLY the location name, nothing else.
        If no location is found, return 'unknown'."""
        
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        location = response.choices[0].message.content.strip()
        
        if location and location.lower() != 'unknown':
            if location not in self.beliefs["venues"]:
                self.beliefs["venues"][location] = classified_venues
            
        return self._format_venue_results(classified_venues)

    def _build_nightlife_query(self, location, filters=None):
        """
        Build an optimized search query based on location and filters.

        Args:
            location (str): The target location for venue search
            filters (dict, optional): Additional search filters (venue type, music, price)

        Returns:
            str: The constructed search query
        """
        query_parts = [f"nightlife venues and entertainment in {location}"]
        
        if filters:
            for filter_type, value in filters.items():
                if filter_type == "venue" and value in self.beliefs["venue_types"]:
                    query_parts.append(f"focusing on {value}")
                elif filter_type == "music" and value in self.beliefs["music_types"]:
                    query_parts.append(f"with {value} music")
                elif filter_type == "price" and value in self.beliefs["price_ranges"]:
                    query_parts.append(f"with {value} prices")
                    
        return " ".join(query_parts)

    def _classify_venues(self, results):
        """
        Classify nightlife venues by their type from search results.

        Args:
            results (list): List of venue search results

        Returns:
            dict: Venues classified by type (bars, clubs, etc.)
        """
        classified = {
            "bars": [],
            "clubs": [],
            "live_music": [],
            "dance_halls": [],
            "cultural_centers": [],
            "cafes": []
        }
        
        for result in results:
            result_lower = result.lower()
            if "bar" in result_lower:
                classified["bars"].append(result)
            elif "club" in result_lower or "disco" in result_lower:
                classified["clubs"].append(result)
            elif "music" in result_lower or "música" in result_lower:
                classified["live_music"].append(result)
            elif "dance" in result_lower or "baile" in result_lower:
                classified["dance_halls"].append(result)
            elif "cultural" in result_lower:
                classified["cultural_centers"].append(result)
            elif "cafe" in result_lower or "café" in result_lower:
                classified["cafes"].append(result)
                
        return classified

    def _format_venue_results(self, classified_venues):
        """
        Format classified venue results into a user-friendly string.

        Args:
            classified_venues (dict): Venues classified by type

        Returns:
            str: Formatted venue listings
        """
        formatted = []
        
        for venue_type, venues in classified_venues.items():
            if venues:
                formatted.append(f"\n{venue_type.upper().replace('_', ' ')}:")
                for venue in venues:
                    formatted.append(f"- {venue}")
                    
        return "\n".join(formatted)

    def get_recommendations(self, destination):
        """
        Generate comprehensive nightlife recommendations for a specific destination.

        Args:
            destination (str): Target destination

        Returns:
            str: Detailed venue recommendations with ratings and descriptions
        """
        nightlife_results = self.search_nightlife(destination)
        
        system_prompt = f"""Basándote en la siguiente información sobre vida nocturna en {destination},
        genera una lista de los 5-10 mejores lugares con este formato para cada uno:
        - Nombre del lugar
        - Tipo (bar, discoteca, club, teatro, etc.)
        - Costo promedio (en USD)
        - Valoración (1-10)
        - Breve descripción del ambiente y tipo de entretenimiento
        
        La respuesta debe ser detallada pero concisa."""
        
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Lugares nocturnos en {destination}: {nightlife_results}"}
            ]
        )
        
        return response.choices[0].message.content

    def process_query(self, query):
        """
        Process a user query to extract venue preferences and get recommendations.

        Args:
            query (str): The user's query string

        Returns:
            str: Nightlife recommendations based on the query
        """
        self.beliefs["current_query"] = query
        return self.action({"type": "query", "content": query})
