from .bdi_agent import BDIAgent
from .blackboard import Blackboard
from datetime import datetime

class HistoricAgent(BDIAgent):
    def __init__(self, name, vector_db=None):
        super().__init__(name, vector_db)
        self.specialization = "historic"
        self.blackboard = Blackboard()

        self.beliefs = {
            "historic_sites": {},
            "architectural_styles": [
                "colonial", 
                "neoclassical",
                "art_deco",
                "modern"
            ],
            "site_types": [
                "museum",
                "church",
                "plaza",
                "monument",
                "historic_building",
                "cultural_center"
            ],
            "current_query": None,
            "destination": None
        }

        self.desires = [
            "process_user_query",
            "recommend_historic_sites"
        ]
        
        self.plans = {
            "process_user_query": {
                "objetivo": "search_historic_sites",
                "precondiciones": ["has_query"],
                "acciones": ["search_historic_sites"]
            },
            "recommend_historic_sites": {
                "objetivo": "provide_recommendations",
                "precondiciones": ["has_destination"],
                "acciones": ["get_recommendations"]
            }
        }

    def search_historic_sites(self, query, relevant_docs=None):
        """
        Search for historic sites matching the provided query.

        Args:
            query (str): The search query string
            relevant_docs (list, optional): Pre-filtered relevant documents

        Returns:
            str: Formatted string containing historic sites search results
        """
        if relevant_docs is None:
            relevant_docs = self.vector_db.similarity_search(
                self._build_historic_query(query)
            )
        
        processed_results = []
        for doc in relevant_docs:
            if any(site_type in doc.page_content.lower() for site_type in self.beliefs["site_types"]):
                processed_results.append(doc.page_content)
                
        classified_sites = self._classify_historic_sites(processed_results)
          
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
            if location not in self.beliefs["historic_sites"]:
                self.beliefs["historic_sites"][location] = classified_sites
            
        self.beliefs["has_results"] = bool(classified_sites)
        self.beliefs["needs_recommendations"] = True
        self.beliefs["needs_event_info"] = True
        self.beliefs["needs_historic_details"] = True
            
        return self._format_historic_results(classified_sites)

    def _build_historic_query(self, location, site_type=None):
        """
        Build an optimized search query for historic sites.

        Args:
            location (str): The target location for historic site search
            site_type (str, optional): Specific type of historic site to search for

        Returns:
            str: The constructed search query
        """
        query_parts = [f"historic and cultural attractions in {location}"]
        
        if site_type:
            if site_type in self.beliefs["site_types"]:
                query_parts.append(f"focusing on {site_type}")
                
        return " ".join(query_parts)

    def _classify_historic_sites(self, results):
        """
        Classify historic sites by their type from search results.

        Args:
            results (list): List of historic site search results

        Returns:
            dict: Sites classified by type (museums, churches, etc.)
        """
        classified = {
            "museums": [],
            "churches": [],
            "plazas": [],
            "monuments": [],
            "historic_buildings": [],
            "cultural_centers": []
        }
        
        for result in results:
            if "museo" in result.lower() or "museum" in result.lower():
                classified["museums"].append(result)
            elif "iglesia" in result.lower() or "church" in result.lower():
                classified["churches"].append(result)
            elif "plaza" in result.lower():
                classified["plazas"].append(result)
            elif "monumento" in result.lower() or "monument" in result.lower():
                classified["monuments"].append(result)
            elif "cultural" in result.lower():
                classified["cultural_centers"].append(result)
            else:
                classified["historic_buildings"].append(result)
                
        return classified

    def _format_historic_results(self, classified_sites):
        """
        Format classified historic sites into a user-friendly string.

        Args:
            classified_sites (dict): Historic sites classified by type

        Returns:
            str: Formatted historic site listings
        """
        formatted = []
        
        for site_type, sites in classified_sites.items():
            if sites:
                formatted.append(f"\n{site_type.upper().replace('_', ' ')}:")
                for site in sites:
                    formatted.append(f"- {site}")
                    
        return " ".join(formatted)

    def search_cultural_events(self, location, date_range=None):
        """
        Search for cultural events in a specific location.

        Args:
            location (str): The target location
            date_range (str, optional): Date range for events search

        Returns:
            str: Formatted list of cultural events
        """
        if not date_range:
            current_date = datetime.now()
            date_range = f"{current_date.strftime('%B %Y')}"
            
        query = f"cultural events and festivals in {location} during {date_range}"
        results = self.vector_db.search(query)
        
        if location not in self.beliefs["cultural_events"]:
            self.beliefs["cultural_events"][location] = []
            
        self.beliefs["cultural_events"][location] = results
        
        return self._format_event_results(results)

    def _format_event_results(self, events):
        """
        Format cultural events results into a user-friendly string.

        Args:
            events (list): List of cultural events

        Returns:
            str: Formatted list of upcoming cultural events
        """
        formatted = ["\nUPCOMING CULTURAL EVENTS:"]
        
        for event in events:
            formatted.append(f"- {event}")
            
        return "\n".join(formatted)

    def get_site_details(self, site_name, location):
        """
        Retrieve detailed information about a specific historic site.

        Args:
            site_name (str): Name of the historic site
            location (str): Location of the site

        Returns:
            str: Detailed description of the historic site
        """
        relevant_docs = self.vector_db.similarity_search(
            f"details about {site_name} in {location}"
        )
        
        details_prompt = f"""
        Based on this information about {site_name}:
        {[doc.page_content for doc in relevant_docs]}
        
        Provide a detailed description including:
        - Historical significance
        - Architectural features
        - Cultural importance
        - Current state and visitor information
        """
        
        details = self.client.chat(
            model="mistral-medium",
            messages=[{
                "role": "user",
                "content": details_prompt
            }]
        ).choices[0].message.content
        
        return details
            
    def get_recommendations(self, destination):
        """
        Generate historic site recommendations for a specific destination.

        Args:
            destination (str): Target destination

        Returns:
            str: Detailed recommendations with ratings and descriptions
        """
        historic_results = self.search_historic_sites(destination)
        
        system_prompt = f"""Basándote en la siguiente información sobre sitios históricos en {destination},
        genera una lista de los 5-10 sitios más importantes con este formato para cada uno:
        - Nombre del sitio
        - Tipo (museo, monumento, plaza, etc.)
        - Costo de entrada (en USD)
        - Valoración (1-10)
        - Breve descripción de su importancia histórica
        
        La respuesta debe ser detallada pero concisa."""
        
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Información sobre sitios históricos: {historic_results}"}
            ]
        )        
        
        return response.choices[0].message.content
        
    def _is_plan_relevant(self, plan) -> bool:
        """Verifica si un plan es relevante para el estado actual"""
        if plan["objetivo"] == "search_historic_sites":
            return "current_query" in self.beliefs and self.beliefs["current_query"] is not None
        elif plan["objetivo"] == "provide_recommendations":
            return "destination" in self.beliefs and self.beliefs["destination"] is not None
        return False

    def _check_precondition(self, precondition) -> bool:
        """Verifica una precondición específica"""
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
        if action == "search_historic_sites":
            query = self.beliefs["current_query"]
            return self.search_historic_sites(query)
            
        elif action == "get_recommendations":
            return self.get_recommendations(self.beliefs["destination"])

    def get_recommendations(self, destination):
        """Obtiene recomendaciones usando el ciclo BDI"""
        percept = {"destination": destination}
        
        return self.action(percept)

    def _compile_historic_details(self, site_name, location):
        """
        Compile comprehensive historical details about a site.

        Args:
            site_name (str): Name of the historic site
            location (str): Location of the site

        Returns:
            str: Detailed historical information about the site
        """
        details_prompt = f"""Proporciona información histórica detallada sobre {site_name} en {location}, incluyendo:
        - Origen y fecha de construcción
        - Eventos históricos importantes
        - Valor cultural y patrimonial
        - Estado actual y relevancia turística
        La respuesta debe ser detallada pero concisa."""
        
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": details_prompt},
                {"role": "user", "content": f"Buscar información histórica de {site_name}"}
            ]
        )
        return response.choices[0].message.content