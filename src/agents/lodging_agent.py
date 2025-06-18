from .bdi_agent import BDIAgent
from .blackboard import Blackboard

class LodgingAgent(BDIAgent):
    def __init__(self, name, vector_db=None):
        super().__init__(name, vector_db)
        self.specialization = "lodging"
        self.blackboard = Blackboard()
        
        self.beliefs = {
            "accommodation_types": ["hotel", "hostal", "casa_particular", "resort"],
            "locations": {},
            "amenities": [],
            "price_ranges": ["economic", "moderate", "luxury"],
            "has_results": False, 
            "needs_recommendations": False 
        }
        
        self.desires = [
            "buscar_alojamientos", 
            "recomendar_hospedaje", 
        ]
        
        self.plans = {
            "buscar_alojamientos": {
                "objetivo": "encontrar_opciones",
                "precondiciones": ["tiene_ubicacion", "no_tiene_resultados"],
                "acciones": ["buscar_en_db"]
            },
            "recomendar_hospedaje": {
                "objetivo": "dar_recomendaciones",
                "precondiciones": ["tiene_ubicacion", "tiene_resultados", "necesita_recomendaciones"],
                "acciones": ["generar_recomendaciones"]
            }
        }

    def search_accommodations(self, query, relevant_docs=None):
        """
        Search for accommodations matching the provided query.

        Args:
            query (str): The search query string
            relevant_docs (list, optional): Pre-filtered relevant documents

        Returns:
            str: Formatted string containing accommodation search results
        """
        if relevant_docs is None:
            relevant_docs = self.vector_db.similarity_search(
                self._build_accommodation_query(query)
            )
            
        processed_results = []
        for doc in relevant_docs:
            if any(acc_type in doc.page_content.lower() for acc_type in self.beliefs["accommodation_types"]):
                processed_results.append(doc.page_content)
                
        classified_results = self._classify_accommodations(processed_results)
        
        location = query.split()[0] 
        if location not in self.beliefs["locations"]:
            self.beliefs["locations"][location] = classified_results
        
        self.beliefs["has_results"] = bool(classified_results)
        self.beliefs["needs_recommendations"] = True
        
        return self._format_accommodation_results(classified_results)
        
    def _build_accommodation_query(self, location, preferences=None):
        """
        Build an optimized search query based on location and user preferences.

        Args:
            location (str): The target location for accommodation search
            preferences (list, optional): List of user preferences

        Returns:
            str: The constructed search query
        """
        query_parts = [f"accommodations in {location}"]
        
        if preferences:
            for pref in preferences:
                if pref in ["beach", "city", "mountain"]:
                    query_parts.append(f"near {pref}")
                elif pref in ["economic", "moderate", "luxury"]:
                    query_parts.append(f"{pref} price")
                else:
                    query_parts.append(pref)
                    
        return " ".join(query_parts)

    def _classify_accommodations(self, results):
        """
        Classify accommodation results by type (hotels, hostals, etc.).

        Args:
            results (list): List of accommodation search results

        Returns:
            dict: Accommodations classified by type
        """
        classified = {
            "hotels": [],
            "hostals": [],
            "casas_particulares": [],
            "resorts": []
        }
        
        for result in results:
            if "hotel" in result.lower():
                classified["hotels"].append(result)
            elif "hostal" in result.lower():
                classified["hostals"].append(result)
            elif "casa" in result.lower():
                classified["casas_particulares"].append(result)
            elif "resort" in result.lower():
                classified["resorts"].append(result)
                
        return classified

    def _format_accommodation_results(self, classified_results):
        """
        Format classified accommodation results into a user-friendly string.

        Args:
            classified_results (dict): Accommodations classified by type

        Returns:
            str: Formatted accommodation listings
        """
        formatted = []
        
        for acc_type, accommodations in classified_results.items():
            if accommodations:
                formatted.append(f"\n{acc_type.upper()}:")
                for acc in accommodations:
                    formatted.append(f"- {acc}")
                    
        return "\n".join(formatted)

    def get_accommodation_suggestion(self, location, preferences, budget):
        """
        Get personalized accommodation suggestions based on user preferences.

        Args:
            location (str): Target location
            preferences (list): User accommodation preferences
            budget (str): User's budget level

        Returns:
            str: Personalized accommodation recommendation
        """
        results = self.search_accommodations(location)
        
        suggestion_prompt = f"""
        Based on these accommodations in {location}:
        {results}
        
        Suggest the best option for a traveler with:
        - Budget: {budget}
        - Preferences: {', '.join(preferences)}
        """
        
        suggestion = self.client.chat(
            model="mistral-medium",
            messages=[{
                "role": "user", 
                "content": suggestion_prompt
            }]
        ).choices[0].message.content
        
        return suggestion
            
    def get_recommendations(self, destination):
        """
        Generate comprehensive lodging recommendations for a specific destination.

        Args:
            destination (str): Target destination

        Returns:
            str: Detailed accommodation recommendations with ratings and descriptions
        """
        lodging_results = self.search_accommodations(destination)
        
        system_prompt = f"""Basándote en la siguiente información sobre alojamientos en {destination},
        genera una lista de los 5-10 mejores lugares con este formato para cada uno:
        - Nombre del alojamiento
        - Tipo (hotel, hostal, casa particular, resort, etc.)
        - Costo por noche (en USD)
        - Valoración (1-10)
        - Breve descripción de las instalaciones y ubicación
        
        La respuesta debe ser detallada pero concisa."""
        
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Información sobre alojamientos: {lodging_results}"}
            ]
        )
        return response.choices[0].message.content
    
    def _is_plan_relevant(self, plan) -> bool:
        """
        Check if a plan is relevant for the current state.

        Args:
            plan (dict): The plan to evaluate

        Returns:
            bool: True if the plan's objective matches current beliefs, False otherwise
        """
        if plan["objetivo"] == "encontrar_opciones":
            return not self.beliefs.get("has_results", False)
        elif plan["objetivo"] == "dar_recomendaciones":
            return self.beliefs.get("has_results", False) and self.beliefs.get("needs_recomendaciones", False)
        return False

    def _check_precondition(self, precondition) -> bool:
        """
        Verify if a specific precondition is met.

        Args:
            precondition (str): The precondition to check

        Returns:
            bool: True if precondition is met, False otherwise
        """
        if precondition == "tiene_ubicacion":
            return "destination" in self.beliefs or "current_query" in self.beliefs
        elif precondition == "tiene_preferencias":
            return bool(self.beliefs.get("preferences", {}))
        elif precondition == "tiene_alojamiento":
            return "selected_accommodation" in self.beliefs
        elif precondition == "tiene_resultados":
            return self.beliefs.get("has_results", False)
        elif precondition == "no_tiene_resultados":
            return not self.beliefs.get("has_results", False)
        elif precondition == "necesita_recomendaciones":
            return self.beliefs.get("needs_recomendaciones", False)
        return False

    def _is_achievable(self, plan) -> bool:
        """
        Check if a plan is achievable based on its preconditions.

        Args:
            plan (dict): The plan to evaluate

        Returns:
            bool: True if all preconditions are met, False otherwise
        """
        return all(self._check_precondition(pre) for pre in plan["precondiciones"])

    def _is_compatible(self, plan) -> bool:
        """
        Check if a plan is compatible with current intentions.

        Args:
            plan (dict): The plan to evaluate

        Returns:
            bool: True if the plan is compatible, False otherwise
        """
        return True

    def _get_next_action(self, intention) -> str:
        """
        Determine the next action for a given intention.

        Args:
            intention (dict): The intention to process

        Returns:
            str: The next action to be performed or None
        """
        if not intention.get("acciones"):
            return None
        return intention["acciones"][0]

    def _perform_action(self, action):
        """
        Execute a specific action based on the current beliefs and intentions.

        Args:
            action (str): The action to perform

        Returns:
            str: Result of the action or None if action cannot be performed
        """
        if action == "buscar_en_db":
            query = self.beliefs.get("current_query", self.beliefs.get("destination", ""))
            return self.search_accommodations(query)
        elif action == "generar_recomendaciones":
            preferences = self.beliefs.get("preferences", {})
            result = self.get_accommodation_suggestion(
                self.beliefs.get("destination", self.beliefs.get("current_query", "")),
                preferences.get("preferences", []),
                preferences.get("budget", "moderate")
            )
            self.beliefs["needs_recomendaciones"] = False
            return result
        return None

    def get_recommendations(self, destination):
        """
        Get recommendations using the BDI cycle.

        Args:
            destination (str): Target destination for recommendations

        Returns:
            str: Generated recommendations based on the BDI cycle execution
        """
        percept = {"destination": destination}
        
        return self.action(percept)