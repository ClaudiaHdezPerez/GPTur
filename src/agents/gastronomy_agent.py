import json
from .bdi_agent import BDIAgent
from .blackboard import Blackboard

class GastronomyAgent(BDIAgent):
    def __init__(self, name, vector_db=None):
        super().__init__(name, vector_db)
        self.specialization = "gastronomy"
        self.blackboard = Blackboard()
        
        self.beliefs = {
            "restaurants": {},
            "cuisine_types": [
                "cuban",
                "seafood",
                "international",
                "creole",
                "italian",
                "fusion"
            ],
            "price_ranges": ["economic", "moderate", "luxury"],
            "special_diets": [
                "vegetarian",
                "vegan",
                "gluten_free"
            ],
            "meal_types": [
                "breakfast",
                "lunch",
                "dinner",
                "snacks",
                "drinks"
            ],
            "current_query": None,
            "destination": None,
            "preferences": None
        }
        
        self.desires = [
            "process_user_query", 
            "recommend_restaurants"
        ]

        self.plans = {
            "process_user_query": {
                "objetivo": "extract_info_from_query",
                "precondiciones": ["has_query"],
                "acciones": ["extract_destination_and_preferences"]
            },
            "recommend_restaurants": {
                "objetivo": "provide_recommendations",
                "precondiciones": ["has_destination"],
                "acciones": ["get_recommendations"]
            }
        }

    def search_restaurants(self, query, relevant_docs=None):
        """
        Search for restaurants matching the specified query.

        Args:
            query (str): The search query string
            relevant_docs (list, optional): Pre-filtered relevant documents

        Returns:
            str: Formatted string containing restaurant search results
        """
        if relevant_docs is None:
            relevant_docs = self.vector_db.similarity_search(
                self._build_restaurant_query(query)
            )
            
        processed_results = []
        for doc in relevant_docs:
            if any(cuisine in doc.page_content.lower() for cuisine in self.beliefs["cuisine_types"]):
                processed_results.append(doc.page_content)
                
        classified_restaurants = self._classify_restaurants(processed_results)
        
        location = query.split()[0] 
        if location not in self.beliefs["restaurants"]:
            self.beliefs["restaurants"][location] = classified_restaurants
            
        return self._format_restaurant_results(classified_restaurants)

    def _build_restaurant_query(self, location, filters=None):
        """
        Constructs an optimized search query based on location and filters.

        Args:
            location (str): The target location for restaurant search
            filters (dict, optional): Additional search filters (cuisine, price, diet, meal)

        Returns:
            str: The constructed search query
        """
        query_parts = [f"restaurants and dining options in {location}"]
        
        if filters:
            for filter_type, value in filters.items():
                if filter_type == "cuisine" and value in self.beliefs["cuisine_types"]:
                    query_parts.append(f"specializing in {value} cuisine")
                elif filter_type == "price" and value in self.beliefs["price_ranges"]:
                    query_parts.append(f"with {value} prices")
                elif filter_type == "diet" and value in self.beliefs["special_diets"]:
                    query_parts.append(f"offering {value} options")
                elif filter_type == "meal" and value in self.beliefs["meal_types"]:
                    query_parts.append(f"for {value}")
                    
        return " ".join(query_parts)

    def _classify_restaurants(self, results):
        """
        Classifies restaurants by cuisine type from search results.

        Args:
            results (list): List of restaurant search results

        Returns:
            dict: Restaurants classified by cuisine type
        """
        classified = {
            "cuban": [],
            "seafood": [],
            "international": [],
            "creole": [],
            "italian": [],
            "fusion": []
        }
        
        for result in results:
            result_lower = result.lower()
            if "cuba" in result_lower or "criolla" in result_lower:
                classified["cuban"].append(result)
            elif "mar" in result_lower or "seafood" in result_lower or "pescado" in result_lower:
                classified["seafood"].append(result)
            elif "international" in result_lower or "internacional" in result_lower:
                classified["international"].append(result)
            elif "creole" in result_lower or "criolla" in result_lower:
                classified["creole"].append(result)
            elif "italian" in result_lower or "italiana" in result_lower:
                classified["italian"].append(result)
            else:
                classified["fusion"].append(result)
                
        return classified

    def _format_restaurant_results(self, classified_restaurants):
        """
        Formats classified restaurant results into a user-friendly string.

        Args:
            classified_restaurants (dict): Restaurants classified by cuisine type

        Returns:
            str: Formatted restaurant listings
        """
        formatted = []
        
        for cuisine_type, restaurants in classified_restaurants.items():
            if restaurants:
                formatted.append(f"\n{cuisine_type.upper()} CUISINE:")
                for restaurant in restaurants:
                    formatted.append(f"- {restaurant}")
                    
        return "\n".join(formatted)

    def get_restaurant_suggestion(self, location, preferences, budget):
        """
        Provides personalized restaurant suggestions based on user preferences.

        Args:
            location (str): Target location
            preferences (dict): User dining preferences
            budget (str): User's budget level

        Returns:
            str: Personalized restaurant recommendation
        """
        results = self.search_restaurants(location)
        
        suggestion_prompt = f"""
        Based on these restaurants in {location}:
        {results}
        
        Suggest the best dining option for a visitor with:
        - Budget: {budget}
        - Cuisine preference: {preferences.get('cuisine', 'any')}
        - Dietary restrictions: {preferences.get('diet', 'none')}
        - Meal type: {preferences.get('meal', 'any')}
        """
        
        suggestion = self.client.chat(
            model="mistral-medium",
            messages=[{
                "role": "user",
                "content": suggestion_prompt
            }]
        ).choices[0].message.content
        
        return suggestion
            
    def _get_recommendations(self, destination):
        """
        Retrieves detailed restaurant recommendations for a specific destination.

        Args:
            destination (str): Target destination

        Returns:
            str: Detailed restaurant recommendations with ratings and descriptions
        """
        prompt = f"""Dame los 10 mejores restaurantes en {destination} con el siguiente formato para cada uno:
        - Nombre del restaurante
        - Tipo de cocina
        - Rango de precios (en USD)
        - Valoración (1-10)
        - Breve descripción de la experiencia gastronómica
        
        La respuesta debe ser detallada pero concisa, enfocada en la calidad de la comida y la experiencia."""
        
        relevant_docs = self.vector_db.similarity_search(prompt)
        
        if relevant_docs:
            system_prompt = """Basándote en la información proporcionada, genera una lista de restaurantes
            siguiendo el formato especificado. Si no hay suficiente información, genera recomendaciones
            razonables basadas en el contexto cultural y gastronómico del destino."""
            
            response = self.client.chat(
                model="mistral-medium",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Información sobre restaurantes en {destination}: " + 
                     "\n".join([doc.page_content for doc in relevant_docs]) + 
                     "\n\nGenera las recomendaciones siguiendo el formato solicitado."}
                ]
            )
            return response.choices[0].message.content
        
        return f"""Aquí hay algunos restaurantes recomendados en {destination}:
        
        1. Café La Habana
        - Cocina tradicional cubana
        - $15-25 USD
        - Valoración: 8/10
        - Auténtica experiencia local con platos típicos y música en vivo
        
        2. El Marinero
        - Mariscos y pescados frescos
        - $20-35 USD
        - Valoración: 9/10
        - Especialidad en pescados locales y mariscos del Caribe
        
        3. La Terraza Internacional
        - Cocina internacional y fusión
        - $25-40 USD
        - Valoración: 8.5/10
        - Ambiente elegante con vista panorámica y menú variado
        
        4. Paladar Típico
        - Cocina criolla
        - $10-20 USD
        - Valoración: 7.5/10
        - Ambiente familiar y platos caseros tradicionales
        
        5. Milano
        - Cocina italiana
        - $20-30 USD
        - Valoración: 8/10
        - Pasta fresca y pizzas artesanales en un ambiente romántico"""

    def _check_precondition(self, precondition) -> bool:
        """
        Verifies if a specific precondition is met.

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

    def _is_plan_relevant(self, plan) -> bool:
        """Check if a plan is relevant for the current state"""
        if plan["objetivo"] == "extract_info_from_query":
            return "current_query" in self.beliefs and self.beliefs["current_query"] is not None
        elif plan["objetivo"] == "provide_recommendations":
            return "destination" in self.beliefs and self.beliefs["destination"] is not None
        return False

    def _is_achievable(self, plan) -> bool:
        """Check if a plan is achievable based on its preconditions"""
        return all(self._check_precondition(pre) for pre in plan["precondiciones"])    
    
    def _is_compatible(self, plan) -> bool:
        """Check if a plan is compatible with current intentions"""
        return True

    def _get_next_action(self, intention) -> str:
        """Determine the next action for an intention"""
        if not intention.get("acciones"):
            return None
        return intention["acciones"][0]

    def _perform_action(self, action):
        """Execute a specific action"""
        if action == "extract_destination_and_preferences":
            query = self.beliefs["current_query"]
            system_prompt = """Extract the destination from this query. Return ONLY the destination name, nothing else.
            If no destination is found, return 'unknown'."""
            
            response = self.client.chat(
                model="mistral-medium",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            destination = response.choices[0].message.content.strip()
            
            if destination and destination.lower() != 'unknown':
                self.beliefs["destination"] = destination
            
            preferences = self._extract_preferences(query)
            self.beliefs["preferences"] = preferences
            
            return self.get_restaurant_suggestion(
                self.beliefs["destination"],
                preferences,
                preferences.get("price_range", "moderate")
            )
        elif action == "get_recommendations":
            return self._get_recommendations(self.beliefs["destination"])

    def process_query(self, query):
        """
        Processes a user query to extract preferences and generate recommendations.

        Args:
            query (str): The user's query string

        Returns:
            str: Restaurant recommendations based on the query
        """
        self.beliefs["current_query"] = query
        words = query.split()
        for word in words:
            if word.istitle():
                self.beliefs["destination"] = word
                break
        
        return self.action({"type": "query", "content": query})

    def get_recommendations(self, destination):
        """
        Gets general restaurant recommendations for a destination.

        Args:
            destination (str): Target destination

        Returns:
            str: List of recommended restaurants
        """
        self.beliefs["destination"] = destination
        self.beliefs["current_query"] = None
        self.beliefs["preferences"] = None 
        
        return self.action({"type": "destination", "content": destination})

    def _extract_preferences(self, query):
        """
        Extracts dining preferences from a user query.

        Args:
            query (str): The user's query string

        Returns:
            dict: Extracted preferences including cuisine, price range, diet, and meal type
        """
        system_prompt = """Eres un asistente que extrae preferencias gastronómicas.
        IMPORTANTE: Tu respuesta debe ser ÚNICAMENTE un objeto JSON válido, sin texto adicional.
        
        Formato requerido:
        {
            "cuisine": "<tipo de cocina o 'any'>",
            "price_range": "<economic/moderate/luxury>",
            "diet": "<vegetarian/vegan/gluten_free/none>",
            "meal": "<breakfast/lunch/dinner/snacks/drinks/any>"
        }
        
        Si una preferencia no está clara en la consulta, usa 'any' o 'none' como valor por defecto."""
        
        try:
            response = self.client.chat(
                model="mistral-medium",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extrae las preferencias gastronómicas de: {query}"}
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
                        
            if response_text.startswith("```") and response_text.endswith("```"):
                response_text = response_text[3:-3].strip()
            
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[7:-3].strip()
            
            preferences = json.loads(response_text)
            
            default_preferences = {
                "cuisine": "any",
                "price_range": "moderate",
                "diet": "none",
                "meal": "any"
            }
            
            valid_preferences = default_preferences.copy()
            
            if "cuisine" in preferences and preferences["cuisine"].lower() in [c.lower() for c in self.beliefs["cuisine_types"] + ["any"]]:
                valid_preferences["cuisine"] = preferences["cuisine"].lower()
            
            if "price_range" in preferences and preferences["price_range"].lower() in self.beliefs["price_ranges"]:
                valid_preferences["price_range"] = preferences["price_range"].lower()
            
            if "diet" in preferences and preferences["diet"].lower() in [d.lower() for d in self.beliefs["special_diets"] + ["none"]]:
                valid_preferences["diet"] = preferences["diet"].lower()
            
            if "meal" in preferences and preferences["meal"].lower() in [m.lower() for m in self.beliefs["meal_types"] + ["any"]]:
                valid_preferences["meal"] = preferences["meal"].lower()
            
            self.beliefs["preferences"] = valid_preferences
            return valid_preferences
            
        except json.JSONDecodeError as e:
            print(f"Error decodificando JSON: {e}")
            print(f"Texto que causó el error: {response_text}")
            return {
                "cuisine": "any",
                "price_range": "moderate",
                "diet": "none",
                "meal": "any"
            }
        except Exception as e:
            print(f"Error extracting preferences: {e}")
            return {
                "cuisine": "any",
                "price_range": "moderate",
                "diet": "none",
                "meal": "any"
            }