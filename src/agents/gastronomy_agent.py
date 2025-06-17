from .bdi_agent import BDIAgent
from .blackboard import Blackboard

class GastronomyAgent(BDIAgent):
    def __init__(self, name, vector_db=None):
        super().__init__(name, vector_db)
        self.specialization = "gastronomy"
        self.blackboard = Blackboard()
        
        # Inicializar creencias específicas de gastronomía
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
            ]
        }
        
        # Definir deseos del agente gastronómico
        self.desires = [
            "buscar_restaurantes",
            "recomendar_lugares",
            "analizar_preferencias"
        ]
        
        # Definir planes disponibles
        self.plans = {
            "buscar_restaurantes": {
                "objetivo": "encontrar_opciones",
                "precondiciones": ["tiene_ubicacion"],
                "acciones": ["buscar_en_db", "clasificar_resultados", "formatear_respuesta"]
            },
            "recomendar_lugares": {
                "objetivo": "dar_recomendaciones",
                "precondiciones": ["tiene_ubicacion", "tiene_preferencias"],
                "acciones": ["analizar_preferencias", "filtrar_opciones", "generar_recomendaciones"]
            },
            "analizar_preferencias": {
                "objetivo": "entender_gustos",
                "precondiciones": ["tiene_consulta"],
                "acciones": ["extraer_preferencias", "clasificar_preferencias"]
            }
        }

    def search_restaurants(self, query, relevant_docs=None):
        """Search for restaurants matching the query"""
        if relevant_docs is None:
            relevant_docs = self.vector_db.similarity_search(
                self._build_restaurant_query(query)
            )
            
        # Process the relevant documents
        processed_results = []
        for doc in relevant_docs:
            if any(cuisine in doc.page_content.lower() for cuisine in self.beliefs["cuisine_types"]):
                processed_results.append(doc.page_content)
                
        # Classify the results
        classified_restaurants = self._classify_restaurants(processed_results)
        
        # Update beliefs
        location = query.split()[0]  # Simple location extraction
        if location not in self.beliefs["restaurants"]:
            self.beliefs["restaurants"][location] = classified_restaurants
            
        return self._format_restaurant_results(classified_restaurants)

    def _build_restaurant_query(self, location, filters=None):
        """Build optimized search query based on location and filters"""
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
        """Classify restaurants by cuisine type"""
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
        """Format restaurant results in a user-friendly way"""
        formatted = []
        
        for cuisine_type, restaurants in classified_restaurants.items():
            if restaurants:
                formatted.append(f"\n{cuisine_type.upper()} CUISINE:")
                for restaurant in restaurants:
                    formatted.append(f"- {restaurant}")
                    
        return "\n".join(formatted)

    def get_restaurant_suggestion(self, location, preferences, budget):
        """Get personalized restaurant suggestions"""
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

    def communicate(self, recipient, message):
        if isinstance(recipient, BDIAgent):
            recipient.receive_message(self, message)
            
    def get_recommendations(self, destination):
        """Get restaurant recommendations for a specific destination"""
        prompt = f"""Dame los 10 mejores restaurantes en {destination} con el siguiente formato para cada uno:
        - Nombre del restaurante
        - Tipo de cocina
        - Rango de precios (en USD)
        - Valoración (1-10)
        - Breve descripción de la experiencia gastronómica
        
        La respuesta debe ser detallada pero concisa, enfocada en la calidad de la comida y la experiencia."""
        
        relevant_docs = self.vector_db.similarity_search(prompt)
        
        # Procesar los documentos relevantes
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
        
        # Si no hay documentos relevantes, generar recomendaciones genéricas
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

    def _is_plan_relevant(self, plan) -> bool:
        """Verifica si un plan es relevante para el estado actual"""
        if plan["objetivo"] == "encontrar_opciones":
            return "destination" in self.beliefs
        elif plan["objetivo"] == "dar_recomendaciones":
            return "current_query" in self.beliefs and "destination" in self.beliefs
        elif plan["objetivo"] == "entender_gustos":
            return "current_query" in self.beliefs
        return False

    def _check_precondition(self, precondition) -> bool:
        """Verifica una precondición específica"""
        if precondition == "tiene_ubicacion":
            return "destination" in self.beliefs
        elif precondition == "tiene_preferencias":
            return bool(self.beliefs.get("preferences", {}))
        elif precondition == "tiene_consulta":
            return "current_query" in self.beliefs
        return False

    def _is_achievable(self, plan) -> bool:
        """Verifica si un plan es alcanzable según las precondiciones"""
        return all(self._check_precondition(pre) for pre in plan["precondiciones"])

    def _is_compatible(self, plan) -> bool:
        """Verifica si un plan es compatible con las intenciones actuales"""
        # Todos los planes de gastronomía son compatibles entre sí
        return True

    def _get_next_action(self, intention) -> str:
        """Determina la siguiente acción para una intención"""
        if not intention.get("acciones"):
            return None
        return intention["acciones"][0]

    def _perform_action(self, action):
        """Ejecuta una acción específica"""
        if action == "buscar_en_db":
            return self.search_restaurants(self.beliefs["destination"])
        elif action == "generar_recomendaciones":
            preferences = self.beliefs.get("preferences", {})
            return self.get_restaurant_suggestion(
                self.beliefs["destination"],
                preferences,
                preferences.get("budget", "moderate")
            )
        elif action == "extraer_preferencias":
            return self._extract_preferences(self.beliefs["current_query"])
        return None

    def get_recommendations(self, destination):
        """Obtiene recomendaciones usando el ciclo BDI"""
        # Crear percepción con el destino
        percept = {"destination": destination}
        
        # Ejecutar ciclo BDI y obtener acción
        return self.action(percept)

    def _extract_preferences(self, query):
        """Extrae preferencias gastronómicas de la consulta"""
        system_prompt = """Extrae las siguientes preferencias del texto:
        - tipo de cocina
        - rango de precio
        - dietas especiales
        - tipo de comida
        Responde en formato JSON"""
        
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        try:
            preferences = eval(response.choices[0].message.content)
            self.beliefs["preferences"] = preferences
            return preferences
        except:
            return {}