from .bdi_agent import BDIAgent
from .blackboard import Blackboard

class LodgingAgent(BDIAgent):
    def __init__(self, name, vector_db=None):
        super().__init__(name, vector_db)
        self.specialization = "lodging"
        self.blackboard = Blackboard()
        
        # Inicializar creencias específicas de alojamiento
        self.beliefs = {
            "accommodation_types": ["hotel", "hostal", "casa_particular", "resort"],
            "locations": {},
            "amenities": [],
            "price_ranges": ["economic", "moderate", "luxury"]
        }
        
        # Definir deseos del agente de alojamiento
        self.desires = [
            "buscar_alojamientos",
            "recomendar_hospedaje",
            "analizar_comodidades"
        ]
        
        # Definir planes disponibles
        self.plans = {
            "buscar_alojamientos": {
                "objetivo": "encontrar_opciones",
                "precondiciones": ["tiene_ubicacion"],
                "acciones": ["buscar_en_db", "clasificar_resultados", "formatear_respuesta"]
            },
            "recomendar_hospedaje": {
                "objetivo": "dar_recomendaciones",
                "precondiciones": ["tiene_ubicacion", "tiene_preferencias"],
                "acciones": ["analizar_requisitos", "filtrar_opciones", "generar_recomendaciones"]
            },
            "analizar_comodidades": {
                "objetivo": "evaluar_instalaciones",
                "precondiciones": ["tiene_alojamiento"],
                "acciones": ["listar_amenities", "evaluar_calidad", "generar_informe"]
            }
        }

    def search_accommodations(self, query, relevant_docs=None):
        """Search for accommodations matching the query"""
        if relevant_docs is None:
            relevant_docs = self.vector_db.similarity_search(
                self._build_accommodation_query(query)
            )
            
        # Process the relevant documents
        processed_results = []
        for doc in relevant_docs:
            if any(acc_type in doc.page_content.lower() for acc_type in self.beliefs["accommodation_types"]):
                processed_results.append(doc.page_content)
                
        # Classify the results
        classified_results = self._classify_accommodations(processed_results)
        
        # Update beliefs
        location = query.split()[0]  # Simple location extraction
        if location not in self.beliefs["locations"]:
            self.beliefs["locations"][location] = classified_results
        
        return self._format_accommodation_results(classified_results)
        
    def _build_accommodation_query(self, location, preferences=None):
        """Build optimized search query based on location and preferences"""
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
        """Classify accommodation results by type"""
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
        """Format results in a user-friendly way"""
        formatted = []
        
        for acc_type, accommodations in classified_results.items():
            if accommodations:
                formatted.append(f"\n{acc_type.upper()}:")
                for acc in accommodations:
                    formatted.append(f"- {acc}")
                    
        return "\n".join(formatted)

    def get_accommodation_suggestion(self, location, preferences, budget):
        """Get personalized accommodation suggestions"""
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

    def communicate(self, recipient, message):
        if isinstance(recipient, BDIAgent):
            recipient.receive_message(self, message)
            
    def get_recommendations(self, destination):
        """Get lodging recommendations for a specific destination"""
        # Primero buscar alojamientos
        lodging_results = self.search_accommodations(destination)
        
        # Preparar el prompt para el LLM
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
        """Verifica si un plan es relevante para el estado actual"""
        if plan["objetivo"] == "encontrar_opciones":
            return "destination" in self.beliefs
        elif plan["objetivo"] == "dar_recomendaciones":
            return "current_query" in self.beliefs and "destination" in self.beliefs
        elif plan["objetivo"] == "evaluar_instalaciones":
            return "selected_accommodation" in self.beliefs
        return False

    def _check_precondition(self, precondition) -> bool:
        """Verifica una precondición específica"""
        if precondition == "tiene_ubicacion":
            return "destination" in self.beliefs
        elif precondition == "tiene_preferencias":
            return bool(self.beliefs.get("preferences", {}))
        elif precondition == "tiene_alojamiento":
            return "selected_accommodation" in self.beliefs
        return False

    def _is_achievable(self, plan) -> bool:
        """Verifica si un plan es alcanzable según las precondiciones"""
        return all(self._check_precondition(pre) for pre in plan["precondiciones"])

    def _is_compatible(self, plan) -> bool:
        """Verifica si un plan es compatible con las intenciones actuales"""
        # Todos los planes de alojamiento son compatibles entre sí
        return True

    def _get_next_action(self, intention) -> str:
        """Determina la siguiente acción para una intención"""
        if not intention.get("acciones"):
            return None
        return intention["acciones"][0]

    def _perform_action(self, action):
        """Ejecuta una acción específica"""
        if action == "buscar_en_db":
            return self.search_accommodations(self.beliefs["destination"])
        elif action == "generar_recomendaciones":
            preferences = self.beliefs.get("preferences", {})
            return self.get_accommodation_suggestion(
                self.beliefs["destination"],
                preferences.get("preferences", []),
                preferences.get("budget", "moderate")
            )
        elif action == "listar_amenities":
            return self._analyze_amenities(self.beliefs["selected_accommodation"])
        return None

    def get_recommendations(self, destination):
        """Obtiene recomendaciones usando el ciclo BDI"""
        # Crear percepción con el destino
        percept = {"destination": destination}
        
        # Ejecutar ciclo BDI y obtener acción
        return self.action(percept)

    def _analyze_amenities(self, accommodation):
        """Analiza las comodidades de un alojamiento"""
        amenities_prompt = f"""Analiza las siguientes características del alojamiento {accommodation}:
        - Instalaciones principales
        - Servicios disponibles
        - Comodidades adicionales
        - Calidad general
        Genera un informe conciso pero completo."""
        
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": amenities_prompt},
                {"role": "user", "content": str(accommodation)}
            ]
        )
        return response.choices[0].message.content