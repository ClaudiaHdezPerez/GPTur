from .bdi_agent import BDIAgent
from .blackboard import Blackboard
from datetime import datetime, time

class NightlifeAgent(BDIAgent):
    def __init__(self, name, vector_db=None):
        super().__init__(name, vector_db)
        self.specialization = "nightlife"
        self.blackboard = Blackboard()
        
        # Inicializar creencias específicas de vida nocturna
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
                    "open": time(20, 0),  # 8:00 PM
                    "close": time(2, 0)    # 2:00 AM
                },
                "late": {
                    "open": time(22, 0),   # 10:00 PM
                    "close": time(6, 0)    # 6:00 AM
                }
            }
        }
        
        # Definir deseos del agente de vida nocturna
        self.desires = [
            "buscar_lugares",
            "recomendar_venues",
            "verificar_horarios",
            "sugerir_actividades"
        ]
        
        # Definir planes disponibles
        self.plans = {
            "buscar_lugares": {
                "objetivo": "encontrar_venues",
                "precondiciones": ["tiene_ubicacion"],
                "acciones": ["buscar_en_db", "clasificar_venues", "formatear_respuesta"]
            },
            "recomendar_venues": {
                "objetivo": "dar_recomendaciones",
                "precondiciones": ["tiene_ubicacion", "tiene_preferencias"],
                "acciones": ["analizar_preferencias", "filtrar_por_tipo", "generar_recomendaciones"]
            },
            "verificar_horarios": {
                "objetivo": "confirmar_disponibilidad",
                "precondiciones": ["tiene_venue", "tiene_hora"],
                "acciones": ["verificar_hora", "comprobar_horario", "informar_estado"]
            },
            "sugerir_actividades": {
                "objetivo": "proponer_opciones",
                "precondiciones": ["tiene_ubicacion", "tiene_hora"],
                "acciones": ["buscar_activos", "filtrar_por_hora", "generar_sugerencias"]
            }
        }

    def search_nightlife(self, query, relevant_docs=None):
        """Search for nightlife venues matching the query"""
        if relevant_docs is None:
            relevant_docs = self.vector_db.similarity_search(
                self._build_nightlife_query(query)
            )
            
        # Process the relevant documents
        processed_results = []
        for doc in relevant_docs:
            if any(venue in doc.page_content.lower() for venue in self.beliefs["venue_types"]):
                processed_results.append(doc.page_content)
                
        # Classify the results
        classified_venues = self._classify_venues(processed_results)
        
        # Update beliefs
        location = query.split()[0]  # Simple location extraction
        if location not in self.beliefs["venues"]:
            self.beliefs["venues"][location] = classified_venues
            
        return self._format_venue_results(classified_venues)

    def _build_nightlife_query(self, location, filters=None):
        """Build optimized search query based on location and filters"""
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
        """Classify nightlife venues by type"""
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
        """Format venue results in a user-friendly way"""
        formatted = []
        
        for venue_type, venues in classified_venues.items():
            if venues:
                formatted.append(f"\n{venue_type.upper().replace('_', ' ')}:")
                for venue in venues:
                    formatted.append(f"- {venue}")
                    
        return "\n".join(formatted)

    def get_venue_suggestion(self, location, preferences):
        """Get personalized venue suggestions"""
        results = self.search_nightlife(location)
        
        current_time = datetime.now().time()
        
        suggestion_prompt = f"""
        Based on these venues in {location}:
        {results}
        
        Current time: {current_time.strftime('%H:%M')}
        
        Suggest the best nightlife option for a visitor with:
        - Preferred venue type: {preferences.get('venue_type', 'any')}
        - Music preference: {preferences.get('music_type', 'any')}
        - Budget: {preferences.get('budget', 'moderate')}
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
        """Get nightlife recommendations for a specific destination"""
        # Primero buscar lugares nocturnos
        nightlife_results = self.search_nightlife(destination)
        
        # Preparar el prompt para el LLM
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
                {"role": "user", "content": f"Información sobre lugares nocturnos: {nightlife_results}"}
            ]
        )
        return response.choices[0].message.content

    def _is_plan_relevant(self, plan) -> bool:
        """Verifica si un plan es relevante para el estado actual"""
        if plan["objetivo"] == "encontrar_venues":
            return "destination" in self.beliefs
        elif plan["objetivo"] == "dar_recomendaciones":
            return "current_query" in self.beliefs and "destination" in self.beliefs
        elif plan["objetivo"] == "confirmar_disponibilidad":
            return "venue_name" in self.beliefs and "current_time" in self.beliefs
        elif plan["objetivo"] == "proponer_opciones":
            return "destination" in self.beliefs and "current_time" in self.beliefs
        return False

    def _check_precondition(self, precondition) -> bool:
        """Verifica una precondición específica"""
        if precondition == "tiene_ubicacion":
            return "destination" in self.beliefs
        elif precondition == "tiene_preferencias":
            return bool(self.beliefs.get("preferences", {}))
        elif precondition == "tiene_venue":
            return "venue_name" in self.beliefs
        elif precondition == "tiene_hora":
            return "current_time" in self.beliefs
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
        if action == "buscar_en_db":
            return self.search_nightlife(self.beliefs["destination"])
        elif action == "generar_recomendaciones":
            return self.get_venue_suggestion(
                self.beliefs["destination"],
                self.beliefs.get("preferences", {})
            )
        elif action == "verificar_hora":
            return self._check_venue_availability(
                self.beliefs["venue_name"],
                self.beliefs["current_time"]
            )
        elif action == "generar_sugerencias":
            return self._suggest_current_activities(
                self.beliefs["destination"],
                self.beliefs["current_time"]
            )
        return None

    def get_recommendations(self, destination):
        """Obtiene recomendaciones usando el ciclo BDI"""
        # Crear percepción con el destino
        percept = {
            "destination": destination,
            "current_time": datetime.now().time()
        }
        
        # Ejecutar ciclo BDI y obtener acción
        return self.action(percept)

    def _check_venue_availability(self, venue_name, current_time):
        """Verifica la disponibilidad de un lugar según la hora"""
        venue = self.beliefs["venues"].get(venue_name, {})
        hours = venue.get("operating_hours", self.beliefs["operating_hours"]["standard"])
        
        is_open = (hours["open"] <= current_time <= hours["close"]) or \
                 (hours["open"] > hours["close"] and 
                  (current_time >= hours["open"] or current_time <= hours["close"]))
                  
        return {
            "venue": venue_name,
            "is_open": is_open,
            "opening_time": hours["open"].strftime("%H:%M"),
            "closing_time": hours["close"].strftime("%H:%M")
        }

    def _suggest_current_activities(self, location, current_time):
        """Sugiere actividades basadas en la ubicación y hora actual"""
        suggestion_prompt = f"""Sugiere actividades nocturnas en {location} para las {current_time.strftime('%H:%M')},
        considerando:
        - Hora actual
        - Tipos de lugares abiertos a esta hora
        - Ambiente típico del momento
        - Recomendaciones de seguridad
        La respuesta debe ser práctica y específica."""
        
        response = self.client.chat(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": suggestion_prompt},
                {"role": "user", "content": f"Actividades nocturnas en {location} ahora"}
            ]
        )
        return response.choices[0].message.content