from .bdi_agent import BDIAgent
from .blackboard import Blackboard
from datetime import datetime

class HistoricAgent(BDIAgent):
    def __init__(self, name, vector_db=None):
        super().__init__(name, vector_db)
        self.specialization = "historic"
        self.blackboard = Blackboard()
        
        # Inicializar creencias específicas de sitios históricos
        self.beliefs = {
            "historic_sites": {},
            "cultural_events": {},
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
            ]
        }
        
        # Definir deseos del agente histórico
        self.desires = [
            "buscar_sitios",
            "recomendar_lugares",
            "informar_eventos",
            "detallar_historia"
        ]
        
        # Definir planes disponibles
        self.plans = {
            "buscar_sitios": {
                "objetivo": "encontrar_sitios",
                "precondiciones": ["tiene_ubicacion"],
                "acciones": ["buscar_en_db", "clasificar_sitios", "formatear_respuesta"]
            },
            "recomendar_lugares": {
                "objetivo": "dar_recomendaciones",
                "precondiciones": ["tiene_ubicacion", "tiene_preferencias"],
                "acciones": ["analizar_intereses", "filtrar_sitios", "generar_recomendaciones"]
            },
            "informar_eventos": {
                "objetivo": "mostrar_eventos",
                "precondiciones": ["tiene_ubicacion", "tiene_fecha"],
                "acciones": ["buscar_eventos", "filtrar_por_fecha", "formatear_eventos"]
            },
            "detallar_historia": {
                "objetivo": "explicar_historia",
                "precondiciones": ["tiene_sitio"],
                "acciones": ["buscar_detalles", "compilar_historia", "generar_narracion"]
            }
        }

    def search_historic_sites(self, query, relevant_docs=None):
        """Search for historic sites matching the query"""
        if relevant_docs is None:
            relevant_docs = self.vector_db.similarity_search(
                self._build_historic_query(query)
            )
        
        # Process the relevant documents
        processed_results = []
        for doc in relevant_docs:
            if any(site_type in doc.page_content.lower() for site_type in self.beliefs["site_types"]):
                processed_results.append(doc.page_content)
                
        # Classify the results
        classified_sites = self._classify_historic_sites(processed_results)
        
        # Update beliefs
        location = query.split()[0]  # Simple location extraction
        if location not in self.beliefs["historic_sites"]:
            self.beliefs["historic_sites"][location] = classified_sites
            
        return self._format_historic_results(classified_sites)

    def _build_historic_query(self, location, site_type=None):
        """Build optimized search query for historic sites"""
        query_parts = [f"historic and cultural attractions in {location}"]
        
        if site_type:
            if site_type in self.beliefs["site_types"]:
                query_parts.append(f"focusing on {site_type}")
                
        return " ".join(query_parts)

    def _classify_historic_sites(self, results):
        """Classify historic sites by type"""
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
        """Format historic sites results in a user-friendly way"""
        formatted = []
        
        for site_type, sites in classified_sites.items():
            if sites:
                formatted.append(f"\n{site_type.upper().replace('_', ' ')}:")
                for site in sites:
                    formatted.append(f"- {site}")
                    
        return "\n".join(formatted)

    def search_cultural_events(self, location, date_range=None):
        """Search for cultural events in a location"""
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
        """Format cultural events in a user-friendly way"""
        formatted = ["\nUPCOMING CULTURAL EVENTS:"]
        
        for event in events:
            formatted.append(f"- {event}")
            
        return "\n".join(formatted)

    def get_site_details(self, site_name, location):
        """Get detailed information about a specific historic site"""
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

    def communicate(self, recipient, message):
        if isinstance(recipient, BDIAgent):
            recipient.receive_message(self, message)
            
    def get_recommendations(self, destination):
        """Get historic site recommendations for a specific destination"""
        # Primero buscar sitios históricos
        historic_results = self.search_historic_sites(destination)
        
        # Preparar el prompt para el LLM
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
        if plan["objetivo"] == "encontrar_sitios":
            return "destination" in self.beliefs
        elif plan["objetivo"] == "dar_recomendaciones":
            return "current_query" in self.beliefs and "destination" in self.beliefs
        elif plan["objetivo"] == "mostrar_eventos":
            return "destination" in self.beliefs and "date_range" in self.beliefs
        elif plan["objetivo"] == "explicar_historia":
            return "site_name" in self.beliefs and "destination" in self.beliefs
        return False

    def _check_precondition(self, precondition) -> bool:
        """Verifica una precondición específica"""
        if precondition == "tiene_ubicacion":
            return "destination" in self.beliefs
        elif precondition == "tiene_preferencias":
            return bool(self.beliefs.get("preferences", {}))
        elif precondition == "tiene_fecha":
            return "date_range" in self.beliefs
        elif precondition == "tiene_sitio":
            return "site_name" in self.beliefs
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
            return self.search_historic_sites(self.beliefs["destination"])
        elif action == "generar_recomendaciones":
            return self.get_site_details(
                self.beliefs["site_name"],
                self.beliefs["destination"]
            )
        elif action == "buscar_eventos":
            return self.search_cultural_events(
                self.beliefs["destination"],
                self.beliefs.get("date_range")
            )
        elif action == "compilar_historia":
            return self._compile_historic_details(
                self.beliefs["site_name"],
                self.beliefs["destination"]
            )
        return None

    def get_recommendations(self, destination):
        """Obtiene recomendaciones usando el ciclo BDI"""
        # Crear percepción con el destino
        percept = {"destination": destination}
        
        # Ejecutar ciclo BDI y obtener acción
        return self.action(percept)

    def _compile_historic_details(self, site_name, location):
        """Compila detalles históricos de un sitio"""
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