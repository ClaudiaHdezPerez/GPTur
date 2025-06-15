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
                    "open": time(20, 0),  # 8:00 PM
                    "close": time(2, 0)    # 2:00 AM
                },
                "late": {
                    "open": time(22, 0),   # 10:00 PM
                    "close": time(6, 0)    # 6:00 AM
                }
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