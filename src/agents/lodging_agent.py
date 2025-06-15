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
            "price_ranges": ["economic", "moderate", "luxury"]
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