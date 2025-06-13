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
            ]
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