from mistralai.client import MistralClient
from vector_db.chroma_storage import VectorStorage
from nlp.processor import NLPProcessor
from typing import Dict, Any, List

class CubaChatbot:
    def __init__(self):
        self.vector_db = VectorStorage()
        self.mistral_client = MistralClient(api_key="XEV0fCx3MqiG9HqVkGc4Hy5qyD3WwPHr")
        self.nlp_processor = NLPProcessor()
        
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocesses the user query using NLP techniques
        """
        processed_text = self.nlp_processor.preprocess_text(query)
        entities = self.nlp_processor.extract_entities(query)
        keywords = self.nlp_processor.extract_keywords(query)
        sentiment = self.nlp_processor.analyze_sentiment(query)
        
        return {
            "processed_text": processed_text,
            "entities": entities,
            "keywords": keywords,
            "sentiment": sentiment
        }
        
    def generate_response(self, query: str) -> Dict[str, Any]:
        # Process the query using NLP
        query_analysis = self.preprocess_query(query)
        
        # Use the processed text and extracted entities for better context matching
        search_query = f"{query_analysis['processed_text']} {' '.join(query_analysis['keywords'])}"
        context = self.vector_db.similarity_search(search_query, k=3)
        
        # Enhance system message with NLP insights
        system_message = (
            f"Contexto: {context}\n"
            f"Entidades detectadas: {query_analysis['entities']}\n"
            f"Sentimiento del usuario: {query_analysis['sentiment']['sentiment']}\n"
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        response = self.mistral_client.chat(
            model="open-mixtral-8x7b",
            messages=messages
        )
        
        return {
            "response": response,
            "analysis": query_analysis
        }
    
    def find_relevant_info(self, text: str) -> List[str]:
        """
        Uses NLP to split and process longer texts into meaningful chunks
        """
        chunks = self.nlp_processor.split_text(text)
        return [chunk for chunk in chunks if self.nlp_processor.analyze_sentiment(chunk)["score"] > 0]