from mistralai.client import MistralClient
from vector_db.chroma_storage import VectorStorage
import os

class CubaChatbot:
    def __init__(self):
        self.vector_db = VectorStorage()
        self.mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
        
    def generate_response(self, query):
        context = self.vector_db.similarity_search(query, k=3)
        messages = [
            {"role": "system", "content": f"Contexto: {context}"},
            {"role": "user", "content": query}
        ]
        return self.mistral_client.chat(
            model="open-mixtral-8x7b",
            messages=messages
        )