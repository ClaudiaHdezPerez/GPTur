from mistralai.client import MistralClient
from vector_db.chroma_storage import VectorStorage

class CubaChatbot:
    def __init__(self):
        self.vector_db = VectorStorage()
        self.mistral_client = MistralClient(api_key="XEV0fCx3MqiG9HqVkGc4Hy5qyD3WwPHr")
        
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