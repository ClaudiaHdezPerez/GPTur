from mistralai.client import MistralClient
from vector_db.chroma_storage import VectorStorage

class CubaChatbot:
    def __init__(self):
        self.vector_db = VectorStorage()
        self.mistral_client = MistralClient(api_key="XEV0fCx3MqiG9HqVkGc4Hy5qyD3WwPHr")