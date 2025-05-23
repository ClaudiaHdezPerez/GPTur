from .base_agent import BaseAgent

class RetrieverAgent(BaseAgent):
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def can_handle(self, task):
        return task.get("type") == "retrieve"

    def handle(self, task, context):
        query = task.get("query")
        return self.vector_db.similarity_search(query)