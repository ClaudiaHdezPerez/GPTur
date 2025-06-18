from .base_agent import BaseAgent

class RetrieverAgent(BaseAgent):
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def can_handle(self, task):
        """
        Check if the agent can handle the given task.

        Args:
            task (dict): The task to be evaluated

        Returns:
            bool: True if the task is of type 'retrieve', False otherwise
        """
        return task.get("type") == "retrieve"

    def handle(self, task, context):
        """
        Process the retrieval task and fetch relevant documents.

        Args:
            task (dict): Contains the query to search for
            context (dict): Additional context information (not used in retrieval)

        Returns:
            list: List of relevant documents from the vector database
        """
        query = task.get("query")
        return self.vector_db.similarity_search(query)