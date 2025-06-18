from .base_agent import BaseAgent

class UpdaterAgent(BaseAgent):
    def __init__(self, updater):
        self.updater = updater

    def can_handle(self, task):
        """
        Check if the agent can handle the given task.

        Args:
            task (dict): The task to be evaluated

        Returns:
            bool: True if the task is of type 'update_sources', False otherwise
        """
        return task.get("type") == "update_sources"

    def handle(self, task, context):
        """
        Process the update task and trigger source updates.

        Args:
            task (dict): Contains the sources to be updated
            context (dict): Additional context information (not used in updating)

        Returns:
            bool: True if update was successful, False otherwise
        """
        sources = task.get("sources")
        return self.updater.update_sources(sources)