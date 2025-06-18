from .base_agent import BaseAgent

class GapDetectorAgent(BaseAgent):
    def __init__(self, detector):
        self.detector = detector

    def can_handle(self, task):
        """
        Check if the agent can handle the given task.

        Args:
            task (dict): The task to be evaluated

        Returns:
            bool: True if the task is of type 'detect_gap', False otherwise
        """
        return task.get("type") == "detect_gap"

    def handle(self, task, context):
        """
        Process the task to detect knowledge gaps in responses.

        Args:
            task (dict): Contains the prompt and response to analyze
            context (dict): Additional context information

        Returns:
            bool: True if a knowledge gap is detected, False otherwise
        """
        prompt = task.get("prompt")
        response = task.get("response")
        return self.detector.check_accuracy(prompt, response)