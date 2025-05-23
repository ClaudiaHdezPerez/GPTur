from .base_agent import BaseAgent

class GapDetectorAgent(BaseAgent):
    def __init__(self, detector):
        self.detector = detector

    def can_handle(self, task):
        return task.get("type") == "detect_gap"

    def handle(self, task, context):
        prompt = task.get("prompt")
        response = task.get("response")
        return self.detector.check_accuracy(prompt, response)