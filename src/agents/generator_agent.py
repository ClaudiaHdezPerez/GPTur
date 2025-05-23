from .base_agent import BaseAgent

class GeneratorAgent(BaseAgent):
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def can_handle(self, task):
        return task.get("type") == "generate"

    def handle(self, task, context):
        prompt = task.get("prompt")
        return self.chatbot.generate_response(prompt)