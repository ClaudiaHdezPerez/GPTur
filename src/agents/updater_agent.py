from .base_agent import BaseAgent

class UpdaterAgent(BaseAgent):
    def __init__(self, updater):
        self.updater = updater

    def can_handle(self, task):
        return task.get("type") == "update_sources"

    def handle(self, task, context):
        sources = task.get("sources")
        return self.updater.update_sources(sources)