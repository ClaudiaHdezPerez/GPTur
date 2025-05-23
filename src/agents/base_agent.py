class BaseAgent:
    def can_handle(self, task):
        raise NotImplementedError

    def handle(self, task, context):
        raise NotImplementedError