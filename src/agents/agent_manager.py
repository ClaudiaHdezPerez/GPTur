class AgentManager:
    def __init__(self, agents):
        self.agents = agents

    def dispatch(self, task, context=None):
        for agent in self.agents:
            if agent.can_handle(task):
                return agent.handle(task, context)
        raise Exception("No agent can handle this task")