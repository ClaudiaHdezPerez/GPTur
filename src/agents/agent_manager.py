class AgentManager:
    def __init__(self, agents):
        self.agents = agents

    def dispatch(self, task, context=None):
        '''
        dispatches a task to the appropriate agent based on the task type.
        '''
        for agent in self.agents:
            if agent.can_handle(task):
                return agent.handle(task, context)
        raise Exception("No agent can handle this task")