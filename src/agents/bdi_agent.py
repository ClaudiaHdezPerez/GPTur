from mistralai.client import MistralClient
import streamlit as st

class BDIAgent:
    def __init__(self, name, vector_db=None):
        self.name = name
        self.vector_db = vector_db
        self.client = MistralClient(api_key="XEV0fCx3MqiG9HqVkGc4Hy5qyD3WwPHr")
        
        self.beliefs = {"context": []}
        self.desires = []
        self.intentions = []
        self.plans = {}
        
    def action(self, percept):
        """
        Execute the BDI algorithm based on the pseudocode:
        function action(p : P): A
            B = brf(B,p)
            D = options(B,D,I)
            I = filter(B,D,I)
            return execute(I)

        Args:
            percept: The perception input for the agent

        Returns:
            The result of executing the selected intentions
        """
        self.brf(percept)
        options = self.generate_options()
        self.intentions = self.filter(options)
        
        return self.execute()
        
    def brf(self, percept):
        """
        Belief Revision Function that updates the agent's beliefs based on new perception.

        Args:
            percept: Can be either a dictionary of beliefs or a tuple containing (user_query, chat_history)
        """
        if isinstance(percept, dict):
            self.beliefs.update(percept)
        elif isinstance(percept, tuple) and len(percept) == 2:
            user_query, chat_history = percept
            self.beliefs.update({
                "current_query": user_query,
                "history": chat_history,
                "data_freshness": self.check_data_freshness()
            })
            
    def generate_options(self) -> list:
        """
        Generate options based on current beliefs, desires, and intentions.

        Returns:
            list: A list of viable plans based on current desires
        """
        options = []
        for desire in self.desires:
            plan = self.plans.get(desire)
            if plan and self._is_plan_relevant(plan):
                options.append(plan)
                
        return options
        
    def _is_plan_relevant(self, plan) -> bool:
        """
        Check if a plan is relevant given the current state.

        Args:
            plan: The plan to evaluate

        Returns:
            bool: True if the plan is relevant, False otherwise
        """
        # Debe ser implementado por los agentes específicos
        pass
        
    def filter(self, options: list) -> list:
        """
        Filter options to determine new intentions based on achievability and compatibility.

        Args:
            options: List of potential plans

        Returns:
            list: Filtered list of intentions
        """
        filtered_intentions = []
        for option in options:
            if self._is_achievable(option) and self._is_compatible(option):
                filtered_intentions.append(option)
                
        return filtered_intentions
        
    def _is_achievable(self, plan) -> bool:
        """
        Verify if a plan is achievable given current circumstances.

        Args:
            plan: The plan to evaluate

        Returns:
            bool: True if the plan is achievable, False otherwise
        """
        # Debe ser implementado por los agentes específicos
        pass
        
    def _is_compatible(self, plan) -> bool:
        """
        Check if a plan is compatible with current intentions.

        Args:
            plan: The plan to evaluate

        Returns:
            bool: True if the plan is compatible, False otherwise
        """
        # Debe ser implementado por los agentes específicos
        pass
        
    def execute(self):
        """
        Execute the selected intentions and return the resulting action.

        Returns:
            The result of performing the selected action, or None if no action is available
        """
        if not self.intentions:
            return None
            
        for intention in self.intentions:
            action = self._get_next_action(intention)
            if action:
                result = self._perform_action(action)
                if hasattr(self, 'specialization') and hasattr(self, 'blackboard') and result:
                    self.blackboard.write(self.name, result)
                return result
        return None
        
    def _get_next_action(self, intention):
        """
        Determine the next action for a given intention.

        Args:
            intention: The intention to process

        Returns:
            The next action to be performed
        """
        # Debe ser implementado por los agentes específicos
        pass
        
    def _perform_action(self, action):
        """
        Execute a specific action.

        Args:
            action: The action to perform

        Returns:
            The result of performing the action
        """
        # Debe ser implementado por los agentes específicos
        pass
        
    def check_data_freshness(self):
        """
        Check the freshness of the data in the session state.

        Returns:
            The timestamp of the last update from session state
        """
        return st.session_state.get("last_update", 0)