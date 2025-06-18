from mistralai.client import MistralClient
import streamlit as st

class BDIAgent:
    def __init__(self, name, vector_db=None):
        self.name = name
        self.vector_db = vector_db
        self.client = MistralClient(api_key="XEV0fCx3MqiG9HqVkGc4Hy5qyD3WwPHr")
        
        # Componentes BDI
        self.beliefs = {"context": []}  # B: Conjunto de creencias
        self.desires = []               # D: Conjunto de deseos
        self.intentions = []            # I: Conjunto de intenciones
        self.plans = {}                 # P: Conjunto de planes disponibles
        
    def action(self, percept):
        """
        Implementación del algoritmo BDI según el pseudocódigo:
        function action(p : P): A
            B = brf(B,p)
            D = options(B,D,I)
            I = filter(B,D,I)
            return execute(I)
        """
        # Paso 1: Actualizar creencias (belief revision function)
        print("Percept", percept)
        self.brf(percept)
        
        print("Beliefs:", self.beliefs)
        
        # Paso 2: Generar opciones basadas en B,D,I
        options = self.generate_options()
        
        # Paso 3: Filtrar y seleccionar intenciones
        self.intentions = self.filter(options)
        
        # Paso 4: Ejecutar intenciones y retornar acción
        return self.execute()
        
    def brf(self, percept):
        """Belief Revision Function
        Actualiza las creencias basado en la percepción"""
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
        """Options: D = options(B,D,I)
        Genera opciones basadas en creencias, deseos e intenciones"""
        options = []
        for desire in self.desires:
            plan = self.plans.get(desire)
            if plan and self._is_plan_relevant(plan):
                options.append(plan)
                
        return options
        
    def _is_plan_relevant(self, plan) -> bool:
        """Verifica si un plan es relevante dado el estado actual
        Debe ser implementado por los agentes específicos"""
        pass
        
    def filter(self, options: list) -> list:
        """Filter: I = filter(B,D,I)
        Filtra las opciones para determinar nuevas intenciones"""
        filtered_intentions = []
        for option in options:
            if self._is_achievable(option) and self._is_compatible(option):
                filtered_intentions.append(option)
                
        return filtered_intentions
        
    def _is_achievable(self, plan) -> bool:
        """Verifica si un plan es alcanzable
        Debe ser implementado por los agentes específicos"""
        pass
        
    def _is_compatible(self, plan) -> bool:
        """Verifica si un plan es compatible con las intenciones actuales
        Debe ser implementado por los agentes específicos"""
        pass
        
    def execute(self):
        """Execute: return execute(I)
        Ejecuta las intenciones seleccionadas y retorna la acción"""
        if not self.intentions:
            return None
            
        for intention in self.intentions:
            action = self._get_next_action(intention)
            if action:
                result = self._perform_action(action)
                # Si el agente es especializado (tiene specialization) y tiene blackboard
                if hasattr(self, 'specialization') and hasattr(self, 'blackboard') and result:
                    self.blackboard.write(self.name, result)
                return result
        return None
        
    def _get_next_action(self, intention):
        """Determina la siguiente acción para una intención
        Debe ser implementado por los agentes específicos"""
        pass
        
    def _perform_action(self, action):
        """Ejecuta una acción específica
        Debe ser implementado por los agentes específicos"""
        pass
        
    def check_data_freshness(self):
        """Verifica la frescura de los datos"""
        return st.session_state.get("last_update", 0)