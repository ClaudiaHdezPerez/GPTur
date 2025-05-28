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
    
    def update_beliefs(self, user_query, chat_history):
        self.beliefs.update({
            "current_query": user_query,
            "history": chat_history,
            "data_freshness": self.check_data_freshness()
        })
    
    def check_data_freshness(self):
        return st.session_state.get("last_update", 0)
    
    def communicate(self, recipient, message):
        recipient.receive_message(self.name, message)
    
    def receive_message(self, sender, message):
        self.beliefs["context"].append(f"{sender}: {message}")