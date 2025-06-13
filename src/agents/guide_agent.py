from .bdi_agent import BDIAgent
from langchain_community.retrievers import BM25Retriever
from crawlers.dynamic_crawler import DynamicCrawler
import time
import streamlit as st

class GuideAgent(BDIAgent):
    def __init__(self, vector_db):
        super().__init__("GuideBot", vector_db)
        self.desires = ["answer_accurately", "keep_data_updated"]
        self.retriever = BM25Retriever.from_documents(vector_db.get_documents())
    
    def deliberate(self):
        if (self.beliefs["data_freshness"] < 24):
            self.intentions.append("trigger_crawler")
        if self.beliefs["current_query"]:
            self.intentions.append("generate_response")
    
    def act(self):
        for intention in self.intentions:
            if intention == "trigger_crawler":
                self.trigger_crawler()
            elif intention == "generate_response":
                return self.generate_response()
    
    def trigger_crawler(self):
        crawler = DynamicCrawler()
        crawler.update_sources(self.vector_db.get_sources())
        self.vector_db.reload_data()
        st.session_state.last_update = time.time()
    
    def generate_response(self):
        context = self.retriever.get_relevant_documents(self.beliefs["current_query"])
        response = self.client.chat(
            model="mistral-medium",
            messages=[{
                "role": "user",
                "content": f"Contexto: {context}\nPregunta: {self.beliefs['current_query']}"
            }]
        )
        return response.choices[0].message.content