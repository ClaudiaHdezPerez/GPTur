from .bdi_agent import BDIAgent
from .blackboard import Blackboard
from .lodging_agent import LodgingAgent
from .historic_agent import HistoricAgent 
from .nightlife_agent import NightlifeAgent
from .gastronomy_agent import GastronomyAgent
from langchain_community.retrievers import BM25Retriever
from crawlers.dynamic_crawler import DynamicCrawler
import time
import streamlit as st
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event

class GuideAgent(BDIAgent):
    def __init__(self, vector_db):
        super().__init__("GuideBot", vector_db)
        self.desires = ["answer_accurately", "keep_data_updated", "coordinate_agents"]
        self.retriever = BM25Retriever.from_documents(vector_db.get_documents())
        self.blackboard = Blackboard()
        self.stop_event = Event()
        
        # Initialize specialized agents
        self.specialized_agents = {
            "lodging": LodgingAgent("LodgingBot", vector_db),
            "historic": HistoricAgent("HistoricBot", vector_db),
            "nightlife": NightlifeAgent("NightlifeBot", vector_db),
            "gastronomy": GastronomyAgent("GastronomyBot", vector_db)
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.specialized_agents))
    
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
    
    def receive_message(self, sender, message):
        # When receiving a message from a specialized agent, write it to the blackboard
        if hasattr(sender, "specialization"):
            self.blackboard.write(sender.name, message)
    
    def process_agent_query(self, agent, query, relevant_docs):
        """Process query for a specific agent in a thread-safe manner"""
        if self.stop_event.is_set():
            return None
            
        try:
            if "lodging" in agent.specialization:
                results = agent.search_accommodations(query, relevant_docs)
            elif "historic" in agent.specialization:
                results = agent.search_historic_sites(query, relevant_docs)
            elif "nightlife" in agent.specialization:
                results = agent.search_nightlife(query, relevant_docs)
            elif "gastronomy" in agent.specialization:
                results = agent.search_restaurants(query, relevant_docs)
            agent.communicate(self, results)
            return results
        except Exception as e:
            print(f"Error with {agent.specialization} agent: {str(e)}")
            return None

    def generate_response(self):
        # Create a unique problem ID for this query
        problem_id = str(uuid.uuid4())
        self.blackboard.set_current_problem(problem_id)
        query = self.beliefs["current_query"]
        self.stop_event.clear()

        # Get relevant documents first
        relevant_docs = self.vector_db.similarity_search(query)

        # Submit tasks to thread pool
        future_to_agent = {
            self.thread_pool.submit(
                self.process_agent_query, agent, query, relevant_docs
            ): agent.specialization for agent in self.specialized_agents.values()
        }

        # Wait for all tasks to complete or handle timeouts
        try:
            for future in as_completed(future_to_agent, timeout=30):  # 30 second timeout
                agent_type = future_to_agent[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Agent {agent_type} generated an exception: {str(e)}")
        except TimeoutError:
            print("Some agents did not complete in time")
            self.stop_event.set()  # Signal threads to stop

        # Collect all contributions from the blackboard
        contributions = self.blackboard.read(problem_id)
        
        # Generate comprehensive response using all contributions
        context = "\n".join([f"{c['agent']}: {c['contribution']}" for c in contributions])
        if not contributions:
            # Fallback if no agent could provide specific information
            response = self.client.chat(
                model="mistral-medium",
                messages=[{
                    "role": "user",
                    "content": f"""Based on these documents about {query}, provide a helpful response:
                    {[doc.page_content for doc in relevant_docs]}"""
                }]
            )
        else:
            response = self.client.chat(
                model="mistral-medium",
                messages=[{
                    "role": "user",                    
                    "content": f"""You are an expert travel guide specializing in Cuban tourism, 
                    with extensive knowledge of the country's culture, history, attractions, and tourist services. 
                    Using this expertise and the following information, provide a comprehensive response:
                    Question: {query}
                    Specialized Information:
                    {context}
                    
                    Please provide a well-organized response that integrates all relevant information. Make sure to maintain a friendly and knowledgeable tone, highlighting the unique aspects of Cuban tourism and culture in your response."""
                }]
            )
        
        # Clean up the blackboard
        self.blackboard.clear_problem(problem_id)
        return response.choices[0].message.content

    def __del__(self):
        """Cleanup thread pool on deletion"""
        self.stop_event.set()
        self.thread_pool.shutdown(wait=True)