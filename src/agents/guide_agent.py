from nlp.processor import NLPProcessor
from .bdi_agent import BDIAgent
from .blackboard import Blackboard
from .lodging_agent import LodgingAgent
from .historic_agent import HistoricAgent 
from .nightlife_agent import NightlifeAgent
from .gastronomy_agent import GastronomyAgent
from .generator_agent import _convert_docs_to_string
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
        
        self.desires = [
            "responder_consultas"
        ]
        
        self.plans = {
            "responder_consultas": {
                "objetivo": "generar_respuesta",
                "precondiciones": ["tiene_consulta", "datos_disponibles"],
                "acciones": ["generar_respuesta"]
            }
        }
        
        self.retriever = BM25Retriever.from_documents(vector_db.get_documents())
        self.blackboard = Blackboard()
        self.stop_event = Event()
        self.nlp_processor = NLPProcessor()
        
        self.specialized_agents = {
            "lodging": LodgingAgent("LodgingBot", vector_db),
            "historic": HistoricAgent("HistoricBot", vector_db),
            "nightlife": NightlifeAgent("NightlifeBot", vector_db),
            "gastronomy": GastronomyAgent("GastronomyBot", vector_db)
        }

        self.thread_pool = ThreadPoolExecutor(
            max_workers=len(self.specialized_agents),
            thread_name_prefix="GPTur_Agent"
        )
        
    def _is_plan_relevant(self, plan) -> bool:
        """
        Check if a plan is relevant for the current state.

        Args:
            plan (dict): The plan to evaluate

        Returns:
            bool: True if the plan's objective matches current beliefs, False otherwise
        """
        if plan["objetivo"] == "generar_respuesta":
            return "current_query" in self.beliefs
        return False

    def _is_achievable(self, plan) -> bool:
        """
        Verify if a plan is achievable based on its preconditions.

        Args:
            plan (dict): The plan to evaluate

        Returns:
            bool: True if all preconditions are met, False otherwise
        """
        for precondition in plan["precondiciones"]:
            if not self._check_precondition(precondition):
                return False
        return True

    def _check_precondition(self, precondition) -> bool:
        """
        Check if a specific precondition is satisfied.

        Args:
            precondition (str): The precondition to verify

        Returns:
            bool: True if the precondition is met, False otherwise
        """
        if precondition == "tiene_consulta":
            return "current_query" in self.beliefs
        elif precondition == "datos_disponibles":
            return bool(self.vector_db.get_documents())
        return False

    def _is_compatible(self, plan) -> bool:
        """Verifica si un plan es compatible con las intenciones actuales"""
        return True

    def _get_next_action(self, intention) -> str:
        """Determina la siguiente acción para una intención"""
        if not intention.get("acciones"):
            return None
            
        return intention["acciones"][0]

    def _perform_action(self, action):
        """Ejecuta una acción específica"""
        if action == "generar_respuesta":
            return self.generate_response()
        return None
    
    def trigger_crawler(self):
        """
        Trigger the web crawler to update information sources and refresh the vector database.
        Updates the last update timestamp in the session state.
        """
        crawler = DynamicCrawler()
        crawler.update_sources(self.vector_db.get_sources())
        self.vector_db.reload_data()
        st.session_state.last_update = time.time()
        
    def process_agent_query(self, agent : BDIAgent, query, relevant_docs):
        """
        Process a query using a specific specialized agent in a thread-safe manner.

        Args:
            agent (BDIAgent): The specialized agent to handle the query
            query (str): The user's query
            relevant_docs (str): Relevant context documents

        Returns:
            Any: The agent's response or None if processing fails
        """
        if self.stop_event.is_set():
            return None
        
        percept = (query, relevant_docs)
            
        try:
            results = agent.action(percept)
            self.blackboard.write(agent.name, results)
            return results
        except Exception as e:
            print(f"Error with {agent.specialization} agent: {str(e)}")
            return None
            
    def preprocess_query(self, query: str):
        """
        Preprocess a user query using NLP techniques for enhanced understanding.

        Args:
            query (str): The raw user query

        Returns:
            dict: Processed query information including text, entities, keywords, and sentiment
        """
        processed_text = self.nlp_processor.preprocess_text(query)
        entities = self.nlp_processor.extract_entities(query)
        keywords = self.nlp_processor.extract_keywords(query)
        sentiment = self.nlp_processor.analyze_sentiment(query)
        
        return {
            "processed_text": processed_text,
            "entities": entities,
            "keywords": keywords,
            "sentiment": sentiment
        }

    def generate_response(self):
        """
        Generate a comprehensive response by coordinating multiple specialized agents.
        
        The method:
        1. Creates a unique problem ID
        2. Analyzes the query using NLP
        3. Retrieves relevant documents
        4. Coordinates specialized agents
        5. Combines their contributions
        
        Returns:
            str: The final consolidated response
        """
        problem_id = str(uuid.uuid4())
        self.blackboard.set_current_problem(problem_id)
        query = self.beliefs["current_query"]
        self.stop_event.clear()

        query_analysis = self.preprocess_query(query)
        
        search_query = f"{query_analysis['processed_text']} {' '.join(query_analysis['keywords'])}"
        
        relevant_docs = self.vector_db.similarity_search(search_query)
        document_text = _convert_docs_to_string(relevant_docs)

        future_to_agent = {
            self.thread_pool.submit(
                self.process_agent_query, agent, query, document_text
            ): agent.specialization for agent in self.specialized_agents.values()
        }

        try:
            for future in as_completed(future_to_agent, timeout=120):
                agent_type = future_to_agent[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Agent {agent_type} generated an exception: {str(e)}")
        except TimeoutError:
            print("Some agents did not complete in time")
            self.stop_event.set()

        contributions = self.blackboard.read(problem_id)

        context = "\n".join([f"{c['agent']}: {c['contribution']}" for c in contributions])
        if not contributions:
            response = self.client.chat(
                model="mistral-medium",
                messages=[{
                    "role": "user",
                    "content": f"""Based on these documents about {query}, provide a helpful response:
                    Context documents: {[doc.page_content for doc in relevant_docs]}
                    Detected entities: {query_analysis['entities']}
                    User sentiment: {query_analysis['sentiment']['sentiment']}"""
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
                    Detected entities: {query_analysis['entities']}
                    User sentiment: {query_analysis['sentiment']['sentiment']}
                    Specialized Information:
                    {context}
                    
                    Please provide a well-organized response that integrates all relevant information.
                    Adapt your tone to match the user's sentiment ({query_analysis['sentiment']['sentiment']}).
                    Make sure to address all mentioned entities and maintain a friendly and knowledgeable tone,
                    highlighting the unique aspects of Cuban tourism and culture in your response."""
                }]
            )
        
        self.blackboard.clear_problem(problem_id)
        return response.choices[0].message.content

    def __del__(self):
        """
        Cleanup method to ensure proper shutdown of thread pool when the agent is destroyed.
        Sets the stop event and performs an orderly shutdown of all threads.
        """
        self.stop_event.set()
        self.thread_pool.shutdown(wait=True)