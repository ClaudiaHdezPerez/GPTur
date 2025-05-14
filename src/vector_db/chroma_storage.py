from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
from .embeddings import get_embeddings

class VectorStorage:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.db = Chroma(
            embedding_function=self.embeddings,
            persist_directory="../vector_db/chroma_data"
        )
    
    def update_index(self):
        loader = JSONLoader(file_path='../data/processed/normalized_data.json')
        documents = loader.load()
        self.db.add_documents(documents)
        
    def reload_data(self):
        self.db.delete_collection()  # Limpiar datos antiguos
        loader = JSONLoader('../data/processed/normalized_data.json')
        self.db.add_documents(loader.load())