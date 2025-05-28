from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
from .embeddings import get_embeddings
from langchain.schema import Document
import os
import json
from pathlib import Path

class VectorStorage:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.collection_name = "cuba_tourism_data"  # Nombre fijo
        self.persist_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "vector_db", "chroma_data")
        )
        
        self._initialize_empty_collection()

        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        self._sources_file = "sources.json"
        self.sources = self._load_sources()

    def _initialize_empty_collection(self):
        """Crea una colección vacía si no existe"""
        Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        ).add_documents([Document(page_content="dummy", metadata={})])
        
    def update_index(self):
        loader = JSONLoader(file_path='../data/processed/normalized_data.json')
        documents = loader.load()
        self.db.add_documents(documents)

    def get_documents(self):
        try:
            chroma_data = self.db._collection.get()
            return [
                Document(
                    page_content=content,
                    metadata=chroma_data["metadatas"][i]
                )
                for i, content in enumerate(chroma_data["documents"])
            ]
        except Exception as e:
            print(f"Error obteniendo documentos: {str(e)}")
            return []
        
    def similarity_search(self, query, k=3):
        """Búsqueda por similitud (para identify_outdated_sources)"""
        return self.db.similarity_search(query, k=k)
        
    def reload_data(self):
        try:
            # Eliminar colección existente
            if self.db._collection:
                self.db.delete_collection()
                
            # Nueva inicialización
            self.db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
            
            # Cargar datos
            json_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "data", 
                "processed", 
                "normalized_data.json"
            ))
            
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Archivo JSON no encontrado: {json_path}")
            
            loader = JSONLoader(
                file_path=json_path,
                jq_schema='.[] | {page_content: (.page_content // "" | tostring), metadata: .metadata}',
                text_content=False
            )
            
            documents = loader.load()
            if documents:
                self.db.add_documents(documents)
                self.db.persist()
                print(f"Se cargaron {len(documents)} documentos")
            else:
                print("Advertencia: No se cargaron documentos")
                
        except Exception as e:
            print(f"Error en reload_data: {str(e)}")
            raise
        
    def _load_sources(self):
        """Carga las fuentes desde el archivo JSON"""
        sources_path = Path(__file__).parent.parent / "data" / self._sources_file
        if sources_path.exists():
            with open(sources_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return [] 
        
    def get_sources(self):
        """Retorna la lista de URLs fuente para el crawler"""
        return self.sources

    def add_source(self, url):
        """Agrega una nueva fuente a la lista"""
        if url not in self.sources:
            self.sources.append(url)
            self._save_sources()

    def _save_sources(self):
        """Guarda las fuentes en el archivo JSON"""
        sources_path = Path(__file__).parent.parent / "data" / self._sources_file
        with open(sources_path, "w", encoding="utf-8") as f:
            json.dump(self.sources, f, indent=2)