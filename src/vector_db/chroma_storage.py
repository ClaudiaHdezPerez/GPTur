from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from .embeddings import get_embeddings
from langchain_core.documents import Document
import os
import json
from pathlib import Path
import chromadb

class VectorStorage:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.collection_name = "cuba_tourism"
        self.persist_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "vector_db", "chroma_data"))
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        self._initialize_collection()
        
        self.db = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        self._sources_file = "sources.json"
        self.sources = self._load_sources()
    
    def _initialize_collection(self):
        """
        Initialize or retrieve the Chroma collection.
        Creates a new collection if it doesn't exist or adds a dummy document if empty.
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            if collection.count() == 0:
                collection.add(
                    documents=["dummy"],
                    metadatas=[{}],
                    ids=["dummy_id"]
                )
        except Exception as e:
            print(f"Creando nueva colección: {str(e)}")
            if (not any([x for x in self.client.list_collections() if x.name == self.collection_name])):
                self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

    def update_index(self):
        """
        Update the vector index with new documents from the JSON data file.
        Loads and processes normalized data, adding new documents to the collection.
        """
        json_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "data", 
            "processed", 
            "normalized_data.json"
        ))
        
        loader = JSONLoader(
            file_path=json_path,
            jq_schema='.[] | {page_content: ([.title, .description, .content] | join("\n")), metadata: {url: .url, city: .city, attractions: .attractions, timestamp: .timestamp, source: .metadata.source, crawl_date: .metadata.crawl_date, language: .metadata.language}}',
            text_content=False
        )
        
        documents = loader.load()
        if documents:
            self.db.add_documents(documents)
            print(f"Índice actualizado con {len(documents)} documentos")

    def get_documents(self):
        """
        Retrieve all documents from the collection.

        Returns:
            list: List of Document objects containing page content and metadata
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            chroma_data = collection.get()
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
        
    def similarity_search(self, query, k=4):
        """
        Perform similarity search on the document collection.

        Args:
            query (str): The search query
            k (int, optional): Number of results to return. Defaults to 4

        Returns:
            list: Top k similar documents
        """
        return self.db.similarity_search(query, k=k)
        
    def reload_data(self):
        """
        Completely reload data from the JSON file.
        Deletes existing collection and recreates it with fresh data.
        
        Raises:
            FileNotFoundError: If the JSON data file is not found
            Exception: For other errors during reload
        """
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            
            self.db = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
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
                jq_schema='.[] | {page_content: (.content // "" | tostring), metadata: {url: .url, city: .city, attractions: .attractions, timestamp: .timestamp, source: .metadata.source, crawl_date: .metadata.crawl_date, language: .metadata.language}}',
                text_content=False
            )
            
            documents = loader.load()
            if documents:
                self.db.add_documents(documents)
                print(f"Se cargaron {len(documents)} documentos")
            else:
                print("Advertencia: No se cargaron documentos")
                
        except Exception as e:
            print(f"Error en reload_data: {str(e)}")
            raise
        
    def _load_sources(self):
        """
        Load source URLs from the JSON file.

        Returns:
            list: List of source URLs or empty list if file doesn't exist
        """
        sources_path = Path(__file__).parent.parent / "data" / self._sources_file
        if sources_path.exists():
            with open(sources_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return [] 
        
    def get_sources(self):
        """
        Get the list of source URLs for the crawler.

        Returns:
            list: List of source URLs
        """
        return self.sources

    def add_source(self, url):
        """
        Add a new source URL to the list.

        Args:
            url (str): URL to add as a source
        """
        if url not in self.sources:
            self.sources.append(url)
            self._save_sources()

    def _save_sources(self):
        """
        Save the current list of sources to the JSON file.
        Updates the sources file with any new additions.
        """
        sources_path = Path(__file__).parent.parent / "data" / self._sources_file
        with open(sources_path, "w", encoding="utf-8") as f:
            json.dump(self.sources, f, indent=2)