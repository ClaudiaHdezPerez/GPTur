from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """Retorna el modelo de embeddings para documentos en español"""
    return HuggingFaceEmbeddings(
        model_name="hiiamsid/sentence_similarity_spanish_es",
        model_kwargs={"device": "cpu"},  # Usar GPU si está disponible: "cuda"
        encode_kwargs={"normalize_embeddings": True}
    )