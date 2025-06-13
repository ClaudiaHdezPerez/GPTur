from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """Retorna el modelo de embeddings para documentos en español"""
    return HuggingFaceEmbeddings(
        model_name="hiiamsid/sentence_similarity_spanish_es",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )