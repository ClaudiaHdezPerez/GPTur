from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """Retorna el modelo de embeddings para documentos en español"""
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )