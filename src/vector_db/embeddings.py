from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Get the embeddings model for Spanish language documents.
    Uses the multilingual-e5-small model from HuggingFace, optimized for CPU.

    Returns:
        HuggingFaceEmbeddings: Configured embeddings model with normalized embeddings
    """
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )