import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    CRAWL_INTERVAL = 86400  # 24 horas en segundos
    VECTOR_DB_PATH = "../vector_db/chroma_data"