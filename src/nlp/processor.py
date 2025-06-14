import spacy
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class NLPProcessor:
    def __init__(self):
        # Load Spanish language model
        self.nlp = spacy.load("es_core_news_md")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses text by removing unnecessary whitespace, normalizing text,
        and performing basic cleaning.
        """
        doc = self.nlp(text.lower().strip())
        # Remove stopwords and punctuation, normalize text
        processed = " ".join([token.lemma_ for token in doc 
                            if not token.is_stop and not token.is_punct])
        return processed

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts named entities from text (locations, organizations, etc.)
        """
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the text using spaCy's pattern matcher
        and custom rules for Spanish text.
        """
        doc = self.nlp(text)
        
        # Simple sentiment analysis based on polarity words
        positive_words = ["excelente", "bueno", "fantástico", "increíble", "maravilloso"]
        negative_words = ["malo", "terrible", "horrible", "pésimo", "deficiente"]
        
        sentiment_score = 0
        for token in doc:
            if token.lemma_.lower() in positive_words:
                sentiment_score += 1
            elif token.lemma_.lower() in negative_words:
                sentiment_score -= 1
        
        return {
            "score": sentiment_score,
            "sentiment": "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        }

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into smaller chunks for processing while maintaining context
        """
        return self.text_splitter.split_text(text)

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extracts main keywords from text based on frequency and importance
        """
        doc = self.nlp(text)
        keywords = []
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                token.pos_ in ['NOUN', 'PROPN', 'ADJ']):
                keywords.append(token.lemma_)
                
        # Count frequencies and get top keywords
        from collections import Counter
        keyword_freq = Counter(keywords)
        return [word for word, _ in keyword_freq.most_common(top_n)]

    def find_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates semantic similarity between two texts
        """
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)
