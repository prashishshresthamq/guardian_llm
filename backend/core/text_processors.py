"""
Guardian LLM - Text Processing Utilities
Text preprocessing and cleaning functions
"""

import re
import string
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


class TextProcessor:
    """Text processing and cleaning utilities"""
    
    def __init__(self):
        """Initialize text processor"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
        
        # Common contractions mapping
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "let's": "let us",
            "it's": "it is",
            "ain't": "is not",
            "y'all": "you all"
        }
        
        # Emoji pattern
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text for analysis
        
        Args:
            text: Raw text input
            
        Returns:
            Dictionary containing processed text components
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Get sentences
        sentences = self.get_sentences(text)
        
        # Remove stopwords for analysis
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Get lemmatized tokens
        lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)
        
        # Extract n-grams
        bigrams = self.get_ngrams(tokens, 2)
        trigrams = self.get_ngrams(tokens, 3)
        
        # Detect language (simplified - assumes English)
        language = self.detect_language(text)
        
        return {
            'original': text,
            'cleaned': cleaned_text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'sentences': sentences,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'language': language,
            'has_urls': self.contains_urls(text),
            'has_emails': self.contains_emails(text),
            'has_mentions': self.contains_mentions(text),
            'has_hashtags': self.contains_hashtags(text)
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'@(\w+)', r'\1', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub(' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\']', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text.lower())
            # Remove pure punctuation tokens
            tokens = [token for token in tokens if token not in string.punctuation]
            return tokens
        except:
            # Fallback to simple split
            return text.lower().split()
    
    def get_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if not self.stop_words:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if not self.lemmatizer:
            return tokens
        
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except:
            return tokens
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from tokens
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            List of n-grams
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text (simplified - always returns 'en')
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code
        """
        # For now, we assume English
        # In production, you might use langdetect or similar
        return 'en'
    
    def contains_urls(self, text: str) -> bool:
        """Check if text contains URLs"""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return bool(url_pattern.search(text))
    
    def contains_emails(self, text: str) -> bool:
        """Check if text contains email addresses"""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return bool(email_pattern.search(text))
    
    def contains_mentions(self, text: str) -> bool:
        """Check if text contains @mentions"""
        return bool(re.search(r'@\w+', text))
    
    def contains_hashtags(self, text: str) -> bool:
        """Check if text contains hashtags"""
        return bool(re.search(r'#\w+', text))
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            'mentions': re.findall(r'@(\w+)', text),
            'hashtags': re.findall(r'#(\w+)', text),
            'phone_numbers': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        }
        
        return entities
    
    def get_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Calculate text complexity metrics
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        sentences = self.get_sentences(text)
        words = self.tokenize(text)
        
        if not sentences or not words:
            return {
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'lexical_diversity': 0,
                'complexity_score': 0
            }
        
        # Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Average word length
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        # Lexical diversity (unique words / total words)
        lexical_diversity = len(set(words)) / len(words) if words else 0
        
        # Simple complexity score
        complexity_score = (avg_sentence_length * 0.5 + avg_word_length * 0.3 + lexical_diversity * 0.2)
        
        return {
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'lexical_diversity': round(lexical_diversity, 2),
            'complexity_score': round(complexity_score, 2)
        }


# Utility functions
def preprocess_for_model(text: str, max_length: int = 512) -> str:
    """
    Preprocess text for model input
    
    Args:
        text: Raw text
        max_length: Maximum length of text
        
    Returns:
        Preprocessed text
    """
    processor = TextProcessor()
    cleaned = processor.clean_text(text)
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned


def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text
    
    Args:
        text: Text to analyze
        num_keywords: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    processor = TextProcessor()
    processed = processor.process(text)
    
    # Get filtered tokens
    tokens = processed['filtered_tokens']
    
    # Count token frequency
    token_freq = {}
    for token in tokens:
        token_freq[token] = token_freq.get(token, 0) + 1
    
    # Sort by frequency
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [token for token, freq in sorted_tokens[:num_keywords]]