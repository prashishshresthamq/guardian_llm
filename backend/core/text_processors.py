import re
import PyPDF2
import docx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles text extraction and preprocessing"""
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_sections(text: str) -> dict:
        """Extract common paper sections"""
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': '',
            'references': ''
        }
        
        # Simple section extraction based on headers
        section_patterns = {
            'abstract': r'(?i)abstract[:\s]+(.*?)(?=\n\s*(?:introduction|1\.|\w+:))',
            'introduction': r'(?i)(?:introduction|1\.)[:\s]+(.*?)(?=\n\s*(?:methodology|method|2\.|\w+:))',
            'methodology': r'(?i)(?:methodology|method|2\.)[:\s]+(.*?)(?=\n\s*(?:results|3\.|\w+:))',
            'results': r'(?i)(?:results|3\.)[:\s]+(.*?)(?=\n\s*(?:conclusion|discussion|4\.|\w+:))',
            'conclusion': r'(?i)(?:conclusion|discussion)[:\s]+(.*?)(?=\n\s*(?:references|\w+:))',
            'references': r'(?i)references[:\s]+(.*?)$'
        }
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        return sections