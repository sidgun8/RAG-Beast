import re
import nltk
import spacy
from typing import List, Dict, Any, Optional
import textstat
from collections import Counter
import string
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


class TextProcessor:
    """Text processing utilities for document analysis and preparation"""
    
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # Try to load spaCy model, fallback to basic processing if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        except Exception as e:
            print(f"⚠️  Error loading spaCy model: {e}")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ''
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        # Remove extra whitespace
        text = text.strip()
        
        return text
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text"""
        if self.nlp:
            return self._extract_key_phrases_spacy(text, max_phrases)
        else:
            return self._extract_key_phrases_nltk(text, max_phrases)
    
    def _extract_key_phrases_spacy(self, text: str, max_phrases: int) -> List[str]:
        """Extract key phrases using spaCy"""
        doc = self.nlp(text)
        phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to 3-word phrases
                phrases.append(chunk.text.lower())
        
        # Count phrase frequency
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(max_phrases)]
    
    def _extract_key_phrases_nltk(self, text: str, max_phrases: int) -> List[str]:
        """Extract key phrases using NLTK"""
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Get bigrams and trigrams
        bigrams = list(nltk.bigrams(words))
        trigrams = list(nltk.trigrams(words))
        
        phrases = []
        phrases.extend([f"{w1} {w2}" for w1, w2 in bigrams])
        phrases.extend([f"{w1} {w2} {w3}" for w1, w2, w3 in trigrams])
        
        # Count phrase frequency
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(max_phrases)]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        if self.nlp:
            return self._extract_entities_spacy(text)
        else:
            return self._extract_entities_nltk(text)
    
    def _extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)
        
        people = []
        places = []
        organizations = []
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON']:
                people.append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                places.append(ent.text)
            elif ent.label_ in ['ORG']:
                organizations.append(ent.text)
        
        return {
            'people': list(set(people)),
            'places': list(set(places)),
            'organizations': list(set(organizations))
        }
    
    def _extract_entities_nltk(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using NLTK"""
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        # Simple entity extraction based on POS tags
        people = []
        places = []
        organizations = []
        
        for word, pos in pos_tags:
            if pos == 'NNP':  # Proper noun
                # Simple heuristic: capitalize first letter words are likely entities
                if word[0].isupper():
                    # This is a very basic approach - spaCy is much better
                    if len(word) > 2:
                        people.append(word)
        
        return {
            'people': list(set(people)),
            'places': list(set(places)),
            'organizations': list(set(organizations))
        }
    
    def tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text"""
        words = nltk.word_tokenize(text.lower())
        return [self.stemmer.stem(word) for word in words if word.isalpha()]
    
    def extract_summary(self, text: str, max_sentences: int = 3) -> str:
        """Extract summary (first few sentences)"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return '. '.join(sentences[:max_sentences]).strip()
    
    def chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
        """Chunk text for processing"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                last_sentence_end = text.rfind('.', start, end)
                last_question_end = text.rfind('?', start, end)
                last_exclamation_end = text.rfind('!', start, end)
                
                last_end = max(last_sentence_end, last_question_end, last_exclamation_end)
                if last_end > start + max_chunk_size * 0.5:
                    end = last_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    'text': chunk,
                    'start': start,
                    'end': min(end, len(text))
                })
            
            start = end - overlap
        
        return chunks
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get comprehensive text statistics"""
        return {
            'word_count': len(text.split()),
            'character_count': len(text),
            'sentence_count': textstat.sentence_count(text),
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text)
        }
    
    def process_document(self, raw_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process document for ingestion"""
        if metadata is None:
            metadata = {}
        
        cleaned_text = self.clean_text(raw_text)
        summary = self.extract_summary(cleaned_text)
        key_phrases = self.extract_key_phrases(cleaned_text)
        entities = self.extract_entities(cleaned_text)
        chunks = self.chunk_text(cleaned_text)
        text_stats = self.get_text_statistics(cleaned_text)
        
        return {
            'original_text': raw_text,
            'cleaned_text': cleaned_text,
            'summary': summary,
            'key_phrases': key_phrases,
            'entities': entities,
            'chunks': chunks,
            'text_statistics': text_stats,
            'metadata': {
                **metadata,
                'word_count': text_stats['word_count'],
                'character_count': text_stats['character_count'],
                'processed_at': datetime.now().isoformat()
            }
        }
