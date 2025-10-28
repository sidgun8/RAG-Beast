import os
import mimetypes
from typing import Dict, Any, Optional
import PyPDF2
import pandas as pd
import json
from docx import Document
import logging

logger = logging.getLogger(__name__)


class FileProcessor:
    """File processing utilities for extracting text from various file types"""
    
    def __init__(self):
        self.supported_types = {
            '.txt': self._extract_text,
            '.pdf': self._extract_pdf,
            '.docx': self._extract_docx,
            '.csv': self._extract_csv,
            '.json': self._extract_json
        }
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file type is supported"""
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_types
    
    def get_file_type(self, file_path: str) -> str:
        """Get file type from file path"""
        _, ext = os.path.splitext(file_path.lower())
        return ext
    
    async def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from file based on file type"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = self.get_file_type(file_path)
        
        if not self.is_supported(file_path):
            raise ValueError(f"Unsupported file type: {file_type}")
        
        try:
            extractor = self.supported_types[file_type]
            return await extractor(file_path)
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    async def _extract_text(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    async def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise
    
    async def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX: {e}")
            raise
    
    async def _extract_csv(self, file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Convert CSV to readable text
            text_parts = []
            for _, row in df.iterrows():
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                text_parts.append(row_text)
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting CSV: {e}")
            raise
    
    async def _extract_json(self, file_path: str) -> str:
        """Extract text from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                return json.dumps(json_data, indent=2)
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = os.stat(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return {
            'filename': os.path.basename(file_path),
            'file_size': stat.st_size,
            'file_type': self.get_file_type(file_path),
            'mime_type': mime_type,
            'is_supported': self.is_supported(file_path),
            'created_at': stat.st_ctime,
            'modified_at': stat.st_mtime
        }
