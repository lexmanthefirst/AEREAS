import io
from typing import List


class DocumentExtractor:
    """Extract text content from uploaded documents"""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.docx', '.pdf', '.doc', '.md'}
    
    @staticmethod
    def extract_text(file_data: io.BytesIO, filename: str) -> str:
        """
        Extract text from a document.
        
        Args:
            file_data: BytesIO object containing file data
            filename: Original filename (for extension detection)
            
        Returns:
            Extracted text content
        """
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if extension == 'txt' or extension == 'md':
            return DocumentExtractor._extract_txt(file_data)
        elif extension == 'docx':
            return DocumentExtractor._extract_docx(file_data)
        elif extension == 'pdf':
            return DocumentExtractor._extract_pdf(file_data)
        elif extension == 'doc':
            raise ValueError("Legacy .doc format not supported. Please use .docx")
        else:
            # Try as plain text
            return DocumentExtractor._extract_txt(file_data)
    
    @staticmethod
    def _extract_txt(file_data: io.BytesIO) -> str:
        """Extract text from plain text file."""
        file_data.seek(0)
        content = file_data.read()
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Could not decode text file")
    
    @staticmethod
    def _extract_docx(file_data: io.BytesIO) -> str:
        """Extract text from Word document."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        file_data.seek(0)
        doc = Document(file_data)

        blocks: List[str] = []

        # Standard document paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                blocks.append(text)

        # Text in tables is common in assignment templates/forms.
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        text = para.text.strip()
                        if text:
                            blocks.append(text)

        # Include header/footer content when present.
        for section in doc.sections:
            for para in section.header.paragraphs:
                text = para.text.strip()
                if text:
                    blocks.append(text)
            for para in section.footer.paragraphs:
                text = para.text.strip()
                if text:
                    blocks.append(text)

        content = '\n\n'.join(blocks).strip()
        if content:
            return content

        raise ValueError(
            "No readable text found in DOCX. Ensure the file contains selectable text "
            "(not only images or embedded objects)."
        )
    
    @staticmethod
    def _extract_pdf(file_data: io.BytesIO) -> str:
        """Extract text from PDF document."""
        try:
            import pypdf
        except ImportError:
            try:
                import PyPDF2 as pypdf
            except ImportError:
                raise ImportError("pypdf not installed. Run: pip install pypdf")
        
        file_data.seek(0)
        reader = pypdf.PdfReader(file_data)
        
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        content = '\n\n'.join(text_parts).strip()
        if content:
            return content

        raise ValueError(
            "No readable text found in PDF. The file may be image-only (scanned). "
            "Use OCR or upload a text-selectable PDF."
        )
    
    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if file type is supported."""
        extension = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
        return extension in DocumentExtractor.SUPPORTED_EXTENSIONS
