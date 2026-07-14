"""
Base worker abstract class for all specialist workers.

Each worker:
- READs from the EvaluationContext (document content)
- Performs specialized NLP analysis
- Returns WorkerResult to be posted to the board
"""

from abc import ABC, abstractmethod
from difflib import SequenceMatcher
import re
import time
from typing import List

from app.models.context import EvaluationContext, TextSpan, WorkerResult


class BaseWorker(ABC):
    """
    Specialist Worker base class.
    
    Subclasses must implement:
    - name: Worker identifier
    - evaluate(): Specialized evaluation logic
    """
    
    name: str = "base_worker"
    
    async def run(self, context: EvaluationContext) -> WorkerResult:
        """Execute worker analysis and return result"""
        start = time.perf_counter()
        result = await self.evaluate_with_context(context)
        
        # Set timing and worker name
        result.processing_time_ms = (time.perf_counter() - start) * 1000
        result.worker_name = self.name
        
        return result

    async def evaluate_with_context(self, context: EvaluationContext) -> WorkerResult:
        """Extended hook for workers that need document structure or prior findings."""
        return await self.evaluate(context.document_content)
    
    @abstractmethod
    async def evaluate(self, document: str) -> WorkerResult:
        """
        Implement specialized evaluation logic.
        
        Args:
            document: The document text to evaluate
            
        Returns:
            WorkerResult with findings, score, and proposed actions
        """
        pass
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be enhanced with spacy
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines (handles \n\s*\n and \r\n)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        if len(paragraphs) <= 1 and len(text) > 100:
            single_paragraphs = text.split('\n')
            if len(single_paragraphs) > 1:
                paragraphs = single_paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _calculate_score(self, total_items: int, error_count: int) -> float:
        """Calculate percentage score based on error rate"""
        if total_items == 0:
            return 100.0
        error_rate = error_count / total_items
        return round(max(0, (1 - error_rate) * 100), 2)

    @staticmethod
    def find_span(document: str, quoted_text: str | None) -> TextSpan | None:
        """Resolve exact quoted text to character offsets in the document."""
        if not quoted_text:
            return None
        idx = document.find(quoted_text)
        if idx == -1:
            return None
        return TextSpan(start=idx, end=idx + len(quoted_text), text=quoted_text)

    @staticmethod
    def find_span_fuzzy(
        document: str,
        quoted_text: str | None,
        threshold: float = 0.75,
    ) -> TextSpan | None:
        """Fuzzy match when LLM quotes text slightly differently."""
        if not quoted_text or len(quoted_text) < 10:
            return None

        # Try exact match first
        exact = BaseWorker.find_span(document, quoted_text)
        if exact is not None:
            return exact

        # Sliding window fuzzy search
        window = len(quoted_text)
        best_ratio = 0.0
        best_start = -1

        step = max(1, window // 4)
        for start in range(0, len(document) - window + 1, step):
            candidate = document[start : start + window]
            ratio = SequenceMatcher(None, quoted_text, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start

        if best_ratio >= threshold and best_start >= 0:
            matched = document[best_start : best_start + window]
            return TextSpan(start=best_start, end=best_start + window, text=matched)
        return None
