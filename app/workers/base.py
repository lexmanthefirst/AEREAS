"""
Base worker abstract class for all specialist workers.

Each worker:
- READs from the EvaluationContext (document content)
- Performs specialized NLP analysis
- Returns WorkerResult to be posted to the board
"""

from abc import ABC, abstractmethod
import time
import re
from typing import List

from app.models.context import EvaluationContext, WorkerResult


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
        
        # Read document from the blackboard
        document = context.document_content
        
        # Perform specialized evaluation
        result = await self.evaluate(document)
        
        # Set timing and worker name
        result.processing_time_ms = (time.perf_counter() - start) * 1000
        result.worker_name = self.name
        
        return result
    
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
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _calculate_score(self, total_items: int, error_count: int) -> float:
        """Calculate percentage score based on error rate"""
        if total_items == 0:
            return 100.0
        error_rate = error_count / total_items
        return round(max(0, (1 - error_rate) * 100), 2)
