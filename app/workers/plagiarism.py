import hashlib
import re
from typing import List, Set

from app.core.config import settings
from app.workers.base import BaseWorker
from app.models.context import WorkerResult, EvaluationAction, ActionType
from app.utils.logger import logger


class PlagiarismWorker(BaseWorker):
    """Plagiarism specialist using semantic similarity detection"""
    
    name = "plagiarism_specialist"
    
    def __init__(self, use_model: bool = False, corpus_path: str = None):
        """
        Initialize Plagiarism Worker.
        
        Args:
            use_model: If True, load Sentence-BERT + FAISS.
            corpus_path: Path to reference corpus for comparison.
        """
        self.use_model = use_model
        self.corpus_path = corpus_path
        self.model = None
        self.index = None
        
        if use_model:
            self._load_model()
    
    def _load_model(self):
        """Load Sentence-BERT model and FAISS index"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            
            # Load FAISS index if corpus available
            if self.corpus_path:
                import faiss
                self.index = faiss.read_index(self.corpus_path)
        except Exception as e:
            logger.warning("Could not load model/index: %s. Using heuristic mode.", e)
            self.use_model = False
    
    async def evaluate(self, document: str) -> WorkerResult:
        """Analyze document for potential plagiarism"""
        sentences = self._split_sentences(document)
        
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []
        
        if self.use_model and self.model is not None and self.index is not None:
            # Full plagiarism check with vector similarity
            results = await self._check_with_model(sentences)
            flagged_items = results
            for item in results:
                if item["similarity"] > 0.9:
                    findings.append(f"Sentence {item['sentence_id']+1}: High similarity ({item['similarity']:.0%})")
                    actions.append(EvaluationAction(
                        type=ActionType.CRITICAL_REVISION,
                        target=f"sentence_{item['sentence_id']}",
                        category="plagiarism",
                        reasoning=f"High similarity to existing source ({item['similarity']:.0%})",
                        confidence=item["similarity"],
                        suggestion="Rewrite in your own words or add proper citation",
                    ))
        else:
            # Heuristic checks when no corpus available
            results = self._heuristic_checks(document, sentences)
            flagged_items = results
            for item in results:
                findings.append(item["message"])
                actions.append(EvaluationAction(
                    type=item["severity"],
                    target=item.get("target", "document"),
                    category="plagiarism",
                    reasoning=item["message"],
                    confidence=item.get("confidence", 0.6),
                    suggestion=item.get("suggestion"),
                ))
        
        # Calculate originality score
        score = self._calculate_originality_score(
            total_sentences=len(sentences),
            flagged_count=len([f for f in flagged_items if f.get("similarity", 0) > 0.8])
        )
        
        if not findings:
            findings.append("No plagiarism indicators detected (heuristic check)")
        
        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={
                "check_mode": "model" if (self.use_model and self.model) else "heuristic",
                "total_sentences": len(sentences),
                "corpus_available": self.index is not None,
            },
        )
    
    async def _check_with_model(self, sentences: List[str]) -> List[dict]:
        """Check sentences against corpus using embeddings"""
        import numpy as np

        results = []
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 5:
                continue  # Skip very short sentences
            
            embedding = self.model.encode([sentence])
            distances, indices = self.index.search(embedding, k=1)
            
            # Convert distance to similarity (assuming L2 distance)
            similarity = 1 / (1 + distances[0][0])
            
            if similarity > 0.7:
                results.append({
                    "sentence_id": i,
                    "text": sentence,
                    "similarity": float(similarity),
                    "matched_index": int(indices[0][0]),
                })
        
        return results
    
    def _heuristic_checks(self, document: str, sentences: List[str]) -> List[dict]:
        """Perform heuristic plagiarism checks without corpus"""
        issues = []
        
        # Check 1: Repetitive phrasing (internal plagiarism)
        repeated = self._find_repeated_phrases(sentences)
        if repeated:
            issues.append({
                "type": "repeated_phrases",
                "message": f"Found {len(repeated)} repeated phrases (possible self-plagiarism)",
                "severity": ActionType.MINOR_IMPROVEMENT,
                "phrases": repeated[:5],
                "suggestion": "Vary your wording to avoid repetition",
                "confidence": 0.5,
            })
        
        # Check 2: Unnaturally perfect prose (copy-paste indicator)
        perfect_signs = self._check_perfect_prose(document)
        for sign in perfect_signs:
            issues.append({
                "type": "perfect_prose",
                "message": sign["message"],
                "severity": ActionType.MODERATE_REVISION,
                "confidence": 0.6,
                "suggestion": "Verify this is original content and not copied",
            })
        
        # Check 3: Inconsistent style (multiple authors indicator)
        style_issues = self._check_style_consistency(sentences)
        if style_issues:
            issues.append({
                "type": "style_inconsistency",
                "message": "Writing style varies significantly across sections",
                "severity": ActionType.MODERATE_REVISION,
                "details": style_issues,
                "confidence": 0.5,
                "suggestion": "Review sections for consistent voice and style",
            })
        
        # Check 4: Missing citations for factual claims
        uncited_facts = self._find_uncited_facts(sentences)
        for fact in uncited_facts[:3]:
            issues.append({
                "type": "uncited_fact",
                "message": f"Factual claim without citation: '{fact[:50]}...'",
                "severity": ActionType.MODERATE_REVISION,
                "target": fact,
                "confidence": 0.7,
                "suggestion": "Add citation for factual claims",
            })
        
        return issues
    
    def _find_repeated_phrases(self, sentences: List[str]) -> List[str]:
        """Find phrases repeated multiple times"""
        # Extract 4-grams
        all_ngrams: dict = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for i in range(len(words) - 3):
                ngram = ' '.join(words[i:i+4])
                all_ngrams[ngram] = all_ngrams.get(ngram, 0) + 1
        
        # Return phrases that appear 3+ times
        repeated = [phrase for phrase, count in all_ngrams.items() if count >= 3]
        return repeated
    
    def _check_perfect_prose(self, document: str) -> List[dict]:
        """Check for signs of unnaturally perfect prose"""
        issues = []
        
        # Check for Wikipedia-style formatting
        if re.search(r'\[\d+\]', document):
            issues.append({
                "message": "Wikipedia-style reference markers found [1]",
            })
        
        # Check for HTML remnants
        if re.search(r'<[^>]+>', document):
            issues.append({
                "message": "HTML tags detected - possible web copy",
            })
        
        # Check for URL remnants
        if re.search(r'https?://|www\.', document):
            issues.append({
                "message": "URLs found in document - possible direct copy",
            })
        
        return issues
    
    def _check_style_consistency(self, sentences: List[str]) -> List[dict]:
        """Check for style consistency across document"""
        # Simple check: variance in sentence length
        lengths = [len(s.split()) for s in sentences]
        if not lengths:
            return []
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        if variance > 100:  # High variance
            return [{"high_length_variance": variance}]
        return []
    
    def _find_uncited_facts(self, sentences: List[str]) -> List[str]:
        """Find factual claims that should have citations"""
        fact_patterns = [
            r'\bstudies show\b',
            r'\bresearch indicates\b',
            r'\baccording to\b',
            r'\b\d+%\b',
            r'\bstatistics\b',
            r'\bin \d{4}\b',
            r'\bexperts ?(say|believe|agree)\b',
        ]
        
        uncited = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if has fact pattern but no citation
            has_fact = any(re.search(p, sentence_lower) for p in fact_patterns)
            has_citation = bool(re.search(r'\([A-Z][a-z]+.*?\d{4}\)', sentence))
            
            if has_fact and not has_citation:
                uncited.append(sentence)
        
        return uncited
    
    def _calculate_originality_score(self, total_sentences: int, flagged_count: int) -> float:
        """Calculate originality score"""
        if total_sentences == 0:
            return 100.0
        
        originality = (total_sentences - flagged_count) / total_sentences
        return round(originality * 100, 2)
