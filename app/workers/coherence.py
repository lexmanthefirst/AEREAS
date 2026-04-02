from typing import List
import numpy as np
import re

from app.workers.base import BaseWorker
from app.models.context import WorkerResult, EvaluationAction, ActionType
from app.utils.logger import logger


class CoherenceWorker(BaseWorker):
    """Coherence specialist using sentence embeddings for transition analysis"""
    
    name = "coherence_specialist"
    MIN_PARAGRAPH_WORDS = 25
    WEAK_TRANSITION_THRESHOLD = 0.22
    MODERATE_TRANSITION_THRESHOLD = 0.30
    MAX_ACTIONS = 8
    
    def __init__(self, use_model: bool = False):
        """
        Initialize Coherence Worker.
        
        Args:
            use_model: If True, load Sentence-BERT model.
                      If False, use rule-based fallback.
        """
        self.use_model = use_model
        self.model = None
        
        if use_model:
            self._load_model()
    
    def _load_model(self):
        """Load Sentence-BERT model for embeddings"""
        try:
            import importlib

            sentence_transformers = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(sentence_transformers, "SentenceTransformer")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("Could not load embedding model: %s. Using rule-based fallback.", e)
            self.use_model = False
    
    async def evaluate(self, document: str) -> WorkerResult:
        """Analyze document coherence"""
        raw_paragraphs = self._split_paragraphs(document)
        paragraphs = self._normalize_paragraphs(raw_paragraphs)
        
        if len(paragraphs) < 2:
            return WorkerResult(
                score=100.0,
                findings=["Document too short for coherence analysis (< 2 paragraphs)"],
                flagged_items=[],
                proposed_actions=[],
                metadata={
                    "total_paragraphs": len(paragraphs),
                    "raw_paragraphs": len(raw_paragraphs),
                },
            )
        
        if self.use_model and self.model is not None:
            return await self._model_based_analysis(paragraphs)
        return self._rule_based_analysis(paragraphs)
    
    async def _model_based_analysis(self, paragraphs: List[str]) -> WorkerResult:
        """Analyze coherence using embeddings"""
        if self.model is None:
            return self._rule_based_analysis(paragraphs)

        # Get embeddings for each paragraph
        embeddings = self.model.encode(paragraphs, normalize_embeddings=True)
        
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []
        transitions: List[dict] = []
        
        # Analyze transitions between consecutive paragraphs
        weak_transition_count = 0
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarity = max(0.0, min(1.0, float(similarity)))
            
            transitions.append({
                "from_paragraph": i,
                "to_paragraph": i + 1,
                "coherence_score": similarity,
            })
            
            # Flag only clearly weak transitions as moderate, and cap noisy action volume.
            if similarity < self.WEAK_TRANSITION_THRESHOLD:
                weak_transition_count += 1
                findings.append(f"Weak transition between paragraphs {i+1} and {i+2}")
                flagged_items.append({
                    "from_paragraph": i,
                    "to_paragraph": i + 1,
                    "similarity": similarity,
                    "issue": "topic_drift",
                })
                if len(actions) < self.MAX_ACTIONS:
                    actions.append(EvaluationAction(
                        type=ActionType.MODERATE_REVISION,
                        target=f"paragraph_{i}_to_{i+1}",
                        category="coherence",
                        reasoning="Topic drift detected - paragraphs lack smooth transition",
                        confidence=round(0.7 + (self.WEAK_TRANSITION_THRESHOLD - similarity), 2),
                        suggestion="Add a bridging sentence that links the previous paragraph to the next one.",
                    ))
            elif similarity < self.MODERATE_TRANSITION_THRESHOLD and len(actions) < self.MAX_ACTIONS:
                actions.append(EvaluationAction(
                    type=ActionType.MINOR_IMPROVEMENT,
                    target=f"paragraph_{i}_to_{i+1}",
                    category="coherence",
                    reasoning="Transition could be clearer between adjacent paragraphs",
                    confidence=0.6,
                    suggestion="Consider adding a transition phrase to improve flow.",
                ))
        
        avg_coherence = float(np.mean([t["coherence_score"] for t in transitions]))
        weak_ratio = weak_transition_count / max(1, len(transitions))
        # Weighted score: semantic continuity + penalty for frequent weak jumps.
        score = round(max(0.0, min(100.0, (avg_coherence * 100) - (weak_ratio * 25))), 2)
        
        if not findings:
            findings.append("Good coherence throughout the document")
        
        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={
                "total_paragraphs": len(paragraphs),
                "average_coherence": avg_coherence,
                "weak_transition_ratio": round(weak_ratio, 3),
                "transitions": transitions,
            },
        )
    
    def _rule_based_analysis(self, paragraphs: List[str]) -> WorkerResult:
        """Fallback rule-based coherence analysis"""
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []
        
        # Transition words that indicate good coherence
        transition_words = {
            'furthermore', 'moreover', 'however', 'therefore', 'consequently',
            'additionally', 'similarly', 'in contrast', 'as a result',
            'on the other hand', 'in conclusion', 'first', 'second', 'finally',
            'thus', 'hence', 'nevertheless', 'nonetheless', 'meanwhile',
        }
        
        transitions_found = 0
        lexical_links_found = 0
        
        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            has_transition = any(tw in para_lower for tw in transition_words)
            if has_transition:
                transitions_found += 1
            if i > 0 and self._has_lexical_overlap(paragraphs[i - 1], para):
                lexical_links_found += 1
            elif i > 0:  # Skip first paragraph
                findings.append(f"Paragraph {i+1} lacks transition words")
                flagged_items.append({
                    "paragraph": i,
                    "issue": "missing_transition",
                })
                actions.append(EvaluationAction(
                    type=ActionType.MINOR_IMPROVEMENT,
                    target=f"paragraph_{i}",
                    category="coherence",
                    reasoning="Consider adding transition words for better flow",
                    confidence=0.6,
                    suggestion="Add words like 'Furthermore', 'However', 'Therefore'",
                ))
        
        # Calculate score based on transition word usage
        expected_transitions = max(0, len(paragraphs) - 1)
        if expected_transitions > 0:
            transition_ratio = transitions_found / expected_transitions
            lexical_ratio = lexical_links_found / expected_transitions
            score = round(((transition_ratio * 0.6) + (lexical_ratio * 0.4)) * 100, 2)
        else:
            score = 100.0
        
        if not findings:
            findings.append("Good use of transition words throughout")
        
        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={
                "total_paragraphs": len(paragraphs),
                "transitions_found": transitions_found,
                "lexical_links_found": lexical_links_found,
            },
        )

    def _normalize_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Merge tiny fragments to avoid over-segmentation from extracted documents."""
        normalized: List[str] = []

        for paragraph in paragraphs:
            text = paragraph.strip()
            if not text:
                continue

            word_count = len(text.split())
            if normalized and word_count < self.MIN_PARAGRAPH_WORDS:
                normalized[-1] = f"{normalized[-1]} {text}".strip()
            else:
                normalized.append(text)

        return normalized

    @staticmethod
    def _has_lexical_overlap(left: str, right: str) -> bool:
        """Simple lexical continuity check for rule-mode coherence scoring."""
        token_pattern = re.compile(r"[a-zA-Z]{4,}")
        left_tokens = set(token_pattern.findall(left.lower()))
        right_tokens = set(token_pattern.findall(right.lower()))
        if not left_tokens or not right_tokens:
            return False

        overlap = left_tokens.intersection(right_tokens)
        return len(overlap) >= 2
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
