import re
from typing import List

from app.core.config import settings
from app.workers.base import BaseWorker
from app.models.context import WorkerResult, EvaluationAction, ActionType
from app.utils.logger import logger


class ArgumentationWorker(BaseWorker):
    """Argumentation specialist for claim/premise/evidence detection"""
    
    name = "argumentation_specialist"
    
    def __init__(self, use_model: bool = False):
        """
        Initialize Argumentation Worker.
        
        Args:
            use_model: If True, load DeBERTa model.
                      If False, use rule-based fallback.
        """
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        self.labels = ["claim", "premise", "evidence"]
        
        if use_model:
            self._load_model()
    
    def _load_model(self):
        """Load DeBERTa model for argumentation classification"""
        try:
            import importlib

            transformers = importlib.import_module("transformers")
            DebertaV2ForSequenceClassification = getattr(
                transformers,
                "DebertaV2ForSequenceClassification",
            )
            DebertaV2Tokenizer = getattr(transformers, "DebertaV2Tokenizer")
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(settings.ARGUMENTATION_MODEL_NAME)
            self.model = DebertaV2ForSequenceClassification.from_pretrained(
                settings.ARGUMENTATION_MODEL_NAME,
                num_labels=3
            )
        except Exception as e:
            logger.warning("Could not load DeBERTa model: %s. Using rule-based fallback.", e)
            self.use_model = False
    
    async def evaluate(self, document: str) -> WorkerResult:
        """Analyze document for argumentative structure"""
        sentences = self._split_sentences(document)
        
        if len(sentences) < 3:
            return WorkerResult(
                score=50.0,
                findings=["Document too short for argumentation analysis"],
                flagged_items=[],
                proposed_actions=[],
            )
        
        argument_map = self._classify_sentences(sentences)
        
        # Count components
        claims = [a for a in argument_map if a["component_type"] == "claim"]
        premises = [a for a in argument_map if a["component_type"] == "premise"]
        evidence = [a for a in argument_map if a["component_type"] == "evidence"]
        
        findings: List[str] = []
        actions: List[EvaluationAction] = []
        
        # Analyze structure quality
        if len(claims) == 0:
            findings.append("No clear claims identified - thesis may be unclear")
            actions.append(EvaluationAction(
                type=ActionType.CRITICAL_REVISION,
                target="document",
                category="argumentation",
                reasoning="Essay lacks clear argumentative claims",
                confidence=0.9,
                suggestion="State your main argument clearly in the introduction",
            ))
        
        if len(premises) < 2:
            findings.append("Insufficient supporting premises")
            actions.append(EvaluationAction(
                type=ActionType.MODERATE_REVISION,
                target="body_paragraphs",
                category="argumentation",
                reasoning="Need more reasoning to support claims",
                confidence=0.8,
                suggestion="Add logical reasoning that supports your main claims",
            ))
        
        if len(evidence) == 0:
            findings.append("No concrete evidence found - add examples or citations")
            actions.append(EvaluationAction(
                type=ActionType.MODERATE_REVISION,
                target="body_paragraphs",
                category="argumentation",
                reasoning="Arguments lack concrete evidence",
                confidence=0.85,
                suggestion="Include statistics, quotes, or examples to support your points",
            ))
        
        # Calculate score based on argument structure
        score = self._evaluate_structure(claims, premises, evidence)
        
        if not findings:
            findings.append("Good argumentative structure with clear claims and evidence")
        
        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=argument_map,
            proposed_actions=actions,
            metadata={
                "claims_count": len(claims),
                "premises_count": len(premises),
                "evidence_count": len(evidence),
            },
        )
    
    def _classify_sentences(self, sentences: List[str]) -> List[dict]:
        """Classify each sentence as claim, premise, or evidence"""
        argument_map = []
        
        for i, sentence in enumerate(sentences):
            if self.use_model and self.model is not None:
                component_type, confidence = self._model_classify(sentence)
            else:
                component_type, confidence = self._rule_classify(sentence)
            
            argument_map.append({
                "sentence_id": i,
                "text": sentence,
                "component_type": component_type,
                "confidence": confidence,
            })
        
        return argument_map
    
    def _model_classify(self, sentence: str) -> tuple:
        """Classify using DeBERTa model"""
        if self.model is None or self.tokenizer is None:
            return self._rule_classify(sentence)

        try:
            import importlib

            torch = importlib.import_module("torch")
            inputs = self.tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = probs.argmax(-1).item()
            confidence = probs.max().item()
            return self.labels[prediction], float(confidence)
        except Exception:
            return self._rule_classify(sentence)
    
    def _rule_classify(self, sentence: str) -> tuple:
        """Rule-based classification fallback"""
        sentence_lower = sentence.lower()
        
        # Claim indicators
        claim_patterns = [
            r'\b(i (believe|argue|think|contend))\b',
            r'\b(this (essay|paper|article) (argues|claims|demonstrates))\b',
            r'\b(should|must|ought to)\b',
            r'\b(is (essential|crucial|important|necessary))\b',
            r'\b(the (best|worst|only|main))\b',
        ]
        
        # Evidence indicators
        evidence_patterns = [
            r'\b(according to|research shows|studies indicate)\b',
            r'\b(for example|for instance|such as)\b',
            r'\b(\d+%|\d+ percent)\b',
            r'\b(in \d{4}|as of \d{4})\b',
            r'\b(data|statistics|survey|experiment)\b',
            r'\(.*?\d{4}.*?\)',  # Citation pattern (Author, 2024)
        ]
        
        # Premise indicators
        premise_patterns = [
            r'\b(because|since|as|given that)\b',
            r'\b(therefore|thus|hence|consequently)\b',
            r'\b(this (means|implies|suggests))\b',
            r'\b(if .* then)\b',
        ]
        
        # Check patterns
        for pattern in claim_patterns:
            if re.search(pattern, sentence_lower):
                return "claim", 0.7
        
        for pattern in evidence_patterns:
            if re.search(pattern, sentence_lower):
                return "evidence", 0.8
        
        for pattern in premise_patterns:
            if re.search(pattern, sentence_lower):
                return "premise", 0.6
        
        # Default to premise (most common)
        return "premise", 0.4
    
    def _evaluate_structure(self, claims: List, premises: List, evidence: List) -> float:
        """Evaluate overall argument structure quality"""
        # Ideal ratios: 1-2 claims per essay, multiple premises, some evidence
        total = len(claims) + len(premises) + len(evidence)
        if total == 0:
            return 0.0
        
        score = 50.0  # Base score
        
        # Bonus for having claims
        if len(claims) >= 1:
            score += 15
        
        # Bonus for having multiple premises
        if len(premises) >= 3:
            score += 20
        elif len(premises) >= 1:
            score += 10
        
        # Bonus for having evidence
        if len(evidence) >= 2:
            score += 15
        elif len(evidence) >= 1:
            score += 10
        
        return min(100.0, round(score, 2))
