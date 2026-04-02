import re
from typing import List, Optional

from app.workers.base import BaseWorker
from app.models.context import WorkerResult, EvaluationAction, ActionType
from app.utils.logger import logger


class GrammarWorker(BaseWorker):
    """Grammar specialist using T5 for grammar correction"""
    
    name = "grammar_specialist"
    
    def __init__(self, use_model: bool = False):
        """
        Initialize Grammar Worker.
        
        Args:
            use_model: If True, load T5 model (requires GPU). 
                      If False, use rule-based fallback.
        """
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        
        if use_model:
            self._load_model()
    
    def _load_model(self):
        """Load T5 grammar correction model"""
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
            self.model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
        except Exception as e:
            logger.warning("Could not load T5 model: %s. Using rule-based fallback.", e)
            self.use_model = False
    
    async def evaluate(self, document: str) -> WorkerResult:
        """Analyze document for grammar errors"""
        sentences = self._split_sentences(document)
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []
        
        for i, sentence in enumerate(sentences):
            # Get correction
            corrected = self._correct_grammar(sentence)
            
            if corrected and corrected.lower().strip() != sentence.lower().strip():
                error_type = self._classify_error(sentence, corrected)
                
                findings.append(f"Sentence {i+1}: {error_type} detected")
                flagged_items.append({
                    "sentence_id": i,
                    "original": sentence,
                    "corrected": corrected,
                    "error_type": error_type,
                })
                actions.append(EvaluationAction(
                    type=ActionType.MINOR_IMPROVEMENT,
                    target=f"sentence_{i}",
                    category="grammar",
                    reasoning=f"{error_type} - correction suggested",
                    confidence=0.85,
                    suggestion=corrected,
                ))
        
        score = self._calculate_score(len(sentences), len(flagged_items))
        
        return WorkerResult(
            score=score,
            findings=findings if findings else ["No grammar issues detected"],
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={"total_sentences": len(sentences), "errors_found": len(flagged_items)},
        )
    
    def _correct_grammar(self, sentence: str) -> Optional[str]:
        """Correct grammar in a sentence"""
        if self.use_model and self.model is not None:
            return self._model_correction(sentence)
        return self._rule_based_correction(sentence)
    
    def _model_correction(self, sentence: str) -> Optional[str]:
        """Use T5 model for correction"""
        try:
            input_text = f"grammar: {sentence}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(**inputs, max_length=512, num_beams=4)
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected
        except Exception:
            return self._rule_based_correction(sentence)
    
    def _rule_based_correction(self, sentence: str) -> Optional[str]:
        """Fallback rule-based grammar checking"""
        corrected = sentence
        
        # Common grammar rules
        corrections = [
            # Subject-verb agreement
            (r'\b(he|she|it) have\b', r'\1 has'),
            (r'\b(they|we|I) has\b', r'\1 have'),
            # Double negatives
            (r"\bdon't never\b", "don't ever"),
            (r"\bcan't hardly\b", "can hardly"),
            # Their/there/they're
            (r'\btheir is\b', 'there is'),
            (r'\btheir are\b', 'there are'),
            # Its/it's
            (r"\bits' ", "its "),
            (r'\bits ([a-z]+ing)\b', r"it's \1"),
            # Your/you're
            (r'\byour ([a-z]+ing)\b', r"you're \1"),
            # A/an agreement
            (r'\ba ([aeiouAEIOU])', r'an \1'),
            (r'\ban ([^aeiouAEIOU\s])', r'a \1'),
            # Missing capitalization at sentence start
            (r'^([a-z])', lambda m: m.group(1).upper()),
        ]
        
        for pattern, replacement in corrections:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        return corrected if corrected != sentence else None
    
    def _classify_error(self, original: str, corrected: str) -> str:
        """Classify the type of grammar error"""
        orig_lower = original.lower()
        corr_lower = corrected.lower()
        
        if 'have' in orig_lower and 'has' in corr_lower or 'has' in orig_lower and 'have' in corr_lower:
            return "Subject-verb agreement"
        if 'their' in orig_lower and 'there' in corr_lower:
            return "Their/there confusion"
        if 'your' in orig_lower and "you're" in corr_lower:
            return "Your/you're confusion"
        if 'its' in orig_lower and "it's" in corr_lower:
            return "Its/it's confusion"
        if original[0].islower() and corrected[0].isupper():
            return "Capitalization"
        if ' a ' in orig_lower and ' an ' in corr_lower or ' an ' in orig_lower and ' a ' in corr_lower:
            return "Article agreement (a/an)"
        
        return "Grammar issue"
