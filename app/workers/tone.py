import re
from typing import List, Tuple

from app.core.config import settings
from app.workers.base import BaseWorker
from app.models.context import WorkerResult, EvaluationAction, ActionType
from app.utils.logger import logger


class ToneWorker(BaseWorker):
    """Tone specialist for academic formality assessment"""
    
    name = "tone_specialist"
    
    def __init__(self, use_model: bool = False):
        """
        Initialize Tone Worker.
        
        Args:
            use_model: If True, load BERT model.
                      If False, use rule-based fallback.
        """
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        
        if use_model:
            self._load_model()
    
    def _load_model(self):
        """Load BERT model for formality classification"""
        try:
            from transformers import BertForSequenceClassification, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(settings.TONE_MODEL_NAME)
            self.model = BertForSequenceClassification.from_pretrained(
                settings.TONE_MODEL_NAME,
                num_labels=2  # formal/informal
            )
        except Exception as e:
            logger.warning("Could not load BERT model: %s. Using rule-based fallback.", e)
            self.use_model = False
    
    async def evaluate(self, document: str) -> WorkerResult:
        """Analyze document for academic tone"""
        sentences = self._split_sentences(document)
        
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []
        formality_scores: List[float] = []
        
        for i, sentence in enumerate(sentences):
            formality, issues = self._assess_formality(sentence)
            formality_scores.append(formality)
            
            if formality < 0.6:  # Threshold for informal
                for issue_type, issue_text in issues:
                    findings.append(f"Sentence {i+1}: {issue_type}")
                    flagged_items.append({
                        "sentence_id": i,
                        "text": sentence,
                        "formality_score": formality,
                        "issue_type": issue_type,
                        "problematic_text": issue_text,
                    })
                    actions.append(EvaluationAction(
                        type=ActionType.MINOR_IMPROVEMENT,
                        target=f"sentence_{i}",
                        category="tone",
                        reasoning=f"Informal language: {issue_type}",
                        confidence=0.75,
                        suggestion=f"Replace informal expression: '{issue_text}'",
                    ))
        
        avg_formality = sum(formality_scores) / len(formality_scores) if formality_scores else 1.0
        score = round(avg_formality * 100, 2)
        
        if not findings:
            findings.append("Appropriate academic tone throughout")
        
        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={
                "total_sentences": len(sentences),
                "informal_sentences": len(flagged_items),
                "average_formality": avg_formality,
            },
        )
    
    def _assess_formality(self, sentence: str) -> Tuple[float, List[Tuple[str, str]]]:
        """Assess formality of a sentence"""
        issues: List[Tuple[str, str]] = []
        sentence_lower = sentence.lower()
        
        # Informal expressions and contractions
        informal_patterns = {
            # Contractions
            r"\bdon't\b": ("Contraction", "don't → do not"),
            r"\bwon't\b": ("Contraction", "won't → will not"),
            r"\bcan't\b": ("Contraction", "can't → cannot"),
            r"\baren't\b": ("Contraction", "aren't → are not"),
            r"\bisn't\b": ("Contraction", "isn't → is not"),
            r"\bdidn't\b": ("Contraction", "didn't → did not"),
            r"\bwouldn't\b": ("Contraction", "wouldn't → would not"),
            r"\bcouldn't\b": ("Contraction", "couldn't → could not"),
            r"\bit's\b": ("Contraction", "it's → it is"),
            r"\bthey're\b": ("Contraction", "they're → they are"),
            r"\bwe're\b": ("Contraction", "we're → we are"),
            r"\byou're\b": ("Contraction", "you're → you are"),
            r"\bthat's\b": ("Contraction", "that's → that is"),
            r"\bthere's\b": ("Contraction", "there's → there is"),
            r"\bhere's\b": ("Contraction", "here's → here is"),
            r"\blet's\b": ("Contraction", "let's → let us"),
            r"\bi've\b": ("Contraction", "I've → I have"),
            r"\bwe've\b": ("Contraction", "we've → we have"),
            r"\bthey've\b": ("Contraction", "they've → they have"),
            r"\bi'm\b": ("Contraction", "I'm → I am"),
            r"\bwasn't\b": ("Contraction", "wasn't → was not"),
            r"\bweren't\b": ("Contraction", "weren't → were not"),
            r"\bhasn't\b": ("Contraction", "hasn't → has not"),
            r"\bhaven't\b": ("Contraction", "haven't → have not"),
            
            # Colloquialisms
            r"\bkinda\b": ("Colloquialism", "kinda → somewhat/rather"),
            r"\bsorta\b": ("Colloquialism", "sorta → somewhat"),
            r"\bgonna\b": ("Colloquialism", "gonna → going to"),
            r"\bwanna\b": ("Colloquialism", "wanna → want to"),
            r"\bgotta\b": ("Colloquialism", "gotta → have to"),
            r"\blots of\b": ("Colloquialism", "lots of → many/numerous"),
            r"\ba lot of\b": ("Colloquialism", "a lot of → many/numerous"),
            r"\ba bunch of\b": ("Colloquialism", "a bunch of → many/several"),
            r"\bstuff\b": ("Colloquialism", "stuff → material/items"),
            r"\bthings\b": ("Vague language", "things → specify what things"),
            r"\bget\b": ("Informal verb", "get → obtain/receive/become"),
            r"\bgot\b": ("Informal verb", "got → obtained/received"),
            r"\bkid\b": ("Informal noun", "kid → child"),
            r"\bkids\b": ("Informal noun", "kids → children"),
            r"\bbig\b": ("Informal adjective", "big → significant/substantial"),
            r"\bnice\b": ("Vague adjective", "nice → pleasant/favorable"),
            r"\bgood\b": ("Vague adjective", "good → effective/beneficial"),
            r"\bbad\b": ("Vague adjective", "bad → detrimental/negative"),
            r"\breally\b": ("Intensifier", "really → significantly/considerably"),
            r"\bvery\b": ("Intensifier", "very → highly/considerably"),
            r"\bpretty\b": ("Intensifier", "pretty (as intensifier) → rather/fairly"),
            
            # First person (optional in academic writing)
            r"\bi think\b": ("First person opinion", "I think → research suggests/it is evident"),
            r"\bi believe\b": ("First person opinion", "I believe → evidence indicates"),
            r"\bi feel\b": ("First person opinion", "I feel → it appears"),
            
            # Rhetorical questions in formal writing
            r"\?": ("Rhetorical question", "Avoid questions; use declarative statements"),
        }
        
        for pattern, (issue_type, suggestion) in informal_patterns.items():
            if re.search(pattern, sentence_lower):
                match = re.search(pattern, sentence_lower)
                if match:
                    issues.append((issue_type, suggestion))
        
        # Calculate formality score
        if len(issues) == 0:
            formality = 1.0
        elif len(issues) == 1:
            formality = 0.7
        elif len(issues) == 2:
            formality = 0.4
        else:
            formality = 0.2
        
        return formality, issues
