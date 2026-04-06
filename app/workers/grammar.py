"""Grammar specialist — LLM-powered with rule-based fallback."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

from app.models.context import ActionType, EvaluationAction, WorkerResult
from app.utils.logger import logger
from app.workers.base import BaseWorker
from app.workers.schemas import GrammarOutput

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class GrammarWorker(BaseWorker):
    """Grammar specialist for academic writing."""

    name = "grammar_specialist"

    SYSTEM_PROMPT = (
        "You are an expert English grammar checker specialised in academic writing.\n"
        "Analyse the document for grammar, mechanics, and syntax errors.\n\n"
        "For EACH issue found:\n"
        "- Quote the EXACT text from the document (verbatim, not paraphrased).\n"
        "- Classify the error type as one of: subject_verb, punctuation, tense, "
        "article, spelling, syntax, parallel_structure, fragment.\n"
        "- Provide the corrected version.\n"
        "- Give a brief one-sentence explanation of the rule violated.\n\n"
        "Focus on: subject-verb agreement, tense consistency, punctuation, "
        "article usage (a/an/the), pronoun reference, parallel structure, "
        "sentence fragments, and run-on sentences.\n\n"
        "Do NOT flag stylistic preferences or tone issues — only clear grammar errors.\n"
        "Assign a score from 0-100 where 100 means no errors found."
    )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    async def evaluate(self, document: str) -> WorkerResult:
        if self.llm_client and self.llm_client.available:
            try:
                return await self._llm_evaluate(document)
            except Exception as exc:
                logger.warning("Grammar LLM failed: %s — falling back to rules.", exc)
        return self._rule_based_evaluate(document)

    # -- LLM path -------------------------------------------------------------

    async def _llm_evaluate(self, document: str) -> WorkerResult:
        output: GrammarOutput = await self.llm_client.generate(  # type: ignore[union-attr]
            system_prompt=self.SYSTEM_PROMPT,
            content=document,
            response_schema=GrammarOutput,
        )
        actions: List[EvaluationAction] = []
        findings: List[str] = []

        for issue in output.issues:
            span = self.find_span(document, issue.quoted_text) or self.find_span_fuzzy(document, issue.quoted_text)
            actions.append(
                EvaluationAction(
                    type=ActionType.MINOR_IMPROVEMENT,
                    target=issue.error_type,
                    category="grammar",
                    reasoning=issue.explanation,
                    confidence=0.9,
                    suggestion=issue.correction,
                    highlight=span,
                    original_text=issue.quoted_text,
                    corrected_text=issue.correction,
                )
            )
            findings.append(f"{issue.error_type}: {issue.explanation}")

        return WorkerResult(
            score=output.score,
            findings=findings or ["No grammar issues detected"],
            flagged_items=[i.model_dump() for i in output.issues],
            proposed_actions=actions,
            metadata={"mode": "llm", "issues_count": len(output.issues)},
        )

    # -- Rule-based fallback ---------------------------------------------------

    def _rule_based_evaluate(self, document: str) -> WorkerResult:
        sentences = self._split_sentences(document)
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []

        for i, sentence in enumerate(sentences):
            corrected = self._rule_based_correction(sentence)
            if corrected and corrected.lower().strip() != sentence.lower().strip():
                error_type = self._classify_error(sentence, corrected)
                span = self.find_span(document, sentence)
                findings.append(f"Sentence {i + 1}: {error_type} detected")
                flagged_items.append(
                    {"sentence_id": i, "original": sentence, "corrected": corrected, "error_type": error_type}
                )
                actions.append(
                    EvaluationAction(
                        type=ActionType.MINOR_IMPROVEMENT,
                        target=f"sentence_{i}",
                        category="grammar",
                        reasoning=f"{error_type} — correction suggested",
                        confidence=0.85,
                        suggestion=corrected,
                        highlight=span,
                        original_text=sentence,
                        corrected_text=corrected,
                    )
                )

        score = self._calculate_score(len(sentences), len(flagged_items))
        return WorkerResult(
            score=score,
            findings=findings or ["No grammar issues detected"],
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={"mode": "rules", "total_sentences": len(sentences), "errors_found": len(flagged_items)},
        )

    def _rule_based_correction(self, sentence: str) -> Optional[str]:
        corrected = sentence
        corrections = [
            (r"\b(he|she|it) have\b", r"\1 has"),
            (r"\b(they|we|I) has\b", r"\1 have"),
            (r"\bdon't never\b", "don't ever"),
            (r"\bcan't hardly\b", "can hardly"),
            (r"\btheir is\b", "there is"),
            (r"\btheir are\b", "there are"),
            (r"\bits' ", "its "),
            (r"\bits ([a-z]+ing)\b", r"it's \1"),
            (r"\byour ([a-z]+ing)\b", r"you're \1"),
            (r"\ba ([aeiouAEIOU])", r"an \1"),
            (r"\ban ([^aeiouAEIOU\s])", r"a \1"),
            (r"^([a-z])", lambda m: m.group(1).upper()),
        ]
        for pattern, replacement in corrections:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        return corrected if corrected != sentence else None

    def _classify_error(self, original: str, corrected: str) -> str:
        ol, cl = original.lower(), corrected.lower()
        if ("have" in ol and "has" in cl) or ("has" in ol and "have" in cl):
            return "Subject-verb agreement"
        if "their" in ol and "there" in cl:
            return "Their/there confusion"
        if "your" in ol and "you're" in cl:
            return "Your/you're confusion"
        if "its" in ol and "it's" in cl:
            return "Its/it's confusion"
        if original[0].islower() and corrected[0].isupper():
            return "Capitalization"
        if (" a " in ol and " an " in cl) or (" an " in ol and " a " in cl):
            return "Article agreement (a/an)"
        return "Grammar issue"
