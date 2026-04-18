"""Tone specialist — LLM-powered with rule-based fallback."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Tuple

from app.models.context import ActionType, EvaluationAction, WorkerResult
from app.utils.logger import logger
from app.workers.base import BaseWorker
from app.workers.schemas import ToneOutput

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class ToneWorker(BaseWorker):
    """Tone specialist for academic formality assessment."""

    name = "tone_specialist"

    SYSTEM_PROMPT = (
        "You are an expert in academic writing style and tone assessment.\n"
        "Analyse the document for formality issues that are inappropriate in "
        "academic/scholarly writing.\n\n"
        "For EACH issue found:\n"
        "- Quote the EXACT text from the document containing the problem.\n"
        "- Classify the issue type as one of: contraction, colloquialism, hedging, "
        "intensifier, vague_language, first_person, rhetorical_question.\n"
        "- Provide the formal academic alternative (the corrected text).\n"
        "- Briefly explain why it is inappropriate in academic writing.\n\n"
        "Common issues to check:\n"
        "- Contractions (don't, can't, won't, it's, they're, etc.)\n"
        "- Colloquialisms (gonna, wanna, stuff, a lot of, etc.)\n"
        "- Vague adjectives/nouns (things, good, bad, nice, big)\n"
        "- Intensifiers (really, very, pretty, extremely)\n"
        "- First-person opinion statements (I think, I believe, I feel)\n"
        "- Rhetorical questions in formal prose\n\n"
        "Rate overall_formality as: formal, mostly_formal, mixed, or informal.\n"
        "Score from 0-100 where 100 means perfect academic tone."
    )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    async def evaluate(self, document: str) -> WorkerResult:
        if self.llm_client and self.llm_client.available:
            try:
                return await self._llm_evaluate(document)
            except Exception as exc:
                logger.warning("Tone LLM failed: %s — falling back to rules.", exc)
        return self._rule_based_evaluate(document)

    # -- LLM path -------------------------------------------------------------

    async def _llm_evaluate(self, document: str) -> WorkerResult:
        output: ToneOutput = await self.llm_client.generate(  # type: ignore[union-attr]
            system_prompt=self.SYSTEM_PROMPT,
            content=document,
            response_schema=ToneOutput,
        )
        actions: List[EvaluationAction] = []
        findings: List[str] = []

        for issue in output.issues:
            span = self.find_span(document, issue.quoted_text) or self.find_span_fuzzy(document, issue.quoted_text)
            findings.append(f"{issue.issue_type}: {issue.explanation or issue.formal_alternative}")
            actions.append(
                EvaluationAction(
                    type=ActionType.MINOR_IMPROVEMENT,
                    target=issue.issue_type,
                    category="tone",
                    reasoning=issue.explanation or f"Informal language: {issue.issue_type}",
                    confidence=0.8,
                    suggestion=issue.formal_alternative,
                    highlight=span,
                    original_text=issue.quoted_text,
                    corrected_text=issue.formal_alternative,
                )
            )

        if not findings:
            findings.append("Appropriate academic tone throughout")

        return WorkerResult(
            score=output.score,
            findings=findings,
            flagged_items=[i.model_dump() for i in output.issues],
            proposed_actions=actions,
            metadata={
                "mode": "llm",
                "overall_formality": output.overall_formality,
                "issues_count": len(output.issues),
            },
        )

    # -- Rule-based fallback ---------------------------------------------------

    def _rule_based_evaluate(self, document: str) -> WorkerResult:
        sentences = self._split_sentences(document)
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []
        formality_scores: List[float] = []

        for i, sentence in enumerate(sentences):
            formality, issues = self._assess_formality(sentence)
            formality_scores.append(formality)

            if formality < 0.6:
                for issue_type, issue_text in issues:
                    span = self.find_span(document, sentence)
                    findings.append(f"Sentence {i + 1}: {issue_type}")
                    flagged_items.append(
                        {
                            "sentence_id": i,
                            "text": sentence,
                            "formality_score": formality,
                            "issue_type": issue_type,
                            "problematic_text": issue_text,
                        }
                    )
                    actions.append(
                        EvaluationAction(
                            type=ActionType.MINOR_IMPROVEMENT,
                            target=f"sentence_{i}",
                            category="tone",
                            reasoning=f"Informal language: {issue_type}",
                            confidence=0.75,
                            suggestion=f"Replace informal expression: '{issue_text}'",
                            highlight=span,
                            original_text=sentence,
                        )
                    )

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
                "mode": "rules",
                "total_sentences": len(sentences),
                "informal_sentences": len(flagged_items),
                "average_formality": avg_formality,
            },
        )

    def _assess_formality(self, sentence: str) -> Tuple[float, List[Tuple[str, str]]]:
        issues: List[Tuple[str, str]] = []
        sentence_lower = sentence.lower()

        informal_patterns = {
            r"\bdon't\b": ("Contraction", "don't -> do not"),
            r"\bwon't\b": ("Contraction", "won't -> will not"),
            r"\bcan't\b": ("Contraction", "can't -> cannot"),
            r"\baren't\b": ("Contraction", "aren't -> are not"),
            r"\bisn't\b": ("Contraction", "isn't -> is not"),
            r"\bdidn't\b": ("Contraction", "didn't -> did not"),
            r"\bwouldn't\b": ("Contraction", "wouldn't -> would not"),
            r"\bcouldn't\b": ("Contraction", "couldn't -> could not"),
            r"\bit's\b": ("Contraction", "it's -> it is"),
            r"\bthey're\b": ("Contraction", "they're -> they are"),
            r"\bwe're\b": ("Contraction", "we're -> we are"),
            r"\byou're\b": ("Contraction", "you're -> you are"),
            r"\bkinda\b": ("Colloquialism", "kinda -> somewhat/rather"),
            r"\bsorta\b": ("Colloquialism", "sorta -> somewhat"),
            r"\bgonna\b": ("Colloquialism", "gonna -> going to"),
            r"\bwanna\b": ("Colloquialism", "wanna -> want to"),
            r"\bgotta\b": ("Colloquialism", "gotta -> have to"),
            r"\blots of\b": ("Colloquialism", "lots of -> many/numerous"),
            r"\ba lot of\b": ("Colloquialism", "a lot of -> many/numerous"),
            r"\bstuff\b": ("Colloquialism", "stuff -> material/items"),
            r"\bthings\b": ("Vague language", "things -> specify what things"),
            r"\breally\b": ("Intensifier", "really -> significantly/considerably"),
            r"\bvery\b": ("Intensifier", "very -> highly/considerably"),
            r"\bi think\b": ("First person opinion", "I think -> research suggests/it is evident"),
            r"\bi believe\b": ("First person opinion", "I believe -> evidence indicates"),
            r"\bi feel\b": ("First person opinion", "I feel -> it appears"),
        }

        for pattern, (issue_type, suggestion) in informal_patterns.items():
            if re.search(pattern, sentence_lower):
                issues.append((issue_type, suggestion))

        if not issues:
            formality = 1.0
        elif len(issues) == 1:
            formality = 0.7
        elif len(issues) == 2:
            formality = 0.4
        else:
            formality = 0.2

        return formality, issues
