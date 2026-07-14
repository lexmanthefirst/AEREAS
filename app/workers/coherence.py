"""Coherence specialist — LLM-powered with rule-based fallback."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

from app.models.context import ActionType, EvaluationAction, WorkerResult
from app.utils.logger import logger
from app.workers.base import BaseWorker
from app.workers.schemas import CoherenceOutput

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class CoherenceWorker(BaseWorker):
    """Coherence specialist for paragraph transitions and topic flow."""

    name = "coherence_specialist"
    MIN_PARAGRAPH_WORDS = 25
    MAX_ACTIONS = 8

    SYSTEM_PROMPT = (
        "You are an expert academic writing coherence analyst.\n"
        "Analyse the document for:\n"
        "1. Paragraph-to-paragraph transitions — are they smooth or abrupt?\n"
        "2. Topic flow — does each paragraph logically follow the previous?\n"
        "3. Logical progression — does the argument build naturally toward a conclusion?\n\n"
        "For each weak or moderate transition, quote:\n"
        "- The EXACT last sentence of the preceding paragraph.\n"
        "- The EXACT first sentence of the following paragraph.\n"
        "Provide a specific bridging suggestion for each.\n\n"
        "Assign severity as 'weak' (topic jumps abruptly) or 'moderate' (transition could be smoother).\n"
        "Provide a topic_flow_assessment summarising the overall flow.\n"
        "Score from 0-100 where 100 means perfect coherence."
    )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    async def evaluate(self, document: str) -> WorkerResult:
        raw_paragraphs = self._split_paragraphs(document)
        paragraphs = self._normalize_paragraphs(raw_paragraphs)

        if len(paragraphs) < 2:
            raw_sentences = self._split_sentences(document)
            paragraphs = []
            for i in range(0, len(raw_sentences), 3):
                group = " ".join(raw_sentences[i:i+3]).strip()
                if group:
                    paragraphs.append(group)

        if len(paragraphs) < 2:
            return WorkerResult(
                score=80.0,
                findings=["Document too short for coherence analysis (< 2 paragraphs)"],
                flagged_items=[],
                proposed_actions=[],
                metadata={"total_paragraphs": len(paragraphs), "raw_paragraphs": len(raw_paragraphs)},
            )

        if self.llm_client and self.llm_client.available:
            try:
                return await self._llm_evaluate(document)
            except Exception as exc:
                logger.warning("Coherence LLM failed: %s — falling back to rules.", exc)
        return self._rule_based_analysis(paragraphs, document)

    # -- LLM path -------------------------------------------------------------

    async def _llm_evaluate(self, document: str) -> WorkerResult:
        output: CoherenceOutput = await self.llm_client.generate(  # type: ignore[union-attr]
            system_prompt=self.SYSTEM_PROMPT,
            content=document,
            response_schema=CoherenceOutput,
        )
        actions: List[EvaluationAction] = []
        findings: List[str] = []

        for i, t in enumerate(output.transitions):
            span = self.find_span(document, t.to_paragraph_text) or self.find_span_fuzzy(document, t.to_paragraph_text)
            action_type = ActionType.MODERATE_REVISION if t.severity == "weak" else ActionType.MINOR_IMPROVEMENT
            findings.append(f"Transition issue: {t.issue}")
            if len(actions) < self.MAX_ACTIONS:
                actions.append(
                    EvaluationAction(
                        type=action_type,
                        target=f"transition_{i}",
                        category="coherence",
                        reasoning=t.issue,
                        confidence=0.85 if t.severity == "weak" else 0.7,
                        suggestion=t.suggestion,
                        highlight=span,
                        original_text=t.to_paragraph_text,
                    )
                )

        if not findings:
            findings.append("Good coherence throughout the document")

        return WorkerResult(
            score=output.score,
            findings=findings,
            flagged_items=[t.model_dump() for t in output.transitions],
            proposed_actions=actions,
            metadata={
                "mode": "llm",
                "topic_flow": output.topic_flow_assessment,
                "weak_transitions": sum(1 for t in output.transitions if t.severity == "weak"),
            },
        )

    # -- Rule-based fallback ---------------------------------------------------

    def _rule_based_analysis(self, paragraphs: List[str], document: str) -> WorkerResult:
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []

        transition_words = {
            "furthermore", "moreover", "however", "therefore", "consequently",
            "additionally", "similarly", "in contrast", "as a result",
            "on the other hand", "in conclusion", "first", "second", "finally",
            "thus", "hence", "nevertheless", "nonetheless", "meanwhile",
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
            elif i > 0:
                span = self.find_span(document, para[:80])
                findings.append(f"Paragraph {i + 1} lacks transition words")
                flagged_items.append({"paragraph": i, "issue": "missing_transition"})
                actions.append(
                    EvaluationAction(
                        type=ActionType.MINOR_IMPROVEMENT,
                        target=f"paragraph_{i}",
                        category="coherence",
                        reasoning="Consider adding transition words for better flow",
                        confidence=0.6,
                        suggestion="Add words like 'Furthermore', 'However', 'Therefore'",
                        highlight=span,
                    )
                )

        expected_transitions = max(0, len(paragraphs) - 1)
        if expected_transitions > 0:
            transition_ratio = transitions_found / expected_transitions
            lexical_ratio = lexical_links_found / expected_transitions
            score = round(((transition_ratio * 0.6) + (lexical_ratio * 0.4)) * 100, 2)
        else:
            score = 80.0

        if not findings:
            findings.append("Good use of transition words throughout")

        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={
                "mode": "rules",
                "total_paragraphs": len(paragraphs),
                "transitions_found": transitions_found,
                "lexical_links_found": lexical_links_found,
            },
        )

    def _normalize_paragraphs(self, paragraphs: List[str]) -> List[str]:
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
        token_pattern = re.compile(r"[a-zA-Z]{4,}")
        left_tokens = set(token_pattern.findall(left.lower()))
        right_tokens = set(token_pattern.findall(right.lower()))
        if not left_tokens or not right_tokens:
            return False
        overlap = left_tokens.intersection(right_tokens)
        return len(overlap) >= 2
