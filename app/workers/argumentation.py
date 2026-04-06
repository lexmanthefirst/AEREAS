"""Argumentation specialist — LLM-powered with rule-based fallback."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

from app.models.context import ActionType, EvaluationAction, WorkerResult
from app.utils.logger import logger
from app.workers.base import BaseWorker
from app.workers.schemas import ArgumentationOutput

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class ArgumentationWorker(BaseWorker):
    """Argumentation specialist for claim/premise/evidence analysis."""

    name = "argumentation_specialist"

    SYSTEM_PROMPT = (
        "You are an expert in argumentative essay analysis for academic writing.\n"
        "Analyse the document to identify:\n"
        "1. The thesis statement (the central claim of the essay), if present.\n"
        "2. Individual argument components — classify each as:\n"
        "   - claim: an assertion the author wants the reader to accept\n"
        "   - premise: reasoning that supports a claim\n"
        "   - evidence: data, citations, examples, or statistics that back a premise\n"
        "   - counterargument: an opposing viewpoint acknowledged by the author\n"
        "3. Logical gaps — assertions made without supporting evidence or reasoning.\n\n"
        "For each component, quote the EXACT text from the document.\n"
        "Rate each component's strength as: strong, adequate, or weak.\n"
        "Explain your rating briefly.\n\n"
        "Score from 0-100 where 100 means exemplary argumentation with clear thesis, "
        "well-supported claims, strong evidence, and addressed counterarguments."
    )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    async def evaluate(self, document: str) -> WorkerResult:
        sentences = self._split_sentences(document)
        if len(sentences) < 3:
            return WorkerResult(
                score=50.0,
                findings=["Document too short for argumentation analysis"],
                flagged_items=[],
                proposed_actions=[],
            )

        if self.llm_client and self.llm_client.available:
            try:
                return await self._llm_evaluate(document)
            except Exception as exc:
                logger.warning("Argumentation LLM failed: %s — falling back to rules.", exc)
        return self._rule_based_evaluate(document, sentences)

    # -- LLM path -------------------------------------------------------------

    async def _llm_evaluate(self, document: str) -> WorkerResult:
        output: ArgumentationOutput = await self.llm_client.generate(  # type: ignore[union-attr]
            system_prompt=self.SYSTEM_PROMPT,
            content=document,
            response_schema=ArgumentationOutput,
        )
        actions: List[EvaluationAction] = []
        findings: List[str] = []

        if output.thesis_statement:
            findings.append(f"Thesis identified: {output.thesis_statement[:100]}")
        else:
            findings.append("No clear thesis statement identified")
            actions.append(
                EvaluationAction(
                    type=ActionType.CRITICAL_REVISION,
                    target="thesis",
                    category="argumentation",
                    reasoning="Essay lacks a clear thesis statement",
                    confidence=0.9,
                    suggestion="State your main argument clearly in the introduction",
                )
            )

        claims = [c for c in output.components if c.component_type == "claim"]
        evidence = [c for c in output.components if c.component_type == "evidence"]
        weak = [c for c in output.components if c.strength == "weak"]

        if not claims:
            findings.append("No clear claims identified")
            actions.append(
                EvaluationAction(
                    type=ActionType.CRITICAL_REVISION,
                    target="document",
                    category="argumentation",
                    reasoning="Essay lacks clear argumentative claims",
                    confidence=0.9,
                    suggestion="State your main argument clearly in the introduction",
                )
            )

        if not evidence:
            findings.append("No concrete evidence found — add examples or citations")
            actions.append(
                EvaluationAction(
                    type=ActionType.MODERATE_REVISION,
                    target="body_paragraphs",
                    category="argumentation",
                    reasoning="Arguments lack concrete evidence",
                    confidence=0.85,
                    suggestion="Include statistics, quotes, or examples to support your points",
                )
            )

        for comp in weak[:5]:
            span = self.find_span(document, comp.quoted_text) or self.find_span_fuzzy(document, comp.quoted_text)
            actions.append(
                EvaluationAction(
                    type=ActionType.MODERATE_REVISION,
                    target=comp.component_type,
                    category="argumentation",
                    reasoning=comp.reasoning or f"Weak {comp.component_type}",
                    confidence=0.8,
                    suggestion=f"Strengthen this {comp.component_type} with more support",
                    highlight=span,
                    original_text=comp.quoted_text,
                )
            )

        for gap in output.logical_gaps[:3]:
            findings.append(f"Logical gap: {gap}")
            actions.append(
                EvaluationAction(
                    type=ActionType.MODERATE_REVISION,
                    target="argumentation",
                    category="argumentation",
                    reasoning=gap,
                    confidence=0.75,
                    suggestion="Address this logical gap with evidence or reasoning",
                )
            )

        if not findings:
            findings.append("Good argumentative structure with clear claims and evidence")

        return WorkerResult(
            score=output.score,
            findings=findings,
            flagged_items=[c.model_dump() for c in output.components],
            proposed_actions=actions,
            metadata={
                "mode": "llm",
                "thesis": output.thesis_statement,
                "claims_count": len(claims),
                "evidence_count": len(evidence),
                "logical_gaps": len(output.logical_gaps),
            },
        )

    # -- Rule-based fallback ---------------------------------------------------

    def _rule_based_evaluate(self, document: str, sentences: List[str]) -> WorkerResult:
        argument_map = self._classify_sentences(sentences)

        claims = [a for a in argument_map if a["component_type"] == "claim"]
        premises = [a for a in argument_map if a["component_type"] == "premise"]
        evidence = [a for a in argument_map if a["component_type"] == "evidence"]

        findings: List[str] = []
        actions: List[EvaluationAction] = []

        if not claims:
            findings.append("No clear claims identified — thesis may be unclear")
            actions.append(
                EvaluationAction(
                    type=ActionType.CRITICAL_REVISION,
                    target="document",
                    category="argumentation",
                    reasoning="Essay lacks clear argumentative claims",
                    confidence=0.9,
                    suggestion="State your main argument clearly in the introduction",
                )
            )

        if len(premises) < 2:
            findings.append("Insufficient supporting premises")
            actions.append(
                EvaluationAction(
                    type=ActionType.MODERATE_REVISION,
                    target="body_paragraphs",
                    category="argumentation",
                    reasoning="Need more reasoning to support claims",
                    confidence=0.8,
                    suggestion="Add logical reasoning that supports your main claims",
                )
            )

        if not evidence:
            findings.append("No concrete evidence found — add examples or citations")
            actions.append(
                EvaluationAction(
                    type=ActionType.MODERATE_REVISION,
                    target="body_paragraphs",
                    category="argumentation",
                    reasoning="Arguments lack concrete evidence",
                    confidence=0.85,
                    suggestion="Include statistics, quotes, or examples to support your points",
                )
            )

        score = self._evaluate_structure(claims, premises, evidence)
        if not findings:
            findings.append("Good argumentative structure with clear claims and evidence")

        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=argument_map,
            proposed_actions=actions,
            metadata={
                "mode": "rules",
                "claims_count": len(claims),
                "premises_count": len(premises),
                "evidence_count": len(evidence),
            },
        )

    def _classify_sentences(self, sentences: List[str]) -> List[dict]:
        argument_map = []
        for i, sentence in enumerate(sentences):
            component_type, confidence = self._rule_classify(sentence)
            argument_map.append(
                {"sentence_id": i, "text": sentence, "component_type": component_type, "confidence": confidence}
            )
        return argument_map

    def _rule_classify(self, sentence: str) -> tuple:
        sentence_lower = sentence.lower()

        claim_patterns = [
            r"\b(i (believe|argue|think|contend))\b",
            r"\b(this (essay|paper|article) (argues|claims|demonstrates))\b",
            r"\b(should|must|ought to)\b",
            r"\b(is (essential|crucial|important|necessary))\b",
            r"\b(the (best|worst|only|main))\b",
        ]
        evidence_patterns = [
            r"\b(according to|research shows|studies indicate)\b",
            r"\b(for example|for instance|such as)\b",
            r"\b(\d+%|\d+ percent)\b",
            r"\b(in \d{4}|as of \d{4})\b",
            r"\b(data|statistics|survey|experiment)\b",
            r"\(.*?\d{4}.*?\)",
        ]
        premise_patterns = [
            r"\b(because|since|as|given that)\b",
            r"\b(therefore|thus|hence|consequently)\b",
            r"\b(this (means|implies|suggests))\b",
            r"\b(if .* then)\b",
        ]

        for pattern in claim_patterns:
            if re.search(pattern, sentence_lower):
                return "claim", 0.7
        for pattern in evidence_patterns:
            if re.search(pattern, sentence_lower):
                return "evidence", 0.8
        for pattern in premise_patterns:
            if re.search(pattern, sentence_lower):
                return "premise", 0.6
        return "premise", 0.4

    def _evaluate_structure(self, claims: List, premises: List, evidence: List) -> float:
        total = len(claims) + len(premises) + len(evidence)
        if total == 0:
            return 0.0
        score = 50.0
        if len(claims) >= 1:
            score += 15
        if len(premises) >= 3:
            score += 20
        elif len(premises) >= 1:
            score += 10
        if len(evidence) >= 2:
            score += 15
        elif len(evidence) >= 1:
            score += 10
        return min(100.0, round(score, 2))
