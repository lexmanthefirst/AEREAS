"""Holistic review specialist — LLM-powered with rule-based fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from app.core.config import settings
from app.models.context import ActionType, EvaluationAction, EvaluationContext, WorkerResult
from app.utils.logger import logger
from app.workers.base import BaseWorker
from app.workers.schemas import ReviewOutput

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class ReviewWorker(BaseWorker):
    """Holistic reviewer that consolidates structure, findings, and research."""

    name = "review_specialist"

    SYSTEM_PROMPT = (
        "You are a rigorous academic reviewer performing a holistic assessment of a student's writing.\n"
        "You will receive the document text along with its section structure and findings from "
        "specialist workers (grammar, coherence, argumentation, tone, citation, plagiarism, research).\n\n"
        "Your job:\n"
        "1. Identify the most critical issues across ALL dimensions.\n"
        "2. Note genuine strengths worth preserving.\n"
        "3. Provide specific, actionable suggestions — not vague advice.\n"
        "4. For each finding, quote the relevant text from the document when possible.\n"
        "5. Rate severity as: critical (must fix), moderate (should fix), or minor (nice to fix).\n\n"
        "Score from 0-100 reflecting the overall quality of the academic writing."
    )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    async def evaluate(self, document: str) -> WorkerResult:
        return WorkerResult(
            score=100.0,
            findings=["Review worker requires evaluation context"],
            flagged_items=[],
            proposed_actions=[],
        )

    async def evaluate_with_context(self, context: EvaluationContext) -> WorkerResult:
        if self.llm_client and self.llm_client.available:
            try:
                return await self._llm_review(context)
            except Exception as exc:
                logger.warning("Review LLM failed: %s — falling back to rules.", exc)
        return self._rule_review(context)

    # -- LLM path -------------------------------------------------------------

    async def _llm_review(self, context: EvaluationContext) -> WorkerResult:
        content = self._build_review_content(context)
        output: ReviewOutput = await self.llm_client.generate(  # type: ignore[union-attr]
            system_prompt=self.SYSTEM_PROMPT,
            content=content,
            response_schema=ReviewOutput,
            model_name=settings.REVIEW_MODEL_NAME,
        )

        actions: List[EvaluationAction] = []
        findings: List[str] = []

        for f in output.findings[:8]:
            span = (
                self.find_span(context.document_content, f.quoted_text)
                or self.find_span_fuzzy(context.document_content, f.quoted_text)
                if f.quoted_text
                else None
            )
            severity_map = {
                "critical": ActionType.CRITICAL_REVISION,
                "moderate": ActionType.MODERATE_REVISION,
                "minor": ActionType.MINOR_IMPROVEMENT,
            }
            findings.append(f.finding)
            actions.append(
                EvaluationAction(
                    type=severity_map.get(f.severity.lower(), ActionType.MINOR_IMPROVEMENT),
                    target=f.target,
                    category="review",
                    reasoning=f.finding,
                    confidence=0.8,
                    suggestion=f.suggestion,
                    highlight=span,
                    original_text=f.quoted_text,
                )
            )

        if not findings:
            findings.append("Holistic review completed — no dominant weakness found.")

        return WorkerResult(
            score=output.score,
            findings=findings,
            flagged_items=[f.model_dump() for f in output.findings],
            proposed_actions=actions[:6],
            metadata={"mode": "llm", "strengths": output.strengths},
        )

    def _build_review_content(self, context: EvaluationContext) -> str:
        parts: List[str] = []

        # Section structure
        section_summary = "\n".join(
            f"- L{s.level}: {s.heading} ({len(s.paragraphs)} paragraphs)"
            for s in context.document_sections[:12]
        )
        parts.append(f"## Document Structure\n{section_summary or '- No sections detected'}\n")

        # Worker findings summary
        worker_lines = []
        for name, result in context.worker_results.items():
            if name == self.name:
                continue
            top_findings = ", ".join(result.findings[:2]) if result.findings else "No issues"
            worker_lines.append(f"- {name}: {result.score:.0f}/100 | {top_findings}")
        parts.append(f"## Specialist Worker Findings\n" + "\n".join(worker_lines) + "\n")

        # Document text (capped)
        parts.append(f"## Document Text\n{context.document_content[:12000]}")

        return "\n\n".join(parts)

    # -- Rule-based fallback ---------------------------------------------------

    def _rule_review(self, context: EvaluationContext) -> WorkerResult:
        findings: List[str] = []
        actions: List[EvaluationAction] = []

        sections = context.document_sections
        body_sections = [s for s in sections if s.paragraphs]
        if len(body_sections) < 3:
            findings.append("Document structure is shallow; stronger section development is needed.")
            actions.append(
                EvaluationAction(
                    type=ActionType.MODERATE_REVISION,
                    target="document_structure",
                    category="review",
                    reasoning="The work is not clearly developed into major sections and subsections.",
                    confidence=0.78,
                    suggestion="Reorganise the paper into clear academic sections with focused paragraphs.",
                )
            )

        low_scores = {
            name: result.score
            for name, result in context.worker_results.items()
            if result.score < 70 and name != self.name
        }
        for worker_name, score in sorted(low_scores.items(), key=lambda item: item[1])[:3]:
            category = worker_name.replace("_specialist", "").replace("_", " ")
            findings.append(f"{category.title()} needs substantial improvement ({score:.1f}/100).")
            actions.append(
                EvaluationAction(
                    type=ActionType.CRITICAL_REVISION if score < 55 else ActionType.MODERATE_REVISION,
                    target=category.replace(" ", "_"),
                    category="review",
                    reasoning=f"The overall review indicates weak performance in {category}.",
                    confidence=0.8,
                    suggestion=f"Revise the {category} dimension before final submission.",
                )
            )

        score_values = []
        research_enabled = settings.ENABLE_WEB_RESEARCH
        
        # Check if citations are present in the document
        has_citations = True
        if "citation_specialist" in context.worker_results:
            cit_res = context.worker_results["citation_specialist"]
            citations_count = cit_res.metadata.get("citations_found", 0) if cit_res.metadata else 0
            doc_content = context.document_content or ""
            doc_lower = doc_content.lower()
            reference_headers = ['references', 'bibliography', 'works cited', 'reference list', 'sources', 'citations']
            has_ref_section = any(
                any(line.strip() == h for line in doc_lower.splitlines())
                for h in reference_headers
            )
            if citations_count == 0 and not has_ref_section:
                has_citations = False

        for name, result in context.worker_results.items():
            if name == self.name:
                continue
            if name == "research_specialist" and not research_enabled:
                continue
            if name == "citation_specialist" and not has_citations:
                continue
            if name == "plagiarism_specialist" and result.score >= 90.0:
                continue
            score_values.append(result.score)
            
        average_score = sum(score_values) / len(score_values) if score_values else 100.0

        if not findings:
            findings.append("The document is broadly well reviewed with no dominant weakness.")

        return WorkerResult(
            score=round(average_score, 2),
            findings=findings,
            flagged_items=[{"low_scores": low_scores, "section_count": len(sections)}],
            proposed_actions=actions[:6],
            metadata={"mode": "rules", "section_count": len(sections), "body_sections": len(body_sections)},
        )
