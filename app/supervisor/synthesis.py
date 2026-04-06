"""SynthesisEngine — consolidates worker findings into a prioritised review."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from app.models.context import ActionType, EvaluationAction, EvaluationContext, SynthesisResult
from app.utils.logger import logger
from app.workers.base import BaseWorker
from app.workers.schemas import SynthesisOutput

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class SynthesisEngine:
    """Synthesises worker findings into a prioritised academic review summary.

    Uses LLM with structured output when available; falls back to
    rule-based sorting and deduplication otherwise.
    """

    SYSTEM_PROMPT = (
        "You are an expert academic writing reviewer synthesising evaluation findings.\n"
        "You will receive scores and findings from specialist workers "
        "(grammar, coherence, argumentation, tone, citation, plagiarism, research, review).\n\n"
        "Your job:\n"
        "1. Produce a clear, well-structured Markdown summary ('reasoning').\n"
        "2. Create a prioritised list of actions from most to least critical.\n"
        "3. Deduplicate overlapping suggestions from different workers.\n"
        "4. Highlight genuine strengths worth preserving.\n"
        "5. For each action, quote the relevant document text when possible.\n"
        "6. Assign severity: critical (blocks submission), moderate (should fix), "
        "minor (polish), or positive (strength to keep).\n\n"
        "Be specific and actionable — never give vague advice like 'improve coherence'."
    )

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.use_llm = use_llm
        self.llm_client = llm_client

    async def synthesize(self, context: EvaluationContext) -> SynthesisResult:
        if self.use_llm and self.llm_client and self.llm_client.available:
            try:
                return await self._llm_synthesis(context)
            except Exception as exc:
                logger.warning("LLM synthesis failed: %s. Falling back to rules.", exc)
        return self._rules_synthesis(context)

    # -- LLM path -------------------------------------------------------------

    async def _llm_synthesis(self, context: EvaluationContext) -> SynthesisResult:
        prompt = self._build_synthesis_prompt(context)
        output: SynthesisOutput = await self.llm_client.generate(  # type: ignore[union-attr]
            system_prompt=self.SYSTEM_PROMPT,
            content=prompt,
            response_schema=SynthesisOutput,
        )

        actions: List[EvaluationAction] = []
        severity_map = {
            "critical": ActionType.CRITICAL_REVISION,
            "moderate": ActionType.MODERATE_REVISION,
            "minor": ActionType.MINOR_IMPROVEMENT,
            "positive": ActionType.POSITIVE_FEEDBACK,
        }

        for sa in output.prioritized_actions:
            span = (
                BaseWorker.find_span(context.document_content, sa.quoted_text)
                or BaseWorker.find_span_fuzzy(context.document_content, sa.quoted_text)
                if sa.quoted_text
                else None
            )
            actions.append(
                EvaluationAction(
                    type=severity_map.get(sa.severity.lower(), ActionType.MINOR_IMPROVEMENT),
                    target=sa.target,
                    category=sa.category,
                    reasoning=sa.reasoning,
                    confidence=0.85,
                    suggestion=sa.suggestion,
                    highlight=span,
                    original_text=sa.quoted_text,
                )
            )

        return SynthesisResult(reasoning=output.reasoning, actions=actions)

    # -- Rule-based fallback ---------------------------------------------------

    def _rules_synthesis(self, context: EvaluationContext) -> SynthesisResult:
        all_actions: List[EvaluationAction] = []
        for result in context.worker_results.values():
            all_actions.extend(result.proposed_actions)

        all_actions.sort(
            key=lambda action: (self._severity_rank(action.type), action.confidence),
            reverse=True,
        )
        deduplicated = self._deduplicate_actions(all_actions)
        reasoning = self._generate_reasoning(context)

        return SynthesisResult(reasoning=reasoning, actions=deduplicated[:20])

    # -- Shared helpers --------------------------------------------------------

    def _build_synthesis_prompt(self, context: EvaluationContext) -> str:
        parts = ["## Worker Scores\n"]

        for worker_name, result in context.worker_results.items():
            parts.append(f"- {worker_name}: {result.score:.0f}/100\n")
            parts.append(f"  Findings: {', '.join(result.findings[:3])}\n")

        if context.document_sections:
            parts.append("\n## Document Structure\n")
            for section in context.document_sections[:12]:
                parts.append(f"- L{section.level}: {section.heading} ({len(section.paragraphs)} paragraphs)\n")

        citation_style = context.document_metadata.get("citation_style", "harvard")
        parts.append(f"\n## Context\n- Citation style: {citation_style}\n")

        parts.extend([
            "\n## Task\n",
            "1. Identify the 3 most important areas for improvement.\n",
            "2. Provide specific, actionable suggestions.\n",
            "3. Highlight strengths worth preserving.\n",
            "4. Give a concise overall assessment.\n",
        ])

        return "".join(parts)

    def _severity_rank(self, action_type: ActionType) -> int:
        return {
            ActionType.CRITICAL_REVISION: 4,
            ActionType.MODERATE_REVISION: 3,
            ActionType.MINOR_IMPROVEMENT: 2,
            ActionType.POSITIVE_FEEDBACK: 1,
        }.get(action_type, 0)

    def _deduplicate_actions(self, actions: List[EvaluationAction]) -> List[EvaluationAction]:
        seen = set()
        unique = []
        for action in actions:
            key = (action.target, action.category)
            if key in seen:
                continue
            seen.add(key)
            unique.append(action)
        return unique

    def _generate_reasoning(self, context: EvaluationContext) -> str:
        lines = ["## Evaluation Summary\n"]

        if context.overall_score is not None:
            lines.append(f"**Overall Score: {context.overall_score:.1f}/100**\n")

        if context.document_sections:
            lines.append(f"Detected {len(context.document_sections)} sections/subsections.\n")

        lines.append("### Analysis Breakdown:\n")
        for worker_name, result in context.worker_results.items():
            category = worker_name.replace("_specialist", "").title()
            score = result.score
            status = "Strong" if score >= 80 else "Mixed" if score >= 60 else "Weak"
            preview = result.findings[0] if result.findings else "No issues"
            lines.append(f"- **{category}**: {status} {score:.1f}/100 — {preview}")

        critical_count = sum(
            1
            for result in context.worker_results.values()
            for action in result.proposed_actions
            if action.type == ActionType.CRITICAL_REVISION
        )
        if critical_count > 0:
            lines.append(f"\n### Critical Concerns\n{critical_count} issues require immediate attention.")

        return "\n".join(lines)
