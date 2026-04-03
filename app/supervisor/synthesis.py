from typing import List, Optional

from app.models.context import (
    ActionType,
    EvaluationAction,
    EvaluationContext,
    SynthesisResult,
)
from app.core.config import settings
from app.utils.logger import logger


class SynthesisEngine:
    """
    Synthesizes worker findings into a prioritized academic review summary.

    Uses LLM with rules-based fallback if:
    - LLM is not configured
    - LLM call fails
    """

    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        self.use_llm = use_llm
        self.api_key = api_key
        self.client = None
        self.model_name = settings.SYNTHESIS_MODEL_NAME

        if use_llm:
            self._init_llm()

    def _init_llm(self) -> None:
        """Initialize Gemini LLM client."""
        try:
            from google import genai

            api_key = self.api_key or settings.GEMINI_API_KEY
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                logger.warning("No GEMINI_API_KEY found. Using rules-based synthesis.")
                self.use_llm = False
        except Exception as exc:
            logger.warning("Could not initialize LLM: %s. Using rules-based synthesis.", exc)
            self.use_llm = False

    async def synthesize(self, context: EvaluationContext) -> SynthesisResult:
        if self.use_llm and self.client is not None:
            try:
                return await self._llm_synthesis(context)
            except Exception as exc:
                logger.warning("LLM synthesis failed: %s. Falling back to rules.", exc)
        return self._rules_synthesis(context)

    async def _llm_synthesis(self, context: EvaluationContext) -> SynthesisResult:
        client = self.client
        if client is None:
            raise RuntimeError("LLM client is not initialized")

        prompt = self._build_synthesis_prompt(context)
        response = client.models.generate_content(model=self.model_name, contents=prompt)
        response_text = (getattr(response, "text", "") or "").strip()
        if not response_text:
            raise ValueError("LLM returned empty synthesis response")
        return self._parse_llm_response(response_text, context)

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

    def _severity_rank(self, action_type: ActionType) -> int:
        ranks = {
            ActionType.CRITICAL_REVISION: 4,
            ActionType.MODERATE_REVISION: 3,
            ActionType.MINOR_IMPROVEMENT: 2,
            ActionType.POSITIVE_FEEDBACK: 1,
        }
        return ranks.get(action_type, 0)

    def _deduplicate_actions(self, actions: List[EvaluationAction]) -> List[EvaluationAction]:
        seen_targets = set()
        unique = []

        for action in actions:
            key = (action.target, action.category)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            unique.append(action)

        return unique

    def _generate_reasoning(self, context: EvaluationContext) -> str:
        lines = ["## Evaluation Summary\n"]

        if context.overall_score is not None:
            lines.append(f"**Overall Score: {context.overall_score:.1f}/100**\n")

        if context.document_sections:
            lines.append(
                f"Detected {len(context.document_sections)} sections/subsections in the submitted document.\n"
            )

        lines.append("### Analysis Breakdown:\n")
        for worker_name, result in context.worker_results.items():
            category = worker_name.replace("_specialist", "").title()
            score = result.score
            status = "Strong" if score >= 80 else "Mixed" if score >= 60 else "Weak"
            findings_preview = result.findings[0] if result.findings else "No issues"
            lines.append(f"- **{category}**: {status} {score:.1f}/100 - {findings_preview}")

        critical_count = sum(
            1
            for result in context.worker_results.values()
            for action in result.proposed_actions
            if action.type == ActionType.CRITICAL_REVISION
        )
        if critical_count > 0:
            lines.append(f"\n### Critical Concerns\n{critical_count} issues require immediate attention.")

        return "\n".join(lines)

    def _build_synthesis_prompt(self, context: EvaluationContext) -> str:
        prompt_parts = [
            "You are an expert academic writing reviewer. Analyze the evaluation results and produce a clear, ",
            "well-prioritized review for the student.\n\n",
            "## Worker Scores\n",
        ]

        for worker_name, result in context.worker_results.items():
            prompt_parts.append(f"- {worker_name}: {result.score}/100\n")
            prompt_parts.append(f"  Findings: {', '.join(result.findings[:3])}\n")

        if context.document_sections:
            prompt_parts.append("\n## Document Structure\n")
            for section in context.document_sections[:12]:
                prompt_parts.append(
                    f"- L{section.level}: {section.heading} ({len(section.paragraphs)} paragraphs)\n"
                )

        prompt_parts.extend(
            [
                "\n## Task\n",
                "1. Identify the 3 most important areas for improvement.\n",
                "2. Provide specific, actionable suggestions.\n",
                "3. Highlight strengths worth preserving.\n",
                "4. Give a concise overall assessment.\n",
                "5. Use section structure and research findings when available.\n",
            ]
        )
        return "".join(prompt_parts)

    def _parse_llm_response(self, response_text: str, context: EvaluationContext) -> SynthesisResult:
        rules_result = self._rules_synthesis(context)
        return SynthesisResult(reasoning=response_text, actions=rules_result.actions)
