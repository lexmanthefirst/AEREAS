from typing import List

from app.core.config import settings
from app.models.context import ActionType, EvaluationAction, EvaluationContext, WorkerResult
from app.utils.logger import logger
from app.workers.base import BaseWorker


class ReviewWorker(BaseWorker):
    """Holistic reviewer that consolidates structure, findings, and research."""

    name = "review_specialist"

    async def evaluate(self, document: str) -> WorkerResult:
        return WorkerResult(
            score=100.0,
            findings=["Review worker requires evaluation context"],
            flagged_items=[],
            proposed_actions=[],
        )

    async def evaluate_with_context(self, context: EvaluationContext) -> WorkerResult:
        if settings.GEMINI_API_KEY:
            llm_result = self._llm_review(context)
            if llm_result is not None:
                return llm_result
        return self._rule_review(context)

    def _rule_review(self, context: EvaluationContext) -> WorkerResult:
        findings: List[str] = []
        actions: List[EvaluationAction] = []

        sections = context.document_sections
        body_sections = [section for section in sections if section.paragraphs]
        if len(body_sections) < 3:
            findings.append("Document structure is shallow; stronger section development is needed.")
            actions.append(
                EvaluationAction(
                    type=ActionType.MODERATE_REVISION,
                    target="document_structure",
                    category="review",
                    reasoning="The work is not clearly developed into major sections and subsections.",
                    confidence=0.78,
                    suggestion="Reorganize the paper into clear academic sections with focused paragraphs.",
                )
            )

        low_scores = {
            name: result.score
            for name, result in context.worker_results.items()
            if result.score < 70 and name not in {self.name}
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

        research_meta = context.worker_results.get("research_specialist")
        if research_meta and research_meta.metadata.get("web_research_enabled"):
            findings.append(
                f"Research review used {research_meta.metadata.get('sources_found', 0)} supporting source candidates."
            )

        score_values = [
            result.score for name, result in context.worker_results.items() if name != self.name
        ]
        average_score = sum(score_values) / len(score_values) if score_values else 100.0

        if not findings:
            findings.append("The document is broadly well reviewed with no dominant weakness.")

        return WorkerResult(
            score=round(average_score, 2),
            findings=findings,
            flagged_items=[{"low_scores": low_scores, "section_count": len(sections)}],
            proposed_actions=actions[:6],
            metadata={
                "review_mode": "rules",
                "section_count": len(sections),
                "body_sections": len(body_sections),
            },
        )

    def _llm_review(self, context: EvaluationContext) -> WorkerResult | None:
        try:
            from google import genai

            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            section_summary = "\n".join(
                f"- L{section.level}: {section.heading} ({len(section.paragraphs)} paragraphs)"
                for section in context.document_sections[:12]
            )
            worker_summary = "\n".join(
                f"- {name}: {result.score}/100 | {', '.join(result.findings[:2])}"
                for name, result in context.worker_results.items()
            )
            prompt = (
                "You are a rigorous academic reviewer. Assess the work using the supplied section structure and "
                "worker findings. Return concise review findings under lines starting with 'FINDING:' and action "
                "items under lines starting with 'ACTION:'. Each action should be 'ACTION: severity | target | suggestion'.\n\n"
                f"Sections:\n{section_summary or '- None'}\n\n"
                f"Worker findings:\n{worker_summary or '- None'}\n\n"
                f"Document:\n{context.document_content[:10000]}"
            )
            response = client.models.generate_content(
                model=settings.REVIEW_MODEL_NAME,
                contents=prompt,
            )
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return None

            findings = []
            actions: List[EvaluationAction] = []
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("FINDING:"):
                    findings.append(stripped.removeprefix("FINDING:").strip())
                elif stripped.startswith("ACTION:"):
                    payload = [part.strip() for part in stripped.removeprefix("ACTION:").split("|")]
                    if len(payload) >= 3:
                        severity, target, suggestion = payload[:3]
                        action_type = {
                            "critical": ActionType.CRITICAL_REVISION,
                            "moderate": ActionType.MODERATE_REVISION,
                        }.get(severity.lower(), ActionType.MINOR_IMPROVEMENT)
                        actions.append(
                            EvaluationAction(
                                type=action_type,
                                target=target,
                                category="review",
                                reasoning="Holistic academic review recommendation.",
                                confidence=0.78,
                                suggestion=suggestion,
                            )
                        )

            if not findings:
                findings.append("Holistic review completed.")

            score_values = [result.score for result in context.worker_results.values()]
            score = sum(score_values) / len(score_values) if score_values else 100.0
            return WorkerResult(
                score=round(score, 2),
                findings=findings[:6],
                flagged_items=[{"raw_review": text}],
                proposed_actions=actions[:6],
                metadata={"review_mode": "llm"},
            )
        except Exception as exc:
            logger.warning("LLM holistic review failed: %s", exc)
            return None
