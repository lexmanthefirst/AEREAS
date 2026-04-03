from typing import List

from app.models.context import ActionType, EvaluationAction, EvaluationContext, WorkerResult
from app.services.research import WebResearchService
from app.workers.base import BaseWorker


class ResearchWorker(BaseWorker):
    """Evidence and source discovery specialist."""

    name = "research_specialist"

    async def evaluate(self, document: str) -> WorkerResult:
        return WorkerResult(
            score=100.0,
            findings=["Research worker requires evaluation context"],
            flagged_items=[],
            proposed_actions=[],
        )

    async def evaluate_with_context(self, context: EvaluationContext) -> WorkerResult:
        research_bundle = await WebResearchService.gather_supporting_sources(
            content=context.document_content,
            sections=context.document_sections,
        )
        results = research_bundle.get("results", [])
        total_sources = sum(len(item.get("sources", [])) for item in results)

        findings: List[str] = []
        actions: List[EvaluationAction] = []

        if not research_bundle.get("enabled"):
            findings.append("Web research is disabled; evaluation used local analysis only.")
        elif total_sources == 0:
            findings.append("No supporting web sources were retrieved for the main topics.")
            actions.append(
                EvaluationAction(
                    type=ActionType.MODERATE_REVISION,
                    target="document",
                    category="research",
                    reasoning="Key topics could not be supported with external sources during review.",
                    confidence=0.65,
                    suggestion="Verify claims against current literature and add stronger references.",
                )
            )
        else:
            findings.append(f"Retrieved {total_sources} external sources across {len(results)} research query areas.")

        for item in results:
            if item.get("errors") and not item.get("sources"):
                findings.append(f"Research lookup failed for '{item['query']}'")

        score = 100.0 if not research_bundle.get("enabled") else min(100.0, 55.0 + (total_sources * 7.5))
        return WorkerResult(
            score=round(score, 2),
            findings=findings or ["Research pass completed."],
            flagged_items=results,
            proposed_actions=actions,
            metadata={
                "web_research_enabled": research_bundle.get("enabled", False),
                "queries": research_bundle.get("queries", []),
                "sources_found": total_sources,
            },
        )
