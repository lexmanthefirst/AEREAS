"""SupervisorAgent — orchestrates the evaluation workflow."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict
from uuid import UUID

from app.core.config import settings
from app.models.context import EvaluationAction, EvaluationContext, LiveTriggerType, WorkerResult
from app.services.document import DocumentExtractor
from app.supervisor.synthesis import SynthesisEngine
from app.workers.grammar import GrammarWorker
from app.workers.coherence import CoherenceWorker
from app.workers.argumentation import ArgumentationWorker
from app.workers.tone import ToneWorker
from app.workers.citation import CitationWorker
from app.workers.plagiarism import PlagiarismWorker
from app.workers.critic import CriticWorker
from app.workers.research import ResearchWorker
from app.workers.reviewer import ReviewWorker
from app.utils.logger import logger

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class SupervisorAgent:
    """Lead Reviewer — orchestrates specialist workers, synthesis, and QA.

    Uses the shared LLMClient for all LLM-powered workers. When no LLM
    client is available, all workers fall back to rule-based analysis.
    """

    WEIGHTS = {
        "grammar_specialist": 0.20,
        "coherence_specialist": 0.15,
        "argumentation_specialist": 0.20,
        "tone_specialist": 0.10,
        "citation_specialist": 0.10,
        "plagiarism_specialist": 0.10,
        "research_specialist": 0.05,
        "review_specialist": 0.10,
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_llm_synthesis: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.workers = {
            "grammar_specialist": GrammarWorker(llm_client=llm_client),
            "coherence_specialist": CoherenceWorker(llm_client=llm_client),
            "argumentation_specialist": ArgumentationWorker(llm_client=llm_client),
            "tone_specialist": ToneWorker(llm_client=llm_client),
            "citation_specialist": CitationWorker(),
            "plagiarism_specialist": PlagiarismWorker(),
            "research_specialist": ResearchWorker(),
            "review_specialist": ReviewWorker(llm_client=llm_client),
        }
        self.critic = CriticWorker(llm_client=llm_client)
        self.synthesizer = SynthesisEngine(use_llm=use_llm_synthesis, llm_client=llm_client)

    async def run_evaluation(
        self,
        content: str,
        document_id: str | None = None,
        document_structure: Dict[str, Any] | None = None,
        citation_style: str = "harvard",
    ) -> EvaluationContext:
        """Execute complete evaluation cycle."""
        structured = document_structure or DocumentExtractor.parse_text_structure(content)
        normalized_content = structured["content"]

        # 1. Create evaluation board
        context_kwargs: Dict[str, Any] = {
            "document_content": normalized_content,
            "document_metadata": {
                "word_count": len(normalized_content.split()),
                "char_count": len(normalized_content),
                "citation_style": citation_style,
                **structured.get("metadata", {}),
            },
            "document_sections": structured.get("sections", []),
            "status": "in_progress",
        }
        if document_id:
            context_kwargs["document_id"] = UUID(document_id)

        context = EvaluationContext(**context_kwargs)

        # Update citation worker style if needed
        citation_worker = self.workers.get("citation_specialist")
        if hasattr(citation_worker, "citation_style"):
            citation_worker.citation_style = citation_style

        # 2. Run all specialist workers concurrently
        worker_names = list(self.workers.keys())

        async def _run_worker(name: str) -> tuple[str, WorkerResult]:
            try:
                result = await self.workers[name].run(context)
                return name, result
            except Exception as e:
                logger.error("Worker %s failed: %s", name, e)
                return name, WorkerResult(
                    worker_name=name,
                    score=0.0,
                    findings=[f"Worker failed: {str(e)}"],
                    flagged_items=[],
                    proposed_actions=[],
                    processing_time_ms=0,
                )

        results = await asyncio.gather(*[_run_worker(n) for n in worker_names])
        for name, result in results:
            context.worker_results[name] = result

        # 3. Synthesize findings
        synthesis = await self.synthesizer.synthesize(context)
        context.synthesis_reasoning = synthesis.reasoning
        context.final_actions = synthesis.actions

        # 4. Calculate aggregated scores
        context.final_scores = {
            name: result.score for name, result in context.worker_results.items()
        }
        context.overall_score = self._calculate_overall_score(context.final_scores, context)

        # 5. Critic QA review
        context.critic_review = await self.critic.review(context)

        # 6. Mark complete
        context.status = "completed"
        context.completed_at = datetime.now(timezone.utc)

        return context

    async def live_check(self, content: str, trigger: LiveTriggerType) -> list[EvaluationAction]:
        """Perform a fast, targeted check for live feedback."""
        actions: list[EvaluationAction] = []
        workers_to_run = []

        if trigger in [LiveTriggerType.PAUSE, LiveTriggerType.SENTENCE, LiveTriggerType.PARAGRAPH]:
            workers_to_run.append(self.workers["grammar_specialist"])
            workers_to_run.append(self.workers["tone_specialist"])

        if trigger == LiveTriggerType.PARAGRAPH:
            workers_to_run.append(self.workers["coherence_specialist"])

        async def _run_live_worker(worker):
            try:
                result = await worker.evaluate(content)
                return result.proposed_actions
            except Exception as e:
                logger.error("Live check worker failed: %s", e)
                return []

        results = await asyncio.gather(*[_run_live_worker(w) for w in workers_to_run])
        for proposed in results:
            actions.extend(proposed)

        return actions

    def _calculate_overall_score(self, scores: Dict[str, float], context: EvaluationContext | None = None) -> float:
        total = 0.0
        weight_sum = 0.0
        
        # Check if research is enabled
        research_enabled = settings.ENABLE_WEB_RESEARCH
        
        # Check if citations are present in the document
        has_citations = True
        if context and "citation_specialist" in context.worker_results:
            cit_res = context.worker_results["citation_specialist"]
            citations_count = cit_res.metadata.get("citations_found", 0) if cit_res.metadata else 0
            doc_content = context.document_content or ""
            doc_lower = doc_content.lower()
            reference_headers = ['references', 'bibliography', 'works cited', 'reference list', 'sources', 'citations']
            has_ref_section = any(h in doc_lower for h in reference_headers)
            if citations_count == 0 and not has_ref_section:
                has_citations = False

        for worker_name, weight in self.WEIGHTS.items():
            if worker_name in scores:
                if worker_name == "research_specialist" and not research_enabled:
                    continue
                if worker_name == "citation_specialist" and not has_citations:
                    continue
                if worker_name == "plagiarism_specialist" and scores[worker_name] >= 90.0:
                    continue
                    
                total += scores[worker_name] * weight
                weight_sum += weight
                
        if weight_sum == 0:
            return 0.0
        return round(total / weight_sum, 2)

    def get_runtime_profile(self) -> Dict[str, object]:
        """Return current runtime mode information."""
        worker_modes: Dict[str, str] = {}
        for worker_name, worker in self.workers.items():
            has_llm = getattr(worker, "llm_client", None) is not None
            llm_available = has_llm and worker.llm_client.available  # type: ignore[union-attr]
            worker_modes[worker_name] = "llm" if llm_available else "rule-based"

        llm_synthesis = (
            "llm" if (self.synthesizer.use_llm and self.synthesizer.llm_client and self.synthesizer.llm_client.available)
            else "rule-based"
        )

        return {
            "llm_synthesis_mode": llm_synthesis,
            "worker_modes": worker_modes,
            "web_research_enabled": settings.ENABLE_WEB_RESEARCH,
        }
