from typing import Dict, Any
from uuid import UUID
from datetime import datetime

from app.models.context import EvaluationContext, WorkerResult, LiveTriggerType, EvaluationAction
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
from app.core.config import settings


class SupervisorAgent:
    """
    Lead Reviewer - Orchestrates the evaluation workflow.
    
    Implements the "Morning Rounds" pattern where specialists
    report to a supervisor who synthesizes their findings..
    """
    
    # Scoring weights for each dimension
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
    
    def __init__(self, use_models: bool = False, use_llm_synthesis: bool = False):
        """
        Initialize SupervisorAgent.
        
        Args:
            use_models: If True, load NLP models in workers.
            use_llm_synthesis: If True, use Gemini Pro for synthesis.
        """
        self.workers = {
            "grammar_specialist": GrammarWorker(use_model=use_models),
            "coherence_specialist": CoherenceWorker(use_model=use_models),
            "argumentation_specialist": ArgumentationWorker(use_model=use_models),
            "tone_specialist": ToneWorker(use_model=use_models),
            "citation_specialist": CitationWorker(),
            "plagiarism_specialist": PlagiarismWorker(use_model=use_models),
            "research_specialist": ResearchWorker(),
            "review_specialist": ReviewWorker(),
        }
        self.critic = CriticWorker()
        self.synthesizer = SynthesisEngine(use_llm=use_llm_synthesis)
    
    async def run_evaluation(
        self,
        content: str,
        document_id: str | None = None,
        document_structure: Dict[str, Any] | None = None,
    ) -> EvaluationContext:
        """
        Execute complete evaluation cycle.
        
        Args:
            content: Document text to evaluate.
            document_id: Optional document ID (generated if not provided).
            
        Returns:
            EvaluationContext with all results.
        """
        structured = document_structure or DocumentExtractor.parse_text_structure(content)
        normalized_content = structured["content"]
        # 1. Create evaluation board (whiteboard)
        context_kwargs = {
            "document_content": normalized_content,
            "document_metadata": {
                "word_count": len(normalized_content.split()),
                "char_count": len(normalized_content),
                **structured.get("metadata", {}),
            },
            "document_sections": structured.get("sections", []),
            "status": "in_progress",
        }
        if document_id:
            context_kwargs["document_id"] = UUID(document_id)

        context = EvaluationContext(**context_kwargs)
        
        # 2. Call each specialist worker
        ordered_workers = [
            "grammar_specialist",
            "coherence_specialist",
            "argumentation_specialist",
            "tone_specialist",
            "citation_specialist",
            "plagiarism_specialist",
            "research_specialist",
            "review_specialist",
        ]
        for worker_name in ordered_workers:
            worker = self.workers[worker_name]
            try:
                result: WorkerResult = await worker.run(context)
                context.worker_results[worker_name] = result
            except Exception as e:
                # Log error but continue with other workers
                context.worker_results[worker_name] = WorkerResult(
                    worker_name=worker_name,
                    score=0.0,
                    findings=[f"Worker failed: {str(e)}"],
                    flagged_items=[],
                    proposed_actions=[],
                    processing_time_ms=0,
                )
        
        # 3. Synthesize all findings
        synthesis = await self.synthesizer.synthesize(context)
        context.synthesis_reasoning = synthesis.reasoning
        context.final_actions = synthesis.actions
        
        # 4. Calculate aggregated scores
        context.final_scores = {
            name: result.score
            for name, result in context.worker_results.items()
        }
        context.overall_score = self._calculate_overall_score(context.final_scores)
        
        # 5. Critic QA review
        context.critic_review = await self.critic.review(context)
        
        # 6. Mark complete
        context.status = "completed"
        context.completed_at = datetime.utcnow()
        
        return context

    async def live_check(self, content: str, trigger: LiveTriggerType) -> list[EvaluationAction]:
        """
        Perform a fast, targeted check for live feedback.
        
        Args:
            content: Current document content.
            trigger: What triggered the check (pause, sentence, paragraph).
            
        Returns:
            List of actionable feedback items.
        """
        actions = []
        
        # Determine which workers to run based on trigger
        workers_to_run = []
        
        # Always run grammar checks on any pause/sentence
        if trigger in [LiveTriggerType.PAUSE, LiveTriggerType.SENTENCE, LiveTriggerType.PARAGRAPH]:
            workers_to_run.append(self.workers["grammar_specialist"])
            workers_to_run.append(self.workers["tone_specialist"])
            
        # Run heavier checks on paragraph completion
        if trigger == LiveTriggerType.PARAGRAPH:
            workers_to_run.append(self.workers["coherence_specialist"])
        
        # Execute selected workers (concurrently ideally, but sequential is fine for now)
        for worker in workers_to_run:
            try:
                # We create a temporary context just for this check
                # Note: creating a full context might be overkill but keeps interface consistent
                # Ideally workers should have a lighter `evaluate_snippet` method
                result = await worker.evaluate(content)
                actions.extend(result.proposed_actions)
            except Exception as e:
                logger.error("Live check worker failed: %s", e)
                
        return actions
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted average of worker scores"""
        total = 0.0
        weight_sum = 0.0
        
        for worker_name, weight in self.WEIGHTS.items():
            if worker_name in scores:
                total += scores[worker_name] * weight
                weight_sum += weight
        
        if weight_sum == 0:
            return 0.0
        
        return round(total / weight_sum, 2)

    def get_runtime_profile(self) -> Dict[str, object]:
        """Return current runtime mode information for transparency in API responses."""
        worker_modes: Dict[str, str] = {}
        for worker_name, worker in self.workers.items():
            use_model = bool(getattr(worker, "use_model", False))
            has_model = getattr(worker, "model", None) is not None
            worker_modes[worker_name] = "model-backed" if (use_model and has_model) else "rule-based"

        llm_synthesis_mode = (
            "model-backed"
            if (self.synthesizer.use_llm and getattr(self.synthesizer, "client", None) is not None)
            else "rule-based"
        )

        return {
            "llm_synthesis_mode": llm_synthesis_mode,
            "worker_modes": worker_modes,
            "web_research_enabled": settings.ENABLE_WEB_RESEARCH,
        }
