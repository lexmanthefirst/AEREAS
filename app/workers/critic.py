"""CriticWorker — quality control reviewer with LLM-powered quality check."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from app.models.context import CriticReview, EvaluationContext
from app.utils.logger import logger
from app.workers.schemas import CriticLLMOutput

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class CriticWorker:
    """Quality control reviewer.

    Validates evaluation consistency, flags low-confidence findings,
    and checks actionability. Uses LLM when available for deeper QA.
    """

    name = "critic"

    SYSTEM_PROMPT = (
        "You are a quality-control reviewer for an academic writing evaluation system.\n"
        "You will receive the evaluation results: worker scores, proposed actions, and synthesis reasoning.\n\n"
        "Your job:\n"
        "1. Check whether the suggested actions are specific and actionable (not vague).\n"
        "2. Check for contradictions between different worker suggestions.\n"
        "3. Check whether severity ratings are proportionate.\n"
        "4. Check whether the overall score is consistent with the individual findings.\n"
        "5. Flag any issues you find.\n\n"
        "Set 'approved' to false ONLY if there are serious quality problems that would "
        "mislead the student (e.g., contradictory advice, wildly inconsistent scores).\n"
        "Minor concerns should still result in approved=true with issues listed."
    )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    async def review(self, context: EvaluationContext) -> CriticReview:
        """Perform quality control review on evaluation results."""
        issues: List[Dict[str, Any]] = []

        # Rule-based checks always run
        issues.extend(self._check_score_variance(context))
        issues.extend(self._check_conflicting_actions(context))
        issues.extend(self._check_plagiarism_consistency(context))
        issues.extend(self._check_confidence_levels(context))
        issues.extend(self._check_completeness(context))

        # LLM quality check when available
        if self.llm_client and self.llm_client.available:
            try:
                llm_result = await self._llm_quality_check(context)
                for issue_text in llm_result.issues:
                    issues.append({
                        "type": "llm_quality_check",
                        "severity": "warning",
                        "message": issue_text,
                    })
                if not llm_result.approved:
                    issues.append({
                        "type": "llm_quality_check",
                        "severity": "critical",
                        "message": "LLM critic flagged evaluation as potentially misleading.",
                    })
            except Exception as exc:
                logger.warning("LLM critic check failed: %s — using rule-based only.", exc)

        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        approved = len(critical_issues) == 0

        return CriticReview(
            approved=approved,
            issues=issues,
            timestamp=datetime.now(timezone.utc),
        )

    async def _llm_quality_check(self, context: EvaluationContext) -> CriticLLMOutput:
        """Ask LLM to review the quality of the evaluation feedback."""
        scores_text = "\n".join(
            f"- {name}: {score:.1f}/100"
            for name, score in context.final_scores.items()
        )
        actions_text = "\n".join(
            f"- [{a.type.value}] {a.category}: {a.reasoning} | Suggestion: {a.suggestion or 'N/A'}"
            for a in context.final_actions[:12]
        )
        content = (
            f"Overall score: {context.overall_score}\n\n"
            f"## Worker Scores\n{scores_text}\n\n"
            f"## Proposed Actions\n{actions_text}\n\n"
            f"## Synthesis Reasoning\n{context.synthesis_reasoning or 'N/A'}"
        )

        return await self.llm_client.generate(  # type: ignore[union-attr]
            system_prompt=self.SYSTEM_PROMPT,
            content=content,
            response_schema=CriticLLMOutput,
        )

    def _check_score_variance(self, context: EvaluationContext) -> List[Dict]:
        issues = []
        scores = context.final_scores

        if len(scores) < 2:
            return issues

        max_score = max(scores.values())
        min_score = min(scores.values())
        variance = max_score - min_score

        if variance > 50:
            issues.append({
                "type": "score_variance",
                "severity": "warning",
                "message": f"Large variance between worker scores ({variance:.1f} points)",
                "details": {
                    "max_worker": max(scores, key=scores.get),  # type: ignore[arg-type]
                    "max_score": max_score,
                    "min_worker": min(scores, key=scores.get),  # type: ignore[arg-type]
                    "min_score": min_score,
                },
            })

        return issues

    def _check_conflicting_actions(self, context: EvaluationContext) -> List[Dict]:
        issues = []
        actions = context.final_actions

        by_target: Dict[str, List] = {}
        for action in actions:
            target = action.target
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(action)

        for target, target_actions in by_target.items():
            if len(target_actions) > 1:
                categories = set(a.category for a in target_actions)
                if len(categories) > 3:
                    issues.append({
                        "type": "action_conflict",
                        "severity": "warning",
                        "message": f"Multiple conflicting concerns for '{target}'",
                        "details": {
                            "target": target,
                            "categories": list(categories),
                        },
                    })

        return issues

    def _check_plagiarism_consistency(self, context: EvaluationContext) -> List[Dict]:
        issues = []

        if "plagiarism_specialist" not in context.worker_results:
            return issues

        plagiarism_score = context.worker_results["plagiarism_specialist"].score
        overall = context.overall_score or 0

        if plagiarism_score < 60 and overall > 80:
            issues.append({
                "type": "plagiarism_concern",
                "severity": "critical",
                "message": "High overall score despite plagiarism concerns",
                "details": {
                    "plagiarism_score": plagiarism_score,
                    "overall_score": overall,
                },
            })

        return issues

    def _check_confidence_levels(self, context: EvaluationContext) -> List[Dict]:
        issues = []

        low_confidence_count = 0
        total_actions = 0

        for action in context.final_actions:
            total_actions += 1
            if action.confidence < 0.5:
                low_confidence_count += 1

        if total_actions > 0 and low_confidence_count / total_actions > 0.5:
            issues.append({
                "type": "low_confidence",
                "severity": "warning",
                "message": f"Many actions have low confidence ({low_confidence_count}/{total_actions})",
                "details": {
                    "low_confidence_count": low_confidence_count,
                    "total_actions": total_actions,
                },
            })

        return issues

    def _check_completeness(self, context: EvaluationContext) -> List[Dict]:
        issues = []

        expected_workers = {
            "grammar_specialist",
            "coherence_specialist",
            "argumentation_specialist",
            "tone_specialist",
            "citation_specialist",
            "plagiarism_specialist",
            "research_specialist",
            "review_specialist",
        }

        actual_workers = set(context.worker_results.keys())
        missing = expected_workers - actual_workers

        if missing:
            issues.append({
                "type": "incomplete_evaluation",
                "severity": "warning",
                "message": f"Missing worker results: {', '.join(missing)}",
                "details": {"missing_workers": list(missing)},
            })

        for worker_name, result in context.worker_results.items():
            if result.score == 0 and "failed" in " ".join(result.findings).lower():
                issues.append({
                    "type": "worker_failure",
                    "severity": "warning",
                    "message": f"Worker '{worker_name}' may have failed",
                    "details": {"worker": worker_name, "findings": result.findings},
                })

        return issues
