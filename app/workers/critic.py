from typing import Dict, Any, List
from datetime import datetime

from app.models.context import EvaluationContext, CriticReview


class CriticWorker:
    """
    Quality control reviewer.
    
    Responsibilities:
    - Check for inconsistencies between worker results
    - Validate score reasonableness
    - Flag low-confidence findings
    - Ensure actionability of recommendations
    """
    
    name = "critic"
    
    async def review(self, context: EvaluationContext) -> CriticReview:
        """Perform quality control review on evaluation results"""
        issues: List[Dict[str, Any]] = []
        
        # Check 1: Score variance between workers
        issues.extend(self._check_score_variance(context))
        
        # Check 2: Conflicting actions
        issues.extend(self._check_conflicting_actions(context))
        
        # Check 3: Plagiarism vs overall score consistency
        issues.extend(self._check_plagiarism_consistency(context))
        
        # Check 4: Low confidence findings
        issues.extend(self._check_confidence_levels(context))
        
        # Check 5: Missing worker results
        issues.extend(self._check_completeness(context))
        
        # Determine if results are approved
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        approved = len(critical_issues) == 0
        
        return CriticReview(
            approved=approved,
            issues=issues,
            timestamp=datetime.utcnow(),
        )
    
    def _check_score_variance(self, context: EvaluationContext) -> List[Dict]:
        """Check for unusual variance in worker scores"""
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
                    "max_worker": max(scores, key=scores.get),
                    "max_score": max_score,
                    "min_worker": min(scores, key=scores.get),
                    "min_score": min_score,
                },
            })
        
        return issues
    
    def _check_conflicting_actions(self, context: EvaluationContext) -> List[Dict]:
        """Check for conflicting recommended actions"""
        issues = []
        actions = context.final_actions
        
        # Group actions by target
        by_target: Dict[str, List] = {}
        for action in actions:
            target = action.target
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(action)
        
        # Check for conflicting actions on same target
        for target, target_actions in by_target.items():
            if len(target_actions) > 1:
                categories = set(a.category for a in target_actions)
                if len(categories) > 3:  # Too many different concerns
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
        """Check consistency between plagiarism score and overall score"""
        issues = []
        
        if "plagiarism_specialist" not in context.worker_results:
            return issues
        
        plagiarism_score = context.worker_results["plagiarism_specialist"].score
        overall = context.overall_score or 0
        
        # High overall score despite plagiarism concerns is suspicious
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
        """Check for too many low-confidence findings"""
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
        """Check that all expected workers have reported"""
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
        
        # Check for worker failures
        for worker_name, result in context.worker_results.items():
            if result.score == 0 and "failed" in ' '.join(result.findings).lower():
                issues.append({
                    "type": "worker_failure",
                    "severity": "warning",
                    "message": f"Worker '{worker_name}' may have failed",
                    "details": {"worker": worker_name, "findings": result.findings},
                })
        
        return issues
