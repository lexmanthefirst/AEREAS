from typing import List, Optional
from datetime import datetime

from app.models.context import (
    EvaluationContext,
    SynthesisResult,
    EvaluationAction,
    ActionType,
)
from app.utils.logger import logger


class SynthesisEngine:
    """
    Synthesizes all worker findings into coherent, prioritized feedback.
    
    Uses LLM (Gemini Pro) with rules-based fallback if:
    - LLM is not configured
    - LLM call fails
    """
    
    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        """
        Initialize Synthesis Engine.
        
        Args:
            use_llm: If True, use Gemini Pro for synthesis.
            api_key: Optional Gemini API key (uses env var if not provided).
        """
        self.use_llm = use_llm
        self.api_key = api_key
        self.client = None
        self.model_name = "gemini-2.5-flash"
        
        if use_llm:
            self._init_llm()
    
    def _init_llm(self):
        """Initialize Gemini LLM client"""
        try:
            from google import genai
            import os
            
            api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                logger.warning("No GEMINI_API_KEY found. Using rules-based synthesis.")
                self.use_llm = False
        except Exception as e:
            logger.warning("Could not initialize LLM: %s. Using rules-based synthesis.", e)
            self.use_llm = False
    
    async def synthesize(self, context: EvaluationContext) -> SynthesisResult:
        """
        Synthesize all worker findings into unified recommendations.
        
        Args:
            context: The evaluation context with all worker results.
            
        Returns:
            SynthesisResult with reasoning and prioritized actions.
        """
        if self.use_llm and self.client is not None:
            try:
                return await self._llm_synthesis(context)
            except Exception as e:
                logger.warning("LLM synthesis failed: %s. Falling back to rules.", e)
                return self._rules_synthesis(context)
        
        return self._rules_synthesis(context)
    
    async def _llm_synthesis(self, context: EvaluationContext) -> SynthesisResult:
        """Use Gemini Pro for intelligent synthesis"""
        client = self.client
        if client is None:
            raise RuntimeError("LLM client is not initialized")

        prompt = self._build_synthesis_prompt(context)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        response_text = (getattr(response, "text", "") or "").strip()
        if not response_text:
            raise ValueError("LLM returned empty synthesis response")
        
        # Parse LLM response
        return self._parse_llm_response(response_text, context)
    
    def _rules_synthesis(self, context: EvaluationContext) -> SynthesisResult:
        """Fallback rules-based synthesis"""
        all_actions: List[EvaluationAction] = []
        
        # Collect all actions from the workers
        for worker_name, result in context.worker_results.items():
            all_actions.extend(result.proposed_actions)
        
        # Prioritize by severity and confidence
        all_actions.sort(key=lambda a: (
            self._severity_rank(a.type),
            a.confidence,
        ), reverse=True)
        
        # Deduplicate similar actions
        deduplicated = self._deduplicate_actions(all_actions)
        
        # Generate reasoning summary
        reasoning = self._generate_reasoning(context)
        
        return SynthesisResult(
            reasoning=reasoning,
            actions=deduplicated[:20],  # Top 20 most important
        )
    
    def _severity_rank(self, action_type: ActionType) -> int:
        """Rank action types by severity"""
        ranks = {
            ActionType.CRITICAL_REVISION: 4,
            ActionType.MODERATE_REVISION: 3,
            ActionType.MINOR_IMPROVEMENT: 2,
            ActionType.POSITIVE_FEEDBACK: 1,
        }
        return ranks.get(action_type, 0)
    
    def _deduplicate_actions(self, actions: List[EvaluationAction]) -> List[EvaluationAction]:
        """Remove duplicate or very similar actions"""
        seen_targets = set()
        unique = []
        
        for action in actions:
            key = (action.target, action.category)
            if key not in seen_targets:
                seen_targets.add(key)
                unique.append(action)
        
        return unique
    
    def _generate_reasoning(self, context: EvaluationContext) -> str:
        """Generate human-readable reasoning from worker results"""
        lines = ["## Evaluation Summary\n"]
        
        # Overall score
        if context.overall_score is not None:
            lines.append(f"**Overall Score: {context.overall_score:.1f}/100**\n")
        
        # Worker summaries
        lines.append("\n### Analysis Breakdown:\n")
        for worker_name, result in context.worker_results.items():
            category = worker_name.replace("_specialist", "").title()
            score = result.score
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            
            findings_preview = result.findings[0] if result.findings else "No issues"
            lines.append(f"- **{category}**: {status} {score:.1f}/100 - {findings_preview}")
        
        # Key concerns
        critical_count = sum(
            1 for r in context.worker_results.values()
            for a in r.proposed_actions
            if a.type == ActionType.CRITICAL_REVISION
        )
        
        if critical_count > 0:
            lines.append(f"\n### ⚠️ Critical Concerns: {critical_count} issues require immediate attention")
        
        return "\n".join(lines)
    
    def _build_synthesis_prompt(self, context: EvaluationContext) -> str:
        """Build prompt for LLM synthesis"""
        prompt_parts = [
            "You are an expert academic writing reviewer. Analyze the following evaluation results",
            "and synthesize them into clear, actionable feedback for a student.\n\n",
            "## Document Scores:\n",
        ]
        
        for worker_name, result in context.worker_results.items():
            prompt_parts.append(f"- {worker_name}: {result.score}/100\n")
            prompt_parts.append(f"  Findings: {', '.join(result.findings[:3])}\n")
        
        prompt_parts.extend([
            "\n## Task:",
            "1. Identify the 3 most important areas for improvement",
            "2. Provide specific, actionable suggestions",
            "3. Highlight any strengths worth preserving",
            "4. Give an overall assessment in 2-3 sentences",
            "\nBe constructive and encouraging while being honest about areas needing work.",
        ])
        
        return "".join(prompt_parts)
    
    def _parse_llm_response(self, response_text: str, context: EvaluationContext) -> SynthesisResult:
        """Parse LLM response into SynthesisResult"""
        # For now, use the response as reasoning and fall back to rules for actions
        rules_result = self._rules_synthesis(context)
        
        return SynthesisResult(
            reasoning=response_text,
            actions=rules_result.actions,
        )
