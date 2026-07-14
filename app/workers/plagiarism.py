"""Plagiarism specialist — heuristic checks for originality indicators."""

import re
from typing import List

from app.models.context import ActionType, EvaluationAction, WorkerResult
from app.workers.base import BaseWorker


class PlagiarismWorker(BaseWorker):
    """Plagiarism specialist using heuristic analysis."""

    name = "plagiarism_specialist"

    async def evaluate(self, document: str) -> WorkerResult:
        sentences = self._split_sentences(document)
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []

        issues = self._heuristic_checks(document, sentences)
        flagged_items = issues
        for item in issues:
            findings.append(item["message"])
            span = self.find_span(document, item.get("target")) if isinstance(item.get("target"), str) else None
            actions.append(
                EvaluationAction(
                    type=item["severity"],
                    target=item.get("target", "document"),
                    category="plagiarism",
                    reasoning=item["message"],
                    confidence=item.get("confidence", 0.6),
                    suggestion=item.get("suggestion"),
                    highlight=span,
                )
            )

        score = self._calculate_originality_score(
            total_sentences=len(sentences),
            flagged_items=flagged_items,
        )

        if not findings:
            findings.append("No plagiarism indicators detected (heuristic check)")

        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={"mode": "heuristic", "total_sentences": len(sentences)},
        )

    def _heuristic_checks(self, document: str, sentences: List[str]) -> List[dict]:
        issues: List[dict] = []

        # Repeated phrases (self-plagiarism indicator)
        repeated = self._find_repeated_phrases(sentences)
        if repeated:
            issues.append(
                {
                    "type": "repeated_phrases",
                    "message": f"Found {len(repeated)} repeated phrases (possible self-plagiarism)",
                    "severity": ActionType.MINOR_IMPROVEMENT,
                    "phrases": repeated[:5],
                    "suggestion": "Vary your wording to avoid repetition",
                    "confidence": 0.5,
                }
            )

        # Unnaturally perfect prose (copy-paste indicator)
        for sign in self._check_perfect_prose(document):
            issues.append(
                {
                    "type": "perfect_prose",
                    "message": sign["message"],
                    "severity": ActionType.MODERATE_REVISION,
                    "confidence": 0.6,
                    "suggestion": "Verify this is original content and not copied",
                }
            )

        # Inconsistent writing style
        style_issues = self._check_style_consistency(sentences)
        if style_issues:
            issues.append(
                {
                    "type": "style_inconsistency",
                    "message": "Writing style varies significantly across sections",
                    "severity": ActionType.MODERATE_REVISION,
                    "details": style_issues,
                    "confidence": 0.5,
                    "suggestion": "Review sections for consistent voice and style",
                }
            )

        # Factual claims without citations
        uncited_facts = self._find_uncited_facts(sentences)
        for fact in uncited_facts[:3]:
            issues.append(
                {
                    "type": "uncited_fact",
                    "message": f"Factual claim without citation: '{fact[:50]}...'",
                    "severity": ActionType.MODERATE_REVISION,
                    "target": fact,
                    "confidence": 0.7,
                    "suggestion": "Add citation for factual claims",
                }
            )

        return issues

    def _find_repeated_phrases(self, sentences: List[str]) -> List[str]:
        all_ngrams: dict = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for i in range(len(words) - 3):
                ngram = " ".join(words[i : i + 4])
                all_ngrams[ngram] = all_ngrams.get(ngram, 0) + 1
        return [phrase for phrase, count in all_ngrams.items() if count >= 3]

    def _check_perfect_prose(self, document: str) -> List[dict]:
        issues = []
        if re.search(r"\[\d+\]", document):
            issues.append({"message": "Wikipedia-style reference markers found [1]"})
        if re.search(r"<[^>]+>", document):
            issues.append({"message": "HTML tags detected — possible web copy"})
        if re.search(r"https?://|www\.", document):
            issues.append({"message": "URLs found in document — possible direct copy"})
        return issues

    def _check_style_consistency(self, sentences: List[str]) -> List[dict]:
        lengths = [len(s.split()) for s in sentences]
        if not lengths:
            return []
        avg_length = sum(lengths) / len(lengths)
        variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
        if variance > 100:
            return [{"high_length_variance": variance}]
        return []

    def _find_uncited_facts(self, sentences: List[str]) -> List[str]:
        fact_patterns = [
            r"\bstudies show\b",
            r"\bresearch indicates\b",
            r"\baccording to\b",
            r"\b\d+%\b",
            r"\bstatistics\b",
            r"\bin \d{4}\b",
            r"\bexperts ?(say|believe|agree)\b",
        ]
        uncited = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            has_fact = any(re.search(p, sentence_lower) for p in fact_patterns)
            has_citation = bool(re.search(r"\([A-Z][a-z]+.*?\d{4}\)", sentence))
            if has_fact and not has_citation:
                uncited.append(sentence)
        return uncited

    def _calculate_originality_score(self, total_sentences: int, flagged_items: List[dict]) -> float:
        if total_sentences == 0:
            return 100.0
        
        # Deduct based on severity of each flagged item
        deductions = 0.0
        for item in flagged_items:
            severity = item.get("severity")
            if severity == ActionType.CRITICAL_REVISION:
                deductions += 30.0
            elif severity == ActionType.MODERATE_REVISION:
                deductions += 10.0
            elif severity == ActionType.MINOR_IMPROVEMENT:
                deductions += 3.0
                
        score = max(0.0, 100.0 - deductions)
        return round(score, 2)
