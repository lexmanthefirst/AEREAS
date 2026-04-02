import re
from typing import List, Set, Tuple

from app.workers.base import BaseWorker
from app.models.context import WorkerResult, EvaluationAction, ActionType


class CitationWorker(BaseWorker):
    """Citation specialist for format validation and reference matching"""
    
    name = "citation_specialist"
    
    # Citation patterns for different styles
    CITATION_PATTERNS = {
        "harvard": [
            r'\(([A-Z][a-zA-Z]+),?\s*(\d{4})\)',              # (Author, 2024)
            r'\(([A-Z][a-zA-Z]+)\s+et\s+al\.?,?\s*(\d{4})\)', # (Author et al., 2024)
            r'([A-Z][a-zA-Z]+)\s+\((\d{4})\)',                # Author (2024)
            r'\(([A-Z][a-zA-Z]+)\s+and\s+([A-Z][a-zA-Z]+),?\s*(\d{4})\)',  # (Author and Author, 2024)
        ],
        "apa": [
            r'\(([A-Z][a-zA-Z]+),\s*(\d{4})\)',               # (Author, 2024)
            r'\(([A-Z][a-zA-Z]+)\s+et\s+al\.,\s*(\d{4})\)',   # (Author et al., 2024)
            r'([A-Z][a-zA-Z]+)\s+\((\d{4})\)',                # Author (2024)
            r'\(([A-Z][a-zA-Z]+)\s+&\s+([A-Z][a-zA-Z]+),\s*(\d{4})\)',  # (Author & Author, 2024)
        ],
        "mla": [
            r'\(([A-Z][a-zA-Z]+)\s+(\d+)\)',                  # (Author 45)
            r'\(([A-Z][a-zA-Z]+)\s+and\s+([A-Z][a-zA-Z]+)\s+(\d+)\)',  # (Author and Author 45)
        ],
    }
    
    def __init__(self, citation_style: str = "harvard"):
        """
        Initialize Citation Worker.
        
        Args:
            citation_style: Citation style to validate against (harvard, apa, mla)
        """
        self.citation_style = citation_style.lower()
    
    async def evaluate(self, document: str) -> WorkerResult:
        """Analyze document for citations and references"""
        sentences = self._split_sentences(document)
        
        findings: List[str] = []
        flagged_items: List[dict] = []
        actions: List[EvaluationAction] = []
        
        # Extract all citations from document
        citations_found = self._extract_citations(document)
        
        # Check for common citation issues
        issues = self._check_citation_issues(document, sentences, citations_found)
        
        for issue in issues:
            findings.append(issue["message"])
            flagged_items.append(issue)
            actions.append(EvaluationAction(
                type=issue["severity"],
                target=issue.get("target", "document"),
                category="citation",
                reasoning=issue["message"],
                confidence=issue.get("confidence", 0.8),
                suggestion=issue.get("suggestion"),
            ))
        
        # Calculate score based on citation quality
        score = self._calculate_citation_score(
            document=document,
            citations_count=len(citations_found),
            issues_count=len(issues),
            word_count=len(document.split())
        )
        
        if not findings:
            findings.append(f"Citations appear properly formatted ({self.citation_style} style)")
        
        return WorkerResult(
            score=score,
            findings=findings,
            flagged_items=flagged_items,
            proposed_actions=actions,
            metadata={
                "citations_found": len(citations_found),
                "citation_style": self.citation_style,
                "unique_authors": list(set(c["author"] for c in citations_found)),
            },
        )
    
    def _extract_citations(self, document: str) -> List[dict]:
        """Extract all citations from the document"""
        citations = []
        patterns = self.CITATION_PATTERNS.get(self.citation_style, self.CITATION_PATTERNS["harvard"])
        
        for pattern in patterns:
            for match in re.finditer(pattern, document):
                groups = match.groups()
                citations.append({
                    "text": match.group(0),
                    "author": groups[0] if groups else "",
                    "year": groups[-1] if len(groups) > 1 and groups[-1].isdigit() else "",
                    "position": match.start(),
                })
        
        return citations
    
    def _check_citation_issues(
        self, document: str, sentences: List[str], citations: List[dict]
    ) -> List[dict]:
        """Check for common citation issues"""
        issues: List[dict] = []
        
        # Issue 1: No citations in academic essay
        word_count = len(document.split())
        if word_count > 200 and len(citations) == 0:
            issues.append({
                "type": "missing_citations",
                "message": "No citations found in the document",
                "severity": ActionType.CRITICAL_REVISION,
                "suggestion": "Academic writing should include citations to support claims",
                "confidence": 0.95,
            })
        
        # Issue 2: Too few citations for length
        expected_citations = word_count // 200  # Roughly 1 citation per 200 words
        if len(citations) > 0 and len(citations) < expected_citations // 2:
            issues.append({
                "type": "insufficient_citations",
                "message": f"Only {len(citations)} citations for {word_count} words",
                "severity": ActionType.MODERATE_REVISION,
                "suggestion": "Consider adding more citations to support your arguments",
                "confidence": 0.7,
            })
        
        # Issue 3: Check for malformed citations
        malformed = self._find_malformed_citations(document)
        for mf in malformed:
            issues.append({
                "type": "malformed_citation",
                "message": f"Possible malformed citation: '{mf}'",
                "severity": ActionType.MINOR_IMPROVEMENT,
                "suggestion": f"Check citation format ({self.citation_style} style)",
                "target": mf,
                "confidence": 0.6,
            })
        
        # Issue 4: Check for missing references section
        has_references = self._has_references_section(document)
        if len(citations) > 0 and not has_references:
            issues.append({
                "type": "missing_references",
                "message": "Citations found but no References/Bibliography section detected",
                "severity": ActionType.CRITICAL_REVISION,
                "suggestion": "Add a References section at the end of the document",
                "confidence": 0.85,
            })
        
        return issues
    
    def _find_malformed_citations(self, document: str) -> List[str]:
        """Find potentially malformed citations"""
        malformed = []
        
        # Patterns that look like citations but are malformed
        suspicious_patterns = [
            r'\([A-Z][a-z]+\s+\d{4}\)',           # Missing comma (Author 2024)
            r'\([a-z][a-zA-Z]*,?\s*\d{4}\)',      # Lowercase author
            r'\([A-Z][a-zA-Z]+,?\s*\d{2}\)',      # 2-digit year
            r'\([A-Z][a-zA-Z]+\s*,\s*,\s*\d{4}\)', # Double comma
        ]
        
        valid_patterns = self.CITATION_PATTERNS.get(self.citation_style, [])
        
        for pattern in suspicious_patterns:
            for match in re.finditer(pattern, document):
                text = match.group(0)
                # Check if it matches any valid pattern
                is_valid = any(re.match(vp, text) for vp in valid_patterns)
                if not is_valid:
                    malformed.append(text)
        
        return malformed[:5]  # Limit to 5 findings
    
    def _has_references_section(self, document: str) -> bool:
        """Check if document has a references section"""
        doc_lower = document.lower()
        reference_headers = [
            'references', 'bibliography', 'works cited',
            'reference list', 'sources', 'citations'
        ]
        return any(header in doc_lower for header in reference_headers)
    
    def _calculate_citation_score(
        self, document: str, citations_count: int, issues_count: int, word_count: int
    ) -> float:
        """Calculate citation quality score"""
        if word_count < 100:
            return 100.0  # Too short to require citations
        
        base_score = 100.0
        
        # Deduct for missing citations
        if citations_count == 0:
            base_score -= 40
        
        # Deduct for issues
        base_score -= issues_count * 15
        
        # Small bonus for good citation density
        expected = word_count // 200
        if expected > 0 and citations_count >= expected:
            base_score = min(100, base_score + 5)
        
        return max(0, round(base_score, 2))
