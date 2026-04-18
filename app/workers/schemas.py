"""Pydantic models for structured LLM output from specialist workers."""

from typing import List, Optional

from pydantic import BaseModel, Field


# -- Grammar ------------------------------------------------------------------

class GrammarIssue(BaseModel):
    quoted_text: str = Field(description="Exact text from the document containing the error")
    error_type: str = Field(description="Category: subject_verb, punctuation, tense, article, spelling, syntax, parallel_structure, fragment")
    correction: str = Field(description="Corrected version of the quoted text")
    explanation: str = Field(description="Brief explanation of the grammar rule violated")


class GrammarOutput(BaseModel):
    issues: List[GrammarIssue] = Field(default_factory=list)
    overall_assessment: str = ""
    score: float = Field(ge=0, le=100, default=80.0, description="Grammar quality score 0-100")


# -- Coherence ----------------------------------------------------------------

class CoherenceTransition(BaseModel):
    from_paragraph_text: str = Field(description="Last sentence of the preceding paragraph")
    to_paragraph_text: str = Field(description="First sentence of the following paragraph")
    issue: str = Field(description="Description of the transition weakness")
    suggestion: str = Field(description="Specific bridging sentence or transition suggestion")
    severity: str = Field(description="weak or moderate")


class CoherenceOutput(BaseModel):
    transitions: List[CoherenceTransition] = Field(default_factory=list)
    topic_flow_assessment: str = ""
    score: float = Field(ge=0, le=100, default=80.0, description="Coherence quality score 0-100")


# -- Argumentation ------------------------------------------------------------

class ArgumentComponent(BaseModel):
    quoted_text: str = Field(description="Exact text from the document")
    component_type: str = Field(description="claim, premise, evidence, or counterargument")
    strength: str = Field(description="strong, adequate, or weak")
    reasoning: str = Field(default="", description="Why this component is rated as such")


class ArgumentationOutput(BaseModel):
    thesis_statement: Optional[str] = Field(default=None, description="The main thesis if identified")
    components: List[ArgumentComponent] = Field(default_factory=list)
    logical_gaps: List[str] = Field(default_factory=list, description="Logical gaps or unsupported assertions")
    overall_assessment: str = ""
    score: float = Field(ge=0, le=100, default=70.0, description="Argumentation quality score 0-100")


# -- Tone ---------------------------------------------------------------------

class ToneIssue(BaseModel):
    quoted_text: str = Field(description="Exact text containing the informality or tone problem")
    issue_type: str = Field(description="contraction, colloquialism, hedging, intensifier, vague_language, first_person, rhetorical_question")
    formal_alternative: str = Field(description="Academic alternative for the quoted text")
    explanation: str = Field(default="", description="Why this is inappropriate in academic writing")


class ToneOutput(BaseModel):
    issues: List[ToneIssue] = Field(default_factory=list)
    overall_formality: str = Field(default="formal", description="formal, mostly_formal, mixed, or informal")
    score: float = Field(ge=0, le=100, default=85.0, description="Academic tone score 0-100")


# -- Review (holistic) --------------------------------------------------------

class ReviewFinding(BaseModel):
    finding: str = Field(description="Description of the issue or observation")
    severity: str = Field(description="critical, moderate, or minor")
    target: str = Field(description="What part of the document this applies to")
    suggestion: str = Field(default="", description="Specific actionable recommendation")
    quoted_text: Optional[str] = Field(default=None, description="Relevant text from the document")


class ReviewOutput(BaseModel):
    findings: List[ReviewFinding] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    overall_assessment: str = ""
    score: float = Field(ge=0, le=100, default=75.0, description="Overall review score 0-100")


# -- Synthesis ----------------------------------------------------------------

class SynthesisAction(BaseModel):
    severity: str = Field(description="critical, moderate, minor, or positive")
    target: str = Field(description="What part of the document to address")
    category: str = Field(description="grammar, coherence, argumentation, tone, citation, plagiarism, research, review")
    reasoning: str = Field(description="Why this action matters")
    suggestion: str = Field(default="", description="Specific recommendation")
    quoted_text: Optional[str] = Field(default=None, description="Relevant text from the document")


class SynthesisOutput(BaseModel):
    reasoning: str = Field(description="Comprehensive evaluation summary in Markdown")
    prioritized_actions: List[SynthesisAction] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    overall_assessment: str = ""


# -- Critic -------------------------------------------------------------------

class CriticLLMOutput(BaseModel):
    issues: List[str] = Field(default_factory=list, description="Quality concerns about the evaluation feedback")
    approved: bool = Field(default=True, description="Whether the evaluation meets quality standards")
