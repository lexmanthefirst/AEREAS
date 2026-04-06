"""Workers package"""
from app.workers.base import BaseWorker
from app.workers.grammar import GrammarWorker
from app.workers.coherence import CoherenceWorker
from app.workers.argumentation import ArgumentationWorker
from app.workers.tone import ToneWorker
from app.workers.citation import CitationWorker
from app.workers.plagiarism import PlagiarismWorker
from app.workers.critic import CriticWorker
from app.workers.research import ResearchWorker
from app.workers.reviewer import ReviewWorker
from app.workers.schemas import (
    GrammarIssue,
    GrammarOutput,
    CoherenceTransition,
    CoherenceOutput,
    ArgumentComponent,
    ArgumentationOutput,
    ToneIssue,
    ToneOutput,
    ReviewFinding,
    ReviewOutput,
    SynthesisAction,
    SynthesisOutput,
    CriticLLMOutput,
)

__all__ = [
    "BaseWorker",
    "GrammarWorker",
    "CoherenceWorker",
    "ArgumentationWorker",
    "ToneWorker",
    "CitationWorker",
    "PlagiarismWorker",
    "CriticWorker",
    "ResearchWorker",
    "ReviewWorker",
    "GrammarIssue",
    "GrammarOutput",
    "CoherenceTransition",
    "CoherenceOutput",
    "ArgumentComponent",
    "ArgumentationOutput",
    "ToneIssue",
    "ToneOutput",
    "ReviewFinding",
    "ReviewOutput",
    "SynthesisAction",
    "SynthesisOutput",
    "CriticLLMOutput",
]
