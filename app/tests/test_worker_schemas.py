"""Tests for worker output schemas — validates Pydantic model constraints."""

import pytest
from pydantic import ValidationError

from app.workers.schemas import (
    ArgumentComponent,
    ArgumentationOutput,
    CoherenceOutput,
    CoherenceTransition,
    CriticLLMOutput,
    GrammarIssue,
    GrammarOutput,
    ReviewFinding,
    ReviewOutput,
    SynthesisAction,
    SynthesisOutput,
    ToneIssue,
    ToneOutput,
)


class TestGrammarSchemas:
    def test_grammar_issue_valid(self):
        issue = GrammarIssue(
            quoted_text="She have a car.",
            error_type="subject_verb",
            correction="She has a car.",
            explanation="Subject-verb agreement error.",
        )
        assert issue.quoted_text == "She have a car."

    def test_grammar_output_defaults(self):
        output = GrammarOutput()
        assert output.issues == []
        assert output.score == 80.0

    def test_grammar_output_score_bounds(self):
        with pytest.raises(ValidationError):
            GrammarOutput(score=101.0)
        with pytest.raises(ValidationError):
            GrammarOutput(score=-1.0)

    def test_grammar_output_from_json(self):
        raw = '{"issues": [{"quoted_text": "test", "error_type": "spelling", "correction": "test2", "explanation": "typo"}], "overall_assessment": "ok", "score": 75}'
        output = GrammarOutput.model_validate_json(raw)
        assert len(output.issues) == 1
        assert output.score == 75.0


class TestCoherenceSchemas:
    def test_coherence_transition(self):
        t = CoherenceTransition(
            from_paragraph_text="End of paragraph one.",
            to_paragraph_text="Start of paragraph two.",
            issue="No logical connection.",
            suggestion="Add a transition sentence.",
            severity="weak",
        )
        assert t.severity == "weak"

    def test_coherence_output_defaults(self):
        output = CoherenceOutput()
        assert output.transitions == []
        assert output.score == 80.0


class TestArgumentationSchemas:
    def test_argument_component(self):
        c = ArgumentComponent(
            quoted_text="Studies show that...",
            component_type="evidence",
            strength="adequate",
        )
        assert c.reasoning == ""

    def test_argumentation_output_with_gaps(self):
        output = ArgumentationOutput(
            thesis_statement="Climate change is real.",
            logical_gaps=["No counterargument addressed."],
            score=65.0,
        )
        assert len(output.logical_gaps) == 1


class TestToneSchemas:
    def test_tone_issue(self):
        issue = ToneIssue(
            quoted_text="I think this is gonna work.",
            issue_type="colloquialism",
            formal_alternative="This approach is likely to succeed.",
        )
        assert issue.explanation == ""

    def test_tone_output_formality_options(self):
        output = ToneOutput(overall_formality="informal", score=40.0)
        assert output.overall_formality == "informal"


class TestReviewSchemas:
    def test_review_finding(self):
        f = ReviewFinding(
            finding="Weak introduction.",
            severity="moderate",
            target="introduction",
            suggestion="Strengthen the hook.",
        )
        assert f.quoted_text is None

    def test_review_output_with_strengths(self):
        output = ReviewOutput(
            strengths=["Clear methodology section."],
            score=82.0,
        )
        assert len(output.strengths) == 1


class TestSynthesisSchemas:
    def test_synthesis_action(self):
        a = SynthesisAction(
            severity="critical",
            target="methodology",
            category="argumentation",
            reasoning="No evidence provided for the main claim.",
        )
        assert a.suggestion == ""

    def test_synthesis_output(self):
        output = SynthesisOutput(
            reasoning="## Summary\nOverall decent paper.",
            prioritized_actions=[],
        )
        assert output.strengths == []


class TestCriticSchema:
    def test_critic_defaults(self):
        output = CriticLLMOutput()
        assert output.approved is True
        assert output.issues == []

    def test_critic_with_issues(self):
        output = CriticLLMOutput(
            issues=["Contradictory suggestions between grammar and tone workers."],
            approved=False,
        )
        assert not output.approved
