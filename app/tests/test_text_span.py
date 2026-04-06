"""Tests for BaseWorker.find_span and find_span_fuzzy."""

from app.workers.base import BaseWorker


class TestFindSpan:
    def test_exact_match(self):
        document = "The quick brown fox jumps over the lazy dog."
        span = BaseWorker.find_span(document, "brown fox")
        assert span is not None
        assert span.start == 10
        assert span.end == 19
        assert span.text == "brown fox"

    def test_exact_match_at_start(self):
        document = "Hello world."
        span = BaseWorker.find_span(document, "Hello")
        assert span is not None
        assert span.start == 0
        assert span.end == 5

    def test_exact_match_not_found(self):
        document = "The quick brown fox."
        span = BaseWorker.find_span(document, "red fox")
        assert span is None

    def test_exact_match_empty_text(self):
        document = "Some text."
        span = BaseWorker.find_span(document, "")
        assert span is None

    def test_exact_match_none_text(self):
        span = BaseWorker.find_span("Some text.", None)
        assert span is None


class TestFindSpanFuzzy:
    def test_fuzzy_close_match(self):
        document = "The methodology section describes the experimental approach."
        span = BaseWorker.find_span_fuzzy(document, "methodology section describe the experimental")
        assert span is not None
        assert "methodology" in span.text

    def test_fuzzy_no_match_below_threshold(self):
        document = "The quick brown fox jumps over the lazy dog."
        span = BaseWorker.find_span_fuzzy(document, "completely unrelated text about nothing")
        assert span is None

    def test_fuzzy_exact_match_works_too(self):
        document = "Academic writing requires formal tone throughout."
        span = BaseWorker.find_span_fuzzy(document, "formal tone throughout")
        assert span is not None
        assert span.text == "formal tone throughout"

    def test_fuzzy_empty_text(self):
        document = "Some text."
        span = BaseWorker.find_span_fuzzy(document, "")
        assert span is None
