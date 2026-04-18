"""RevisionService — builds a revised draft from worker findings."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Dict, List

from app.models.context import ActionType, EvaluationContext
from app.utils.logger import logger

if TYPE_CHECKING:
    from app.llm.client import LLMClient


class RevisionService:
    """Build a revised draft from worker findings and synthesized actions."""

    @staticmethod
    async def revise_document(
        context: EvaluationContext,
        use_llm: bool = True,
        llm_client: LLMClient | None = None,
    ) -> Dict[str, Any]:
        original = context.document_content
        revised = context.document_content
        rewrite_summary: List[str] = []
        mode = "rules"
        llm_similarity = None

        grammar_result = context.worker_results.get("grammar_specialist")
        if grammar_result:
            revised = RevisionService._apply_grammar_revisions(revised, grammar_result.flagged_items)
            if grammar_result.flagged_items:
                rewrite_summary.append(
                    f"Applied grammar corrections to {len(grammar_result.flagged_items)} sentence(s)."
                )

        tone_result = context.worker_results.get("tone_specialist")
        if tone_result:
            revised = RevisionService._apply_tone_revisions(revised, tone_result.flagged_items)
            if tone_result.flagged_items:
                rewrite_summary.append("Applied tone normalization for informal expressions.")

        high_priority_actions = [
            action
            for action in context.final_actions
            if action.type in {ActionType.CRITICAL_REVISION, ActionType.MODERATE_REVISION}
        ]

        if use_llm and high_priority_actions:
            llm_result = await RevisionService._llm_rewrite(
                context=context,
                original_content=original,
                current_revision=revised,
                high_priority_actions=high_priority_actions,
                llm_client=llm_client,
            )
            if llm_result["accepted"]:
                revised = llm_result["content"]
                mode = "hybrid"
                llm_similarity = llm_result["similarity"]
                rewrite_summary.append(
                    f"LLM rewrite applied for {len(high_priority_actions)} critical/moderate action(s)."
                )
                rewrite_summary.append(llm_result["reason"])
            else:
                rewrite_summary.append("LLM rewrite skipped; rules-based revision retained.")

        similarity = round(SequenceMatcher(None, original, revised).ratio(), 4)
        if not rewrite_summary:
            rewrite_summary.append("No rewrite needed; document retained as-is.")

        tracked_changes, change_summary = RevisionService._build_tracked_changes(original, revised)

        return {
            "revised_content": revised,
            "revision_mode": mode,
            "rewrite_summary": rewrite_summary,
            "quality_metrics": {
                "lexical_similarity": similarity,
                "llm_similarity": llm_similarity,
                "high_priority_actions": len(high_priority_actions),
            },
            "tracked_changes": tracked_changes,
            "change_summary": change_summary,
        }

    @staticmethod
    def _build_tracked_changes(original: str, revised: str) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        before_units = RevisionService._split_sentences_with_spans(original)
        after_units = RevisionService._split_sentences_with_spans(revised)

        before_texts = [unit["text"] for unit in before_units]
        after_texts = [unit["text"] for unit in after_units]

        matcher = SequenceMatcher(None, before_texts, after_texts)
        tracked_changes: List[Dict[str, Any]] = []
        change_summary = {"unchanged": 0, "modified": 0, "added": 0, "removed": 0}

        change_id = 1
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for before_i, after_j in zip(range(i1, i2), range(j1, j2)):
                    before = before_units[before_i]
                    after = after_units[after_j]
                    tracked_changes.append(
                        {
                            "change_id": change_id,
                            "change_type": "unchanged",
                            "before_index": before_i,
                            "after_index": after_j,
                            "before_text": before["text"],
                            "after_text": after["text"],
                            "before_span": {"start": before["start"], "end": before["end"], "text": before["text"]},
                            "after_span": {"start": after["start"], "end": after["end"], "text": after["text"]},
                            "similarity": 1.0,
                        }
                    )
                    change_summary["unchanged"] += 1
                    change_id += 1

            elif tag == "replace":
                pair_count = min(i2 - i1, j2 - j1)
                for offset in range(pair_count):
                    before = before_units[i1 + offset]
                    after = after_units[j1 + offset]
                    tracked_changes.append(
                        {
                            "change_id": change_id,
                            "change_type": "modified",
                            "before_index": i1 + offset,
                            "after_index": j1 + offset,
                            "before_text": before["text"],
                            "after_text": after["text"],
                            "before_span": {"start": before["start"], "end": before["end"], "text": before["text"]},
                            "after_span": {"start": after["start"], "end": after["end"], "text": after["text"]},
                            "similarity": round(SequenceMatcher(None, before["text"], after["text"]).ratio(), 4),
                        }
                    )
                    change_summary["modified"] += 1
                    change_id += 1

                for before_i in range(i1 + pair_count, i2):
                    before = before_units[before_i]
                    tracked_changes.append(
                        {
                            "change_id": change_id,
                            "change_type": "removed",
                            "before_index": before_i,
                            "after_index": None,
                            "before_text": before["text"],
                            "after_text": None,
                            "before_span": {"start": before["start"], "end": before["end"], "text": before["text"]},
                            "after_span": None,
                            "similarity": 0.0,
                        }
                    )
                    change_summary["removed"] += 1
                    change_id += 1

                for after_j in range(j1 + pair_count, j2):
                    after = after_units[after_j]
                    tracked_changes.append(
                        {
                            "change_id": change_id,
                            "change_type": "added",
                            "before_index": None,
                            "after_index": after_j,
                            "before_text": None,
                            "after_text": after["text"],
                            "before_span": None,
                            "after_span": {"start": after["start"], "end": after["end"], "text": after["text"]},
                            "similarity": 0.0,
                        }
                    )
                    change_summary["added"] += 1
                    change_id += 1

            elif tag == "delete":
                for before_i in range(i1, i2):
                    before = before_units[before_i]
                    tracked_changes.append(
                        {
                            "change_id": change_id,
                            "change_type": "removed",
                            "before_index": before_i,
                            "after_index": None,
                            "before_text": before["text"],
                            "after_text": None,
                            "before_span": {"start": before["start"], "end": before["end"], "text": before["text"]},
                            "after_span": None,
                            "similarity": 0.0,
                        }
                    )
                    change_summary["removed"] += 1
                    change_id += 1

            elif tag == "insert":
                for after_j in range(j1, j2):
                    after = after_units[after_j]
                    tracked_changes.append(
                        {
                            "change_id": change_id,
                            "change_type": "added",
                            "before_index": None,
                            "after_index": after_j,
                            "before_text": None,
                            "after_text": after["text"],
                            "before_span": None,
                            "after_span": {"start": after["start"], "end": after["end"], "text": after["text"]},
                            "similarity": 0.0,
                        }
                    )
                    change_summary["added"] += 1
                    change_id += 1

        return tracked_changes, change_summary

    @staticmethod
    async def _llm_rewrite(
        context: EvaluationContext,
        original_content: str,
        current_revision: str,
        high_priority_actions: List[Any],
        llm_client: LLMClient | None = None,
    ) -> Dict[str, Any]:
        if not llm_client or not llm_client.available:
            return {
                "accepted": False,
                "content": current_revision,
                "reason": "No LLM client available for rewrite.",
                "similarity": round(SequenceMatcher(None, original_content, current_revision).ratio(), 4),
            }

        try:
            actions_text = "\n".join(
                f"- [{a.type.value}] {a.category}: {a.reasoning}. Suggestion: {a.suggestion or 'N/A'}"
                for a in high_priority_actions
            )
            section_text = "\n".join(
                f"- L{section.level}: {section.heading}"
                for section in context.document_sections[:12]
                if section.heading
            )
            research_worker = context.worker_results.get("research_specialist")
            research_text = ""
            if research_worker:
                snippets = []
                for item in research_worker.flagged_items[:3]:
                    query = item.get("query", "")
                    sources = item.get("sources", [])
                    if sources:
                        top = sources[0]
                        snippets.append(f"- {query}: {top.get('title', '')} {top.get('url', '')}".strip())
                if snippets:
                    research_text = "\n".join(snippets)

            system_prompt = (
                "You are an academic writing revision assistant. Rewrite the draft to address ONLY the listed "
                "critical/moderate issues. Preserve section headings, preserve meaning, keep citations and claims, "
                "and do not invent facts. Improve academic polish and clarity. Return only the revised document text."
            )
            content = (
                f"Section structure:\n{section_text or '- None'}\n\n"
                f"Research notes:\n{research_text or '- None'}\n\n"
                f"Issues:\n{actions_text}\n\n"
                f"Current draft:\n{current_revision}"
            )

            from app.core.config import settings

            candidate = await llm_client.generate(
                system_prompt=system_prompt,
                content=content,
                model_name=settings.REVISION_MODEL_NAME,
            )
            candidate = str(candidate).strip()

            if not candidate:
                return {
                    "accepted": False,
                    "content": current_revision,
                    "reason": "LLM returned empty output.",
                    "similarity": round(SequenceMatcher(None, original_content, current_revision).ratio(), 4),
                }

            similarity = round(SequenceMatcher(None, original_content, candidate).ratio(), 4)
            if similarity < 0.55:
                return {
                    "accepted": False,
                    "content": current_revision,
                    "reason": f"LLM rewrite rejected by similarity guardrail ({similarity}).",
                    "similarity": similarity,
                }

            return {
                "accepted": True,
                "content": candidate,
                "reason": "LLM rewrite passed guardrails.",
                "similarity": similarity,
            }
        except Exception as e:
            logger.warning("LLM rewrite failed: %s", e)
            return {
                "accepted": False,
                "content": current_revision,
                "reason": f"LLM rewrite failed: {e}",
                "similarity": round(SequenceMatcher(None, original_content, current_revision).ratio(), 4),
            }

    @staticmethod
    def _apply_grammar_revisions(document: str, flagged_items: List[Dict]) -> str:
        sentences = RevisionService._split_sentences(document)

        for item in flagged_items:
            sentence_id = item.get("sentence_id")
            corrected = item.get("corrected")
            if sentence_id is None or corrected is None:
                continue
            if 0 <= sentence_id < len(sentences):
                sentences[sentence_id] = corrected

        return " ".join(sentences)

    @staticmethod
    def _apply_tone_revisions(document: str, flagged_items: List[Dict]) -> str:
        revised = document

        replacements = {
            r"\bwanna\b": "want to",
            r"\bgonna\b": "going to",
            r"\bgotta\b": "have to",
            r"\bkinda\b": "somewhat",
            r"\bsorta\b": "somewhat",
            r"\blots of\b": "many",
            r"\ba lot of\b": "many",
            r"\bstuff\b": "material",
            r"\bdon't\b": "do not",
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bit's\b": "it is",
            r"\bthey're\b": "they are",
            r"\bwe're\b": "we are",
            r"\byou're\b": "you are",
        }

        if not flagged_items:
            return revised

        for pattern, replacement in replacements.items():
            revised = re.sub(pattern, replacement, revised, flags=re.IGNORECASE)

        return revised

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _split_sentences_with_spans(text: str) -> List[Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)", flags=re.MULTILINE)

        for match in pattern.finditer(text):
            raw = match.group(0)
            stripped = raw.strip()
            if not stripped:
                continue

            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())

            start = match.start() + leading
            end = match.end() - trailing

            units.append({"text": stripped, "start": start, "end": end})

        return units
