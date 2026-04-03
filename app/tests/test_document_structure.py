from app.services.document import DocumentExtractor


def test_parse_text_structure_detects_headings_and_subheadings():
    text = (
        "INTRODUCTION\n\n"
        "This is the opening paragraph.\n\n"
        "1.1 Problem Statement\n\n"
        "This is the subsection body.\n\n"
        "Conclusion\n\n"
        "This is the closing paragraph."
    )

    extracted = DocumentExtractor.parse_text_structure(text, filename="sample.txt")

    headings = [section.heading for section in extracted["sections"]]
    assert "Introduction" in headings
    assert "Problem Statement" in headings
    assert "Conclusion" in headings


def test_parse_text_structure_preserves_section_levels():
    text = (
        "# Background\n\n"
        "Context paragraph.\n\n"
        "## Literature Review\n\n"
        "Review paragraph."
    )

    extracted = DocumentExtractor.parse_text_structure(text, filename="sample.md")
    sections = {section.heading: section.level for section in extracted["sections"]}

    assert sections["Background"] == 1
    assert sections["Literature Review"] == 2
