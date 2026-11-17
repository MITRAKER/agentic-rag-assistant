from rag_assistant.answering import Answer, Citation, Segment, Source
from rag_assistant.vectorstore import SearchHit


def _sources(n: int) -> list[Source]:
    return [Source(f"text {i}", f"docs/doc{i}.md", f"Doc {i}", 0.5) for i in range(n)]


def test_render_numbers_sources_by_first_citation():
    answer = Answer(
        "q",
        [
            Segment("First claim.", [Citation(2, "x")]),
            Segment(" Glue text "),
            Segment("second claim.\n\n", [Citation(0, "y"), Citation(2, "x")]),
        ],
        _sources(3),
    )
    # source #2 was retrieved third but cited first, so it becomes [1]
    assert answer.cited_source_indices() == [2, 0]
    assert answer.render() == "First claim. [1] Glue text second claim. [1][2]"


def test_text_has_no_markers_and_coverage_counts_cited_chars():
    answer = Answer("q", [Segment("abcd", [Citation(0, "abcd")]), Segment("efgh")], _sources(1))
    assert answer.text == "abcdefgh"
    assert answer.coverage == 0.5


def test_empty_answer_has_zero_coverage():
    assert Answer("q", [], []).coverage == 0.0


def test_source_from_hit_uses_title_and_page():
    hit = SearchHit("t", "/x/report.pdf", 0.9, {"page": 4, "chunk_index": 0})
    src = Source.from_hit(hit)
    assert src.title == "report"  # no title metadata: falls back to the file name
    assert src.label == "report.pdf, page 4"
