from backend.app.services import book_agent


def make_match(
    *,
    book_id: str,
    score: float,
    title: str,
    authors: list[str] | None = None,
    languages: list[str] | None = None,
    subjects: list[str] | None = None,
    bookshelves: list[str] | None = None,
    download_count: float = 0,
    first_publish_year: float | None = None,
    chunk_text: str = "",
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "title": title,
        "authors": authors or [],
        "languages": languages or [],
        "subjects": subjects or [],
        "bookshelves": bookshelves or [],
        "download_count": download_count,
        "chunk_text": chunk_text,
    }
    if first_publish_year is not None:
        metadata["first_publish_year"] = first_publish_year
    return {"id": book_id, "score": score, "metadata": metadata}


def test_candidate_from_match_handles_metadata_types() -> None:
    candidate = book_agent.candidate_from_match(
        make_match(
            book_id="book-1",
            score=0.42,
            title="Frankenstein",
            authors=["Shelley, Mary Wollstonecraft"],
            languages=["en"],
            download_count=1200,
            first_publish_year=1818.0,
        )
    )

    assert candidate is not None
    assert candidate.id == "book-1"
    assert candidate.title == "Frankenstein"
    assert candidate.authors == ("Shelley, Mary Wollstonecraft",)
    assert candidate.first_publish_year == 1818


def test_deduplicate_candidates_keeps_stronger_score() -> None:
    candidates = book_agent.candidates_from_matches(
        [
            make_match(book_id="low", score=0.2, title="The Odyssey", authors=["Homer"]),
            make_match(book_id="high", score=0.9, title="Odyssey", authors=["Homer"]),
        ]
    )

    deduped = book_agent.deduplicate_candidates(candidates)

    assert len(deduped) == 1
    assert deduped[0].id == "high"


def test_filter_candidates_by_relative_score_keeps_close_results() -> None:
    candidates = book_agent.candidates_from_matches(
        [
            make_match(book_id="a", score=1.0, title="A"),
            make_match(book_id="b", score=0.8, title="B"),
            make_match(book_id="c", score=0.5, title="C"),
            make_match(book_id="d", score=0.2, title="D"),
        ]
    )

    filtered = book_agent.filter_candidates_by_relative_score(
        candidates,
        threshold=0.75,
        min_keep=1,
    )

    assert [candidate.id for candidate in filtered] == ["a", "b"]


def test_rank_candidates_enforces_language_and_year_constraints() -> None:
    request = book_agent.ExtractedRequest(
        query="English philosophical novels before 1900",
        intent="recommendation",
        language_code="en",
        year=book_agent.YearConstraint(lte=1899),
        topics=("philosophical",),
    )
    candidates = book_agent.candidates_from_matches(
        [
            make_match(
                book_id="valid",
                score=0.8,
                title="Valid Book",
                languages=["en"],
                subjects=["Philosophy"],
                first_publish_year=1880,
            ),
            make_match(
                book_id="wrong-language",
                score=0.9,
                title="French Book",
                languages=["fr"],
                subjects=["Philosophy"],
                first_publish_year=1880,
            ),
            make_match(
                book_id="wrong-year",
                score=0.95,
                title="Modern Book",
                languages=["en"],
                subjects=["Philosophy"],
                first_publish_year=1910,
            ),
        ]
    )

    ranked = book_agent.rank_candidates(candidates, request)

    assert [item.candidate.id for item in ranked] == ["valid"]


def test_rank_candidates_boosts_topic_author_and_popularity() -> None:
    request = book_agent.ExtractedRequest(
        query="popular travel books by Mark Twain",
        intent="author_lookup",
        author="Mark Twain",
        topics=("travel",),
        wants_popular=True,
    )
    candidates = book_agent.candidates_from_matches(
        [
            make_match(
                book_id="twain",
                score=0.7,
                title="A Tramp Abroad",
                authors=["Twain, Mark"],
                subjects=["Travel"],
                download_count=5000,
                chunk_text="A humorous travel journey.",
            ),
            make_match(
                book_id="other",
                score=0.9,
                title="Unrelated",
                authors=["Other, Writer"],
                subjects=["Drama"],
                download_count=1,
            ),
        ]
    )

    ranked = book_agent.rank_candidates(candidates, request, enforce_hard_filters=False)

    assert ranked[0].candidate.id == "twain"
    assert "matches Mark Twain" in ranked[0].reason


def test_rank_candidates_matches_single_author_surname() -> None:
    request = book_agent.ExtractedRequest(
        query="Find books by Shakespeare",
        intent="author_lookup",
        author="Shakespeare",
    )
    candidates = book_agent.candidates_from_matches(
        [
            make_match(
                book_id="shakespeare",
                score=0.8,
                title="The Complete Works of William Shakespeare",
                authors=["Shakespeare, William"],
            )
        ]
    )

    ranked = book_agent.rank_candidates(candidates, request, enforce_hard_filters=False)

    assert ranked[0].candidate.id == "shakespeare"
    assert "matches Shakespeare" in ranked[0].reason


def test_rank_candidates_excludes_title_reference_itself() -> None:
    request = book_agent.ExtractedRequest(
        query="Something like Frankenstein",
        intent="title_reference",
        title_reference="Frankenstein",
        topics=("gothic",),
    )
    candidates = book_agent.candidates_from_matches(
        [
            make_match(
                book_id="same",
                score=0.99,
                title="Frankenstein",
                subjects=["Gothic fiction"],
            ),
            make_match(
                book_id="alternative",
                score=0.8,
                title="The Castle of Otranto",
                subjects=["Gothic fiction"],
            ),
        ]
    )

    ranked = book_agent.rank_candidates(candidates, request, enforce_hard_filters=False)

    assert [item.candidate.id for item in ranked] == ["alternative"]


def test_rank_candidates_excludes_reference_title_variants() -> None:
    request = book_agent.ExtractedRequest(
        query="Suggest books for someone who loved The Count of Monte Cristo",
        intent="title_reference",
        title_reference="The Count of Monte Cristo",
        topics=("revenge", "justice"),
    )
    candidates = book_agent.candidates_from_matches(
        [
            make_match(
                book_id="source",
                score=0.99,
                title="The Count of Monte Cristo",
                subjects=["Revenge"],
            ),
            make_match(
                book_id="source-volume",
                score=0.98,
                title="The Count of Monte Cristo, Volume 1",
                subjects=["Revenge"],
            ),
            make_match(
                book_id="alternative",
                score=0.8,
                title="The Son of Monte-Cristo",
                subjects=["Revenge"],
            ),
        ]
    )

    ranked = book_agent.rank_candidates(candidates, request, enforce_hard_filters=False)

    assert [item.candidate.id for item in ranked] == ["alternative"]


def test_select_recommendations_clamps_count() -> None:
    candidates = [
        book_agent.RankedCandidate(
            candidate=book_agent.BookCandidate(id=str(index), score=1, title=f"Book {index}"),
            rank_score=10 - index,
            reason="reason",
        )
        for index in range(7)
    ]

    assert len(book_agent.select_recommendations(candidates, requested_count=10)) == 5
    assert len(book_agent.select_recommendations(candidates, requested_count=0)) == 1


def test_format_display_helpers() -> None:
    assert book_agent.format_authors(()) == "Unknown author"
    assert book_agent.format_authors(("A", "B", "C")) == "A, B, et al."
    assert book_agent.format_year(1900) == "1900"
    assert book_agent.language_display("en") == "English"
