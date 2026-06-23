"""Tests for heading-aware chunking and contextual-header embedding.

These cover the fix that keeps each exercise's sub-section chunks
(Overview/Setup/Coaching Cues/...) attached to their parent
"Exercise N: <name>" heading, so they don't get orphaned from the
exercise name during retrieval.
"""

from build_knowledge_base import chunk_text, embedding_text, is_heading


def test_is_heading_allows_titlecase_connectors():
    # Title case lowercases short connector words ("to", "of"); these must
    # still register as headings so e.g. "Sit to Stand" isn't read as body.
    assert is_heading("Exercise 5: Sit to Stand")
    assert is_heading("Putting It Together: Session Design")
    assert is_heading("Common Errors and Corrections")
    # Real sentences (end punctuation / long) stay non-headings.
    assert not is_heading("Sit at the front edge of a chair and stand up.")


def _titles(text, start_parent=""):
    chunks, parent = chunk_text(text, start_parent=start_parent)
    return [st for _, st in chunks], parent


def test_subsection_inherits_exercise_parent():
    text = (
        "Exercise 2: Hurdle Step\n\n"
        "Overview\n\n"
        "The Hurdle Step trains single-leg stance stability.\n\n"
        "Coaching Cues\n\n"
        "Lift one knee up toward your chest, keeping your foot dorsiflexed.\n\n"
        "Common Errors and Corrections\n\n"
        "Hip drop on the standing leg means weak glute medius.\n\n"
    )
    titles, _ = _titles(text)
    # Every sub-section keeps the exercise name attached.
    assert any(t == "Exercise 2: Hurdle Step" for t in titles)
    assert "Exercise 2: Hurdle Step — Overview" in titles
    assert "Exercise 2: Hurdle Step — Coaching Cues" in titles
    assert "Exercise 2: Hurdle Step — Common Errors and Corrections" in titles


def test_new_chapter_clears_exercise_parent():
    text = (
        "Exercise 9: Shoulder Extension\n\n"
        "Rep Guidance\n\n"
        "Do 10 to 15 reps.\n\n"
        "Putting It Together: Session Design\n\n"
        "FREQUENCY\n\n"
        "Three sessions per week with a rest day between each.\n\n"
    )
    titles, parent = _titles(text)
    assert "Exercise 9: Shoulder Extension — Rep Guidance" in titles
    # The session-design content must NOT be tagged with the exercise.
    assert not any("Shoulder Extension" in t and "FREQUENCY" in t for t in titles)
    assert "FREQUENCY" in titles
    assert parent == ""  # exercise context ended


def test_parent_threads_across_pages():
    # Page 1 ends inside an exercise; page 2 starts with a bare sub-section.
    page1 = "Exercise 4: Inline Lunge\n\nOverview\n\nA split-stance balance drill.\n\n"
    page2 = "Rep Guidance\n\nDo 8 to 12 reps per leg.\n\n"
    _, parent1 = _titles(page1)
    assert parent1 == "Exercise 4: Inline Lunge"
    titles2, _ = _titles(page2, start_parent=parent1)
    assert "Exercise 4: Inline Lunge — Rep Guidance" in titles2


def test_embedding_text_prepends_section_title():
    body = "Lift one knee up toward your chest."
    out = embedding_text(body, "Exercise 2: Hurdle Step — Coaching Cues")
    assert out.startswith("Exercise 2: Hurdle Step — Coaching Cues")
    assert body in out
    # No section title → unchanged body.
    assert embedding_text(body, "") == body
