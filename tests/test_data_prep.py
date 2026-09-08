"""
Unit tests for lora/data_prep.py's extract_answer_text -- the fix for
SQuAD v2's unanswerable-question handling.
"""
from lora.data_prep import extract_answer_text


def test_extracts_first_answer_when_present():
    answer = {"text": ["Paris", "Paris, France"], "answer_start": [10, 10]}
    result = extract_answer_text(answer, unanswerable_target="No answer available.")
    assert result == "Paris"


def test_returns_unanswerable_target_for_empty_answers():
    """This is the bug fix: SQuAD v2 unanswerable questions have an
    empty text list, and must not silently become an empty-string
    training target."""
    answer = {"text": [], "answer_start": []}
    result = extract_answer_text(answer, unanswerable_target="No answer available.")
    assert result == "No answer available."


def test_never_returns_empty_string_for_unanswerable_case():
    answer = {"text": [], "answer_start": []}
    result = extract_answer_text(answer, unanswerable_target="No answer available.")
    assert result != ""


def test_unanswerable_target_is_configurable():
    answer = {"text": [], "answer_start": []}
    result = extract_answer_text(answer, unanswerable_target="UNANSWERABLE")
    assert result == "UNANSWERABLE"


def test_single_answer_case():
    answer = {"text": ["42"], "answer_start": [5]}
    result = extract_answer_text(answer, unanswerable_target="No answer available.")
    assert result == "42"
