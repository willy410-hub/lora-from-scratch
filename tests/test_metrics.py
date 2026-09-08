from lora.metrics import compute_bleu, compute_rouge


def test_compute_bleu_identical_text_scores_high():
    score = compute_bleu("the quick brown fox", "the quick brown fox")
    assert score > 0.9


def test_compute_bleu_unrelated_text_scores_low():
    score = compute_bleu("the quick brown fox", "completely different words entirely")
    assert score < 0.3


def test_compute_bleu_returns_float_in_valid_range():
    score = compute_bleu("hello world", "hello there world")
    assert 0.0 <= score <= 1.0


def test_compute_rouge_identical_text_scores_perfect():
    scores = compute_rouge("the quick brown fox", "the quick brown fox")
    assert scores["rougeL"].fmeasure == 1.0


def test_compute_rouge_returns_all_three_metrics():
    scores = compute_rouge("some reference text here", "some generated text here")
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores


def test_compute_rouge_unrelated_text_scores_low():
    scores = compute_rouge("the quick brown fox jumps", "zebra airplane calculator")
    assert scores["rougeL"].fmeasure < 0.2
