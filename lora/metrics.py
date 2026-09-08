"""Sequence-generation evaluation metrics: BLEU and ROUGE."""
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def compute_bleu(reference_text: str, hypothesis_text: str) -> float:
    """Calculate sentence-level BLEU score between reference and generated text."""
    ref_tokens = [reference_text.split()]
    hyp_tokens = hypothesis_text.split()
    smooth_fn = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth_fn)


def compute_rouge(reference_text: str, hypothesis_text: str) -> dict:
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference_text, hypothesis_text)
