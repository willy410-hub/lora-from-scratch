import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_bleu(reference_text, hypothesis_text):
    """Calculates sentence-level BLEU score between reference and generated text."""
    ref_tokens = [reference_text.split()]
    hyp_tokens = hypothesis_text.split()
    smooth_fn = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth_fn)

def compute_rouge(reference_text, hypothesis_text):
    """Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference_text, hypothesis_text)

if __name__ == "__main__":
    sample_ref = "The quick brown fox jumps over the lazy dog"
    sample_hyp = "A quick brown fox jumped over a lazy dog"

    print("📊 Evaluation Metrics Module Ready!")
    print(f"Sample BLEU Score: {compute_bleu(sample_ref, sample_hyp):.4f}")
    
    scores = compute_rouge(sample_ref, sample_hyp)
    print(f"Sample ROUGE-L Score: {scores['rougeL'].fmeasure:.4f}")