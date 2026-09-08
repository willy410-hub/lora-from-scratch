"""
Pure data-preprocessing helpers shared by the training scripts.

Kept free of any TensorFlow/datasets/transformers imports so the
logic itself (not the framework plumbing around it) is directly
unit-testable.
"""


def extract_answer_text(answer: dict, unanswerable_target: str) -> str:
    """
    Extract the training target for one SQuAD v2 example.

    SQuAD v2's defining feature is that roughly half its questions are
    unanswerable from the given context, represented as an empty
    `answer["text"]` list. Returning an empty string for those examples
    (the original behavior) trains the model to output nothing rather
    than to state that a question is unanswerable. This returns
    `unanswerable_target` instead, an explicit, learnable refusal.
    """
    if answer["text"]:
        return answer["text"][0]
    return unanswerable_target
