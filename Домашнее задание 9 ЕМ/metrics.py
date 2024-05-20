from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    for sent in reference:
        for pair in sent.sure:
            if pair not in sent.possible:
                sent.possible.append(pair)
        

    A_intersect_P = sum([len([pair for pair in pred_sent if pair in ref_sent.possible]) for ref_sent, pred_sent in zip(reference, predicted)])
    A = sum([len(pred_sent) for pred_sent in predicted])

    return A_intersect_P, A


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    A_intersect_S = sum([len([pair for pair in pred_sent if pair in ref_sent.sure]) for ref_sent, pred_sent in zip(reference, predicted)])
    S = sum([len(ref_sent.sure) for ref_sent in reference])

    return A_intersect_S, S


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    A_intersect_P, A = compute_precision(reference, predicted)
    A_intersect_S, S = compute_recall(reference, predicted)

    return 1 - (A_intersect_P + A_intersect_S) / (A + S)
