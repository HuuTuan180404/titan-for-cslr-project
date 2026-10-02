from typing import Sequence

import torch


def edit_distance(prediction: Sequence[int], target: Sequence[int]) -> int:
    """
    Compute Levenshtein edit distance.

    The edit distance is the minimum number of:
        - substitutions
        - insertions
        - deletions

    required to transform prediction into target.

    Args:
        prediction: Predicted token sequence.
        target: Ground-truth token sequence.

    Returns:
        Edit distance.
    """

    n = len(prediction)
    m = len(target)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Transform prediction[:i] -> empty target
    for i in range(n + 1):
        dp[i][0] = i

    # Transform empty prediction -> target[:j]
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if prediction[i - 1] == target[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            substitution = dp[i - 1][j - 1] + substitution_cost

            insertion = dp[i][j - 1] + 1

            deletion = dp[i - 1][j] + 1

            dp[i][j] = min(substitution, insertion, deletion)

    return dp[n][m]


def ctc_greedy_decode(
    logits: torch.Tensor, input_lengths: torch.Tensor, blank_id: int
) -> list[list[int]]:
    """
    Greedy CTC decoding.

    Steps:
        1. Argmax over class dimension.
        2. Remove consecutive duplicate tokens.
        3. Remove blank tokens.

    Args:
        logits:
            CTC logits with shape [B, T, C].

        input_lengths:
            Valid sequence length for each sample.
            Shape [B].

        blank_id:
            Index of the CTC blank class.

    Returns:
        Decoded sequences.

        Example:
            [
                [12, 35, 18],
                [4, 7],
            ]
    """

    predictions = logits.argmax(dim=-1)

    decoded_sequences = []

    for i in range(predictions.size(0)):
        length = int(input_lengths[i].item())

        sequence = predictions[i, :length]

        # Remove consecutive duplicates.
        sequence = torch.unique_consecutive(sequence)

        # Remove CTC blank.
        sequence = sequence[sequence != blank_id]

        decoded_sequences.append(sequence.cpu().tolist())

    return decoded_sequences


def decode_targets(
    target_ids: torch.Tensor, target_lengths: torch.Tensor
) -> list[list[int]]:
    """
    Convert concatenated CTC targets into separate sequences.

    Example:

        target_ids:
            [1, 2, 3, 8, 9]

        target_lengths:
            [3, 2]

        returns:
            [
                [1, 2, 3],
                [8, 9],
            ]
    """

    targets = []

    start = 0

    for length in target_lengths:
        length = int(length.item())

        end = start + length

        targets.append(target_ids[start:end].cpu().tolist())

        start = end

    return targets


def compute_wer(predictions: list[list[int]], targets: list[list[int]]) -> float:
    """
    Compute corpus-level Word Error Rate.

    WER = total edit distance / total target tokens.

    Args:
        predictions:
            List of predicted sequences.

        targets:
            List of ground-truth sequences.

    Returns:
        WER as a ratio.

        Example:
            0.25 = 25%
    """

    if len(predictions) != len(targets):
        raise ValueError("Number of predictions and targets must match.")

    total_errors = 0
    total_words = 0

    for prediction, target in zip(predictions, targets):
        total_errors += edit_distance(prediction, target)

        total_words += len(target)

    if total_words == 0:
        return 0, 0

    return total_errors, total_words
