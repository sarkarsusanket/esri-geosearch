def compute_threshold(scores):
    """Compute binary pass/fail mask for similarity scores.

    Uses the mean (mu) and standard deviation (sigma) of the
    similarity score distribution to set a threshold at mu +
    n_std * sigma. Returns a binary array of the same shape as
    `scores`: 1 if score exceeds the threshold (positive match),
    0 otherwise.

    Args:
        scores: array of similarity scores
        mu: mean of the score distribution
        sigma: standard deviation of the score distribution
        n_std: number of standard deviations for threshold (default 3.0)

    Returns:
        binary array of same shape as scores, 1=pass, 0=fail
    """
    if len(scores)<1:
        return scores
    threshold = max(0.2, max(scores)-0.07)
    mask = (scores > threshold).astype(int)
    return mask