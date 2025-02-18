def utils_find_best_performing_models(model_scores, n_best: int = 3) -> list:
    """
    Sorts model_scores by the scores in the dict.

    Parameters
    ----------
    model_scores: list
        list of dicts that will be sorted.

    n_best: int
        number of instances that will be left in the list.

    Returns
    -------
    list: The sorted list
    """
    return sorted(model_scores, key=lambda x: x.get("score"), reverse=True)[:n_best]
