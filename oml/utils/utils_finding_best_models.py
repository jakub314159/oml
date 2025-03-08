def utils_find_best_performing_models(model_scores, n_best: int = 3, reverse: bool = False) -> list[dict]:
    """
    Sorts model_scores by the scores in the dict.

    Parameters
    ----------
    model_scores : list
        list of dicts that will be sorted.

    n_best : int
        number of instances that will be left in the list.

    reverse : bool
        The order in which the models will be shown

    Returns
    -------
    list[dict]
        The sorted list
    """
    if reverse:
        return sorted(model_scores, key=lambda x: x.get("score"))[:n_best]
    return sorted(model_scores, key=lambda x: x.get("score"), reverse=True)[:n_best]
