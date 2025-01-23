def find_best_performing_models(model_scores, n_best: int = 3, slice: bool = False):
    return sorted(model_scores, key=lambda x: x.get("score"), reverse=True)[:n_best]

