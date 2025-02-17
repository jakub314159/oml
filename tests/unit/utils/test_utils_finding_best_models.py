from oml.utils.utils_finding_best_models import utils_find_best_performing_models


def test_utils_finding_best_models():
    n_best = 2
    model_scores = [{"score": 12}, {"score": 3}, {"score": 325}, {"score": 24}]

    return_list = utils_find_best_performing_models(model_scores, n_best=n_best)

    assert len(return_list) == n_best
    assert return_list[0]["score"] == 325
