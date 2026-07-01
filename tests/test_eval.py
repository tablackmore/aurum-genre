import numpy as np
from aurum_genre.eval import macro_auc, calibrate_thresholds

def test_macro_auc_perfect_separation_is_one():
    y_true = np.array([[1,0],[0,1],[1,0],[0,1]])
    y_score = np.array([[0.9,0.1],[0.1,0.9],[0.8,0.2],[0.2,0.8]])
    assert macro_auc(y_true, y_score) == 1.0

def test_calibrate_picks_separating_threshold():
    y_true = np.array([[1],[1],[0],[0]])
    y_score = np.array([[0.8],[0.7],[0.2],[0.1]])
    th = calibrate_thresholds(y_true, y_score, ["x"])
    assert 0.2 < th["x"] <= 0.7
