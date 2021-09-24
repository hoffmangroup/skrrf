from skrrf._forest import RegularizedRandomForestClassifier
import numpy as np

X = np.array([
    [0, 0, 0, 3, 3, 3] + [3] * 50,
    [0, 0, 4, 3, 3, 2] + [10] * 50,
    [0, 2, 4, 1, 3, 5] + [0] * 25 + [1] * 25,
]).transpose()

y = [0, 0, 0, 1, 1, 1] + [1] * 50

min_sample_count = np.bincount(y).min()

clf = RegularizedRandomForestClassifier(
    n_estimators=20,
    random_state=0,
    n_jobs=1,
    class_weight="balanced",
    max_samples=min_sample_count * 2)

clf.fit(X, y)
