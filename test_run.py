from _forest import RegularizedRandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

print(X.shape)

X = np.array([
    [0,0,0,3,3,3],
    [0,0,4,3,3,2],
    [0,2,4,1,3,5],
]).transpose()

y = [0,0,0,1,1,1]


print(X.shape)

clf = RegularizedRandomForestClassifier(max_depth=2, random_state=0, n_jobs=1)
clf.f = set([1])
clf.fit(X, y)

# print(clf.predict([[0, 0, 0, 0]]))
