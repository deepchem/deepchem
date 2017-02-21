import sklearn
import sklearn.datasets
import numpy as np
#import openbabel as ob
from sklearn.linear_model import LogisticRegression

print("sklearn.__version__")
print(sklearn.__version__)
print("np.__version__")
print(np.__version__)

dataset = sklearn.datasets.load_digits(n_class=2)
X, y = dataset.data, dataset.target

sklearn_model = LogisticRegression()
sklearn_model.fit(X, y)
