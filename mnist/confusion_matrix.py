"""
Generates the confusion matrix for the provided model using cleartext
evaluation.
"""
import setup

from core import models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.datasets import fetch_mldata

model = models.MLModel(source="objects/ml_models/final.mlm")

mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"].astype('int')
X_test_, y_test = X[60000:], y[60000:]
X_test = np.ones((10000, 785))
X_test[:, 1:] = X_test_

predictions = [np.array(model.evaluate(x)).argmax() for x in X_test]
M = [[0 for i in range(10)] for j in range(10)]

for i in range(10000):
    value = y_test[i]
    pred = predictions[i]
    M[value][pred] += 1


conf_mat = pd.DataFrame(
    M,
    index=[i for i in "0123456789"],
    columns=[i for i in "0123456789"],
)
plt.figure(figsize=(10, 10))
sn.heatmap(conf_mat, cmap="inferno_r", fmt='d', annot=True)
plt.savefig('confusion_matrix.png', format='png')
plt.show()
