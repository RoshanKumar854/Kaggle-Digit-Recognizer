import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv(r"C:\Users\rk141\Downloads\digit-recognizer\train.csv")
test_data = pd.read_csv(r"C:\Users\rk141\Downloads\digit-recognizer\test.csv")

X = train_data.iloc[:,1:]
y = train_data.iloc[:, :1]

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X_reduced, y)

y_pred = softmax_reg.predict(X_reduced)

from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)

pca_test = PCA(n_components=154)
X_test_reduced = pca_test.fit_transform(test_data)
Predictions = softmax_reg.predict(X_test_reduced)

