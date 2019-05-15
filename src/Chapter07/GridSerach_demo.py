from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np

# param_grid = [{"n_estimators": [i for i in range(200, 1100, 100)], "learning_rate": [i/10 for i in range(1, 11)]}]

#%%
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0, test_size=.2)

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3), algorithm="SAMME.R", n_estimators=1000, learning_rate=.7
)

param_grid = {"learning_rate": [i / 10 for i in range(1, 11)],
              "n_estimators": [i for i in range(200, 1100, 100)]}  # 转化为字典格式，网络搜索要求

grid_search = GridSearchCV(ada_clf, param_grid, cv=5, n_jobs=-1)
grid_result = grid_search.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
print("Test set score:{:.4f}" .format(grid_search.score(X_test, y_test)))


scores = cross_val_score(ada_clf, data.data, data.target, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# ada_clf.fit(X_train, y_train)
#
# y_predict = ada_clf.predict(X_test)
#
# print("Adaboost 的预测成功率为{:.4f}".format(accuracy_score(y_test, y_predict)))
