# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
X, y = make_moons(n_samples=1000, noise=0.1, random_state=30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
以下為硬投票方式
"""
# %%
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf),
                                          ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_train, y_train)

#%%
from sklearn.metrics import accuracy_score 
for clf in (log_clf, rnd_clf, svm_clf, voting_clf): 
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test) 
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""
以下為軟投票方式
"""
# %%
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf),
                                          ('svc', svm_clf)], voting='soft')
voting_clf.fit(X_train, y_train)

"""
分别fit各个单独分类器与集成分类器做对比
"""
#%%
from sklearn.metrics import accuracy_score 
for clf in (log_clf, rnd_clf, svm_clf, voting_clf): 
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test) 
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""
以下是Bagging集成方式,将oob_score设为True来计算oob_score()的值
"""
#%%
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,        
  max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True) 
bag_clf.fit(X_train, y_train) 
y_pred = bag_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(bag_clf.oob_score_)

#%%
"""
以下是Pasting集成方式, 注意Pasting方式不能测算oob_score的值
"""
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,        
  max_samples=100, bootstrap=False, n_jobs=-1) 
bag_clf.fit(X_train, y_train) 
y_pred = bag_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
# print(bag_clf.oob_score_)


#%%
"""
以下是随机森林方式
"""
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1) 
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

print("RandomForest:", accuracy_score(y_test, y_pred))


#%%
"""
特征评分
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris() 
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1) 
rnd_clf.fit(iris["data"], iris["target"]) 
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_): 
    print(name, score) 