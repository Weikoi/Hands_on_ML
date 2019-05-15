# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

pd.set_option('display.max_columns', 2000)
# %%
housing = pd.read_csv("D:\CS_Learning\Machine Learning\Hands_on_ML\datasets\housing\housing.csv")
housing.hist(bins=50, figsize=(20, 15))
plt.show()

#%%
# 浏览数据
housing.iloc[0]
#%%
housing.describe()
#%%
housing.head()
#%%
housing.info()
#%%
housing.hist(bins=30,figsize=(20, 15))
plt.show()
# %%
#
# 原始数据的收入数据被特征缩放到 5-15， 现在把他离散化成15个类， 然后把大于第5个类的所有类都集中到第5类中， 最终得到五个分类
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)  # 注意where的用法是把结果是false的替换
# %%
# 画直方图
housing.hist(bins=50, figsize=(20, 15))
plt.show()
print(housing.describe())
# %%
# 注意random_state的参数没有任何实际意义，只是一种随机状态，设置了以后，多次运行得到的随机数是一样的
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(housing["income_cat"].value_counts() / len(housing))
# %%

# 把income_cat 列删除
for demo_set in (strat_train_set, strat_test_set):
    demo_set.drop(["income_cat"], axis=1, inplace=True)
# %%

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
# %%

# 画出经纬度的热点图
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()
plt.show()
# %%

# 查看标准相关系数
corr_matrix = housing.corr()
corr_matrix

# %%
# 针对median_house_value的相关系数排序
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# %%
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.show()
# %%
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# %%
corr_matrix = housing.corr()
corr_matrix
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# %%
housing.dropna(subset=["total_bedrooms"])  # 选项1
housing.drop("total_bedrooms", axis=1)  # 选项2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)  # 选项3

# %%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

# %%
imputer.fit(housing_num)

# %%
imputer.statistics_

# %%
housing_num.median().values

# %%
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# %%
housing["ocean_proximity"]

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded

# %%
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
housing_cat_1hot

# %%
housing_cat_1hot.toarray()
