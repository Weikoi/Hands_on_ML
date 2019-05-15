#%%%
from sklearn.decomposition import PCA
import pandas as pd

#%%
housing = pd.read_csv("D:\CS_Learning\Machine Learning\Hands_on_ML\datasets\housing\housing.csv")
housing.info()

#%%
housing.head()

#%%
housing.describe()

#%%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
drop_ocean = housing.drop("ocean_proximity", axis=1)
imp_housing = imputer.fit_transform(drop_ocean)
imp_housing = pd.DataFrame(imp_housing, columns=drop_ocean.columns)
imp_housing["index"] = pd.Series([i for i in range(len(imp_housing))])
imp_housing.set_index("index")
imputer.statistics_

#%%
imp_housing.head()
#%%
import featuretools as ft
es = ft.EntitySet()
es = es.entity_from_dataframe(dataframe=imp_housing, entity_id="drop_ocean", index="index")

#%%
pca = PCA(n_components=4)
X2D = pca.fit_transform(imp_housing)
X2D