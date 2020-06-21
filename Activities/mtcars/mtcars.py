# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import libraries
from sagemaker import get_execution_role                            
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# %%
role = get_execution_role()
awsBucketName = ''
fileName = 'mtcars.csv'
bucketPath = f"s3://{awsBucketName}/{fileName}"
mtcars = pd.read_csv(bucketPath)

# %%
mtcars.dtypes.value_counts()

# %%
mtcars.select_dtypes('object').apply(pd.Series.nunique, axis = 0) # Get number of unique categorical values for each object column

# %%
mtcars = mtcars.drop(columns = ["carName"])

# %%
mtcars

# %%
#Encode transmissionType
le = LabelEncoder()
le.fit(mtcars["transmissionType"])
mtcars["transmissionType"] = le.transform(mtcars["transmissionType"])

# %%
mtcars["transmissionType"].astype(int).plot.hist();

# %%
mtcars.shape

# %%
mtcars = pd.get_dummies(mtcars)

# %%
mtcars.shape

# %%
mtcars

# %%
mtcars.isnull().sum() #No missing values

# %%
mtcars.corr()['transmissionType'].sort_values() #check correlation with respect to our target column. Lets predict transmission type.

# %%
mtcars["weight"].astype(int).plot.hist();

# %%
mtcars["numberOfGears"].astype(int).plot.hist();

# %%

