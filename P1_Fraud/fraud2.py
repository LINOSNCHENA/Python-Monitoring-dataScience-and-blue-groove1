import numpy as np
import pandas as pd
eHouse="file:///Users/linos/Downloads/PyData/reportx.csv"
df = pd.read_csv(eHouse)

#Display the shape of the data set
print('Size of weather data frame is :',df.shape)
print(df[0:5])
print ("============================================== DATA LOADING ===========================")
print(df.count().sort_values())
#df = df.drop(columns=['hear','temperature','memory','protein'],axis=1)

print ("=========================================== DATA PREPOCESSING =========================")
print(df.shape)
print(df.count().sort_values())
#Removing null values
df = df.dropna(how='any')
print(df.shape)

print ("============================================= DATA OUTLIERS =========================")
from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df= df[(z < 3).all(axis=1)]
print(df.shape)
print ("============================================= DATA NORMALIZE =========================")
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
df.iloc[4:10]

#Using SelectKBest to get the top features!
from sklearn.feature_selection import SelectKBest, chi2
X = df.loc[:,df.columns!='smoking']
y = df[['smoking']]
selector = SelectKBest(chi2, k=3)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])
print ("============================================= DATA ENDING =========================")