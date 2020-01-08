import pandas as pd
eHouse="file:///Users/linos/Downloads/InstalledApps/deafness.csv"
df = pd.read_csv(eHouse)
x = df.iloc[:,:-1].astype(float).values
y = df.iloc[:,-1].values

# feature selection based on feature importance using Random Forest
from sklearn.ensemble import RandomForestClassifier
rd = RandomForestClassifier(n_estimators=100) 
from sklearn.feature_selection import SelectFromModel
# select the features with importance higher than the threshold
rd = SelectFromModel(rd, threshold = 0.10)
rd = rd.fit(x, y)
# selected features with a lower dimension
x_selected = rd.transform(x)

# split the data into training and test sets
from sklearn.model_selection import train_test_split
train1,test1,train2, test2, = train_test_split(x_selected, y, test_size=0.20, random_state=84)

# pipeline: Scaler-->Logistic Regression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipeline_classifier = Pipeline([('scl', StandardScaler()), 
('Pemba', LogisticRegression(solver='lbfgs')) ]) #==========

# tune the inverse regularization parameter C in logistic regression model
param_range = [0.021, 0.001, 0.01021, 0.01001]
param_grid=[{'Pemba__C': param_range} ]

# grid search with 4-fold cross validation   # =======================================================================
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator=pipeline_classifier, param_grid=param_grid, scoring='accuracy', cv=6)
gs.fit(train1, train2)

print("================= START ======================="),print("")

print("Trainging data Vs Testing Data")
# print(x)print(y)print(z)
print('The best accuracy: ', gs.best_score_)
print('The best parameters: ', gs.best_params_) ,print(" # ====== ********************* ===== #")
# the best configuration and train model
muntu = gs.best_estimator_
muntu.fit(train1, train2)

# Test the health of inmates in Smart house
print ('Test accuracy: ', muntu.score(test1, test2))
print(" X=========== 3* diagnosis ======X")
print(test2)
print ('Condition of Subject X1 accuracy: ', muntu.score(test1, test2))
print ('Condition of Subject x2 accuracy: ', muntu.score(test1, test2))
print ('Condition of Subject X3 accuracy: ', muntu.score(test1, test2)) ,print("")
print("================= SEND ==========================")