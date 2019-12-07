import pandas as pd
eHouse="file:///Users/linos/Downloads/PyData/eHouse_datax.csv"
df = pd.read_csv(eHouse)
x = df.iloc[:,:-1].astype(float).values
y = df.iloc[:,-1].values

# feature selection based on feature importance using Random Forest
from sklearn.ensemble import RandomForestClassifier
rd = RandomForestClassifier(n_estimators=100) #======================================== Estimators
from sklearn.feature_selection import SelectFromModel
# select the features with importance higher than the threshold
rd = SelectFromModel(rd, threshold = 0.10)
rd = rd.fit(x, y)
# selected features with a lower dimension
x_selected = rd.transform(x)

# split the data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_selected, y, test_size=0.20, random_state=84)

# pipeline: Scaler-->Logistic Regression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipeline_classifier = Pipeline([('scl', StandardScaler()), ('Pemba', LogisticRegression(solver='lbfgs')) ]) #==========

# tune the inverse regularization parameter C in logistic regression model
param_range = [0.021, 0.001, 0.01021, 0.01001]
param_grid=[{'Pemba__C': param_range} ]

# grid search with 4-fold cross validation   # =======================================================================
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator=pipeline_classifier, param_grid=param_grid, scoring='accuracy', cv=6)
gs.fit(x_train, y_train)

print("================= SENIOR-CITIZENS CHECKING REPORT - START ======================="),print("")

print("Trainging data Vs Testing Data")
# print(x)print(y)print(z)
print('The best accuracy: ', gs.best_score_)
print('The best parameters: ', gs.best_params_) ,print(" # ====== ********************* ===== #")
# the best configuration and train model
best_model = gs.best_estimator_
best_model.fit(x_train, y_train)

# Test the health of inmates in Smart house
print ('Test accuracy: ', best_model.score(x_test, y_test))
print(" X======== Results from three of the subjects for medical diagnosis ======X")
print(y_test)
print ('Condition of Subject X1 accuracy: ', best_model.score(x_test, y_test))
print ('Condition of Subject x2 accuracy: ', best_model.score(x_test, y_test))
print ('Condition of Subject X3 accuracy: ', best_model.score(x_test, y_test)) ,print("")
print("================= SENIOR-CITIZENS CHECKING REPORT - END ==========================")