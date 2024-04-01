from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy

df = pd.read_csv(r'data.csv')
x = df.drop(columns='HCHO conversion')
y = df['HCHO conversion']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=114)

from sklearn.model_selection import KFold
n_splits = 10
KFold = KFold(n_splits, shuffle=False, random_state=None)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=114)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

param_grid = {
    'n_estimators': list(range(160,220,10)),
    'max_depth': list(range(14,20,1)),
    'min_samples_split': list(range(1,3,1)),
    'min_samples_leaf': list(range(1,4,1)),
    'max_features': ['log2', 'sqrt']
}
model = RandomForestRegressor()
gsearch = GridSearchCV(estimator = model,param_grid = param_grid,
                        scoring='neg_mean_squared_error',n_jobs=-1,cv=KFold)
gsearch.fit(X_train,y_train)
print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)

model = RandomForestRegressor(
    max_depth = 18,
    max_features= 'sqrt',
    min_samples_leaf= 1,
    min_samples_split= 2,
    n_estimators=180)
model.fit(X_train,y_train)

y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2(traing)%.3f'%r2_score(y_train,y_train_pred))
print('R2MSE(traing)%.3f'%r2_score(y_test,y_test_pred))
print('MSE(traing)%.3f' % mean_squared_error(y_train,y_train_pred))
print('MSE(test)%.3f' % mean_squared_error(y_test,y_test_pred))
