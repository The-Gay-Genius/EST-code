import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble

df = pd.read_csv(r'data.csv')

x = df.drop(columns='HCHO conversion')
y = df['HCHO conversion']

from sklearn.model_selection import KFold
n_splits = 10
KFold = KFold(n_splits, shuffle=False, random_state=None)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=114)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)
X_train,X_test

params = {
    "n_estimators": 200,
    "max_depth": 2,
    "min_samples_split": 7,
    "learning_rate": 0.018,
}
model = ensemble.GradientBoostingRegressor(**params)
param_test_max_depth = {
 'max_depth':list(range(2,10,1))
}
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test_max_depth,
                        scoring='neg_mean_squared_error',n_jobs=4,cv=KFold)
gsearch1.fit(X_train,y_train)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

params = {
    "n_estimators": 200,
    "max_depth": 9,
    "min_samples_split": 7,
    "learning_rate": 0.018,
}
model = ensemble.GradientBoostingRegressor(**params)

param_test_max_depth = {
 'min_samples_split':list(range(2,10,1))
}
gsearch2 = GridSearchCV(estimator = model,param_grid = param_test_max_depth,
                        scoring='neg_mean_squared_error',n_jobs=4,cv=KFold)
gsearch2.fit(X_train,y_train)
print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)

params = {
    "n_estimators": 200,
    "max_depth": 9,
    "min_samples_split": 4,
    "learning_rate": 0.018,
}
model = ensemble.GradientBoostingRegressor(**params)
param_test_max_depth = {
 'learning_rate':[i/100.0 for i in range(0,15,1)]
}
gsearch3 = GridSearchCV(estimator = model,param_grid = param_test_max_depth,
                        scoring='neg_mean_squared_error',n_jobs=-1,cv=KFold)
gsearch3.fit(X_train,y_train)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)

params = {
    "n_estimators": 200,
    "max_depth": 9,
    "min_samples_split": 4,
    "learning_rate": 0.11,
}
model = ensemble.GradientBoostingRegressor(**params)
param_test_max_depth = {
  'n_estimators':list(range(100,400,20))
}
gsearch3 = GridSearchCV(estimator = model,param_grid = param_test_max_depth,
                        scoring='neg_mean_squared_error',n_jobs=4,cv=KFold)
gsearch3.fit(X_train,y_train)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)

params = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_split": 4,
    "learning_rate": 0.012,
}
model = ensemble.GradientBoostingRegressor(**params)
model.fit(X_train, y_train)
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse_tr_pls = mean_squared_error(y_train,y_train_pred)
mse_te_pls = mean_squared_error(y_test,y_test_pred)
print('MSE(traing)%.3f' % mse_tr_pls)
print('MSE(test)%.3f' % mse_te_pls)
print('R2(traing)%.3f'%r2_score(y_train,y_train_pred))
print('R2MSE(traing)%.3f'%r2_score(y_test,y_test_pred))

y_pred_train_gbr = model.predict(X_train)
y_pred_test_gbr = model.predict(X_test)
plt.figure(figsize = (4,4))
plt.scatter(y_train,y_pred_train_gbr,alpha = 0.5,color = 'blue',label = 'training')
plt.scatter(y_test,y_pred_test_gbr,alpha = 0.5,color = 'red',label = 'test')
plt.legend()
plt.xlabel('DFT')
plt.ylabel('predication')
plt.savefig('GDY.png',dpi=400)
plt.show()

