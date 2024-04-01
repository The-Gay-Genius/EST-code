import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r'data.csv')
x = df.drop(columns='HCHO conversion')
y = df['HCHO conversion']

from sklearn.model_selection import KFold
n_splits = 10
KFold = KFold(n_splits, shuffle=False, random_state=None)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=114514)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

lasso=Lasso(alpha=0.1, max_iter=1000)
grid = {'alpha':[i/100.0 for i in range(0,25,1)]}
gsearch = GridSearchCV(lasso,param_grid=grid,cv=KFold,n_jobs=-1)
gsearch.fit(X_train,y_train)
print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)

lasso=Lasso(alpha=0.01, max_iter=1000)
lasso.fit(X_train,y_train)
y_test_pred = lasso.predict(X_test)
y_train_pred = lasso.predict(X_train)

from sklearn.metrics import r2_score,mean_squared_error
print('R2(traing)%.3f'%r2_score(y_train,y_train_pred))
print('R2MSE(traing)%.3f'%r2_score(y_test,y_test_pred))
print('MSE(traing)%.3f' % mean_squared_error(y_train,y_train_pred))
print('MSE(test)%.3f' % mean_squared_error(y_test,y_test_pred))