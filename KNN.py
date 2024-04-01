import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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

from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors':np.arange(1,10),
        'p':np.arange(1,3),
        'weights':['uniform','distance']
       }
knn = KNeighborsRegressor()
gsearch = GridSearchCV(knn,param_grid = grid,cv=KFold,scoring='neg_mean_squared_error',n_jobs=-1)
gsearch.fit(X_train,y_train)
print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)

knn=KNeighborsRegressor(n_neighbors = 3,
        p = 1,
        weights = 'distance')
knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)
y_train_pred = knn.predict(X_train)

from sklearn.metrics import r2_score,mean_squared_error
print('R2(traing)%.3f'%r2_score(y_train,y_train_pred))
print('R2MSE(traing)%.3f'%r2_score(y_test,y_test_pred))
print('MSE(traing)%.3f' % mean_squared_error(y_train,y_train_pred))
print('MSE(test)%.3f' % mean_squared_error(y_test,y_test_pred))
