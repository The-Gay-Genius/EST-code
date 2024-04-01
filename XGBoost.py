from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy


df = pd.read_csv(r'data.csv')
x = df.drop(columns='HCHO conversion')
y = df['HCHO conversion']
from sklearn.model_selection import KFold
n_splits = 10
KFold = KFold(n_splits, shuffle=False, random_state=None)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=114)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

xgb = XGBR(learning_rate =0.1,
                    n_estimators=150,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'reg:squarederror',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
param_test_max_depth_min_child_weight = {
 'max_depth':list(range(3,10,1)),
'min_child_weight':list(range(1,6,2))
}
gsearch1 = GridSearchCV(estimator = xgb,param_grid = param_test_max_depth_min_child_weight,
                        scoring='neg_mean_squared_error',n_jobs=4,cv=KFold)
gsearch1.fit(X_train,y_train)
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

xgb = XGBR(learning_rate =0.1,
                    n_estimators=150,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'reg:squarederror',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

param_test_gamma = {
 'gamma':[i/10.0 for i in range(0,1)]
}
gsearch2 = GridSearchCV(estimator = xgb, param_grid = param_test_gamma, scoring='neg_mean_squared_error',n_jobs=4, cv=KFold)
gsearch2.fit(X_train,y_train)
print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)

xgb = XGBR(learning_rate =0.1,
                    n_estimators=150,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'reg:squarederror',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

param_test_subsample_colsample_bytree = {
 'subsample':[i/100.0 for i in range(60,90,2)],
 'colsample_bytree':[i/100.0 for i in range(60,90,2)]
}
gsearch3 = GridSearchCV(estimator = xgb, param_grid = param_test_subsample_colsample_bytree,
                        scoring='neg_mean_squared_error',n_jobs=4, cv=KFold)
gsearch3.fit(X_train,y_train)
print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)

xgb = XGBR(learning_rate =0.1,
                    n_estimators=150,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.74,
                    colsample_bytree=0.86,
                    objective= 'reg:squarederror',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

param_test_learning_rate = {
 'learning_rate':[i/100.0 for i in range(0,15,1)]
}

gsearch4 = GridSearchCV(estimator = xgb, param_grid = param_test_learning_rate,
                        scoring='neg_mean_squared_error',n_jobs=4, cv=5)
gsearch4.fit(X_train,y_train)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)

xgb = XGBR(learning_rate =0.1,
                    n_estimators=150,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.74,
                    colsample_bytree=0.86,
                    objective= 'reg:squarederror',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

param_test_learning_rate = {
 'learning_rate':[i/100.0 for i in range(0,15,1)]
}

gsearch4 = GridSearchCV(estimator = xgb, param_grid = param_test_learning_rate,
                        scoring='neg_mean_squared_error',n_jobs=4, cv=5)
gsearch4.fit(X_train,y_train)
print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)

xgb = XGBR(learning_rate =0.11,
                    n_estimators=380,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.74,
                    colsample_bytree=0.86,
                    objective= 'reg:squarederror',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
xgb.fit(X_train,y_train)
y_train_pred=xgb.predict(X_train)
y_test_pred=xgb.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2(traing)%.3f'%r2_score(y_train,y_train_pred))
print('R2(traing)%.3f'%r2_score(y_test,y_test_pred))
print('MSE(traing)%.3f' % mean_squared_error(y_train,y_train_pred))
print('MSE(test)%.3f' % mean_squared_error(y_test,y_test_pred))