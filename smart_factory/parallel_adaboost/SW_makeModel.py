# Decision Tree, Adaboost, KNN

# EMA 여부 정할 수 있음.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


DTmodel_list=[]
ABmodel_list=[]
KNNmodel_list=[]

# Low Pass Filter (EMA 방식)
def LPF(f_cut,data):
    w_cut = 2 * np.pi * f_cut
    tau = 1 / w_cut
    ts = 1

    lpf_result = [data[0]]

    for i in range(1, len(data)):
        value = (tau * lpf_result[i - 1] + ts * data[i]) / (ts + tau)
        lpf_result.append(value)

    return(lpf_result)

# Making Decision Tree Model
def decisiontreeModel(x, y, modelnum,maxdepth,decisionsample,DTmodel_list):

    # decision tree model 만들기
    decision_tree_model = DecisionTreeRegressor(random_state=None, max_depth=maxdepth)
    bagging_model = BaggingRegressor(base_estimator=decision_tree_model, n_estimators=decisionsample, verbose=0).fit(x,y)
    print(str(modelnum) + '번째 Decision Tree model is made!')

    DTmodel_list.append(bagging_model)
    return DTmodel_list

# Making Adaboost Tree Model
def adaboostModel(x, y, modelnum,adaboostsample, learningrate,ABmodel_list):

    adaboost_model = make_pipeline(StandardScaler(), AdaBoostRegressor(
        base_estimator=GradientBoostingRegressor(min_samples_split=2, loss='ls', n_estimators=adaboostsample, learning_rate=learningrate,
                                                 random_state=None))).fit(x,y)
    print(str(modelnum) + '번째 Adaboost model is made!')

    ABmodel_list.append(adaboost_model)
    return ABmodel_list

# Making KNN Model

def KNNModel(k,x,y,modelnum,KNNmodel_list):
    KNN_model = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(x,y)
    print(str(modelnum) + '번째 KNN model is made!')
    KNNmodel_list.append(KNN_model)
    return KNNmodel_list


# implement models
def PredictandRMSE(modellst,test_x,goal_x,test_y,goal_y):
    RMSE_test_list = []
    RMSE_goal_list = []
    predict_list = []
    for model in modellst:
        test = model.predict(test_x)
        predict = model.predict(goal_x)
        RMSE_test = mean_squared_error(test_y, test, squared=False)
        RMSE_goal = mean_squared_error(goal_y, predict, squared=False)
        RMSE_test_list.append(RMSE_test)
        RMSE_goal_list.append(RMSE_goal)
        predict_list.append(predict)
    # print('test 날짜에 대한 RMSE값 (가중치 부여 기준): ', np.round(RMSE_test_list, 3))
    # print('실제로 예측하고자 하는 날짜에 대한 RMSE값: ', np.round(RMSE_goal_list, 3))
    # print()
    return RMSE_test_list, predict_list

# normalization and weight
def normalization(RMSE_test_list, predict_list, tot_model):

    multiplyScale = []
    RMSE_rev = []
    for num in range(0, tot_model):
        RMSE_rev.append(1/RMSE_test_list[num])
    for num in range(0, tot_model):
        multiplyScale.append(RMSE_rev[num] / sum(RMSE_rev))

    print('Real multiply value:', multiplyScale)

    # offering weights
    weighted_predict = 0
    for predict, weight in zip(predict_list, multiplyScale):
        weighted_predict += predict * weight

    return weighted_predict
