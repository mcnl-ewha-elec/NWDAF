# 1주일치 data > 그 다음주 월요일을 예측하는 병렬 모델
# Decision Tree, Adaboost, KNN
# 성능 test data: 토 > 일 data

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
import os
import csv
import SW_makeModel as Model

### Original Data -> 주차별 Data로 분리 ###
machine_list=['3drobotwelter', 'bending', 'lasercutting', 'lasershaping']
for machine in machine_list:
    df=pd.read_csv('aachensf'+machine+'.csv')

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')

    for a in range (3,8):
        for b in range (1,32):
            df_new=df[(pd.DatetimeIndex(df.index).month == a)&(pd.DatetimeIndex(df.index).day == b)]
            if len(df_new)==0:
                continue
            elif len(df_new)!=1440:
                continue
            else:
                text = str(a) + str(b) + "_" + machine + ".csv"
                df_new = df_new.rename(columns={'Energy consumption per timeslot [MWh]': 'Energy consumption per timeslot [kWh]'})
                y = df_new['Energy consumption per timeslot [kWh]'].to_numpy()
                y = y * 1000
                df_new['Energy consumption per timeslot [kWh]']=y
                df_new.to_csv(text, mode='w')




# 해당 일자 동안의 각 시각당 에너지 사용량 합을 구한 csv file 만들기
for a in range(3, 8):
    for b in range(1, 32):

        # 날짜
        print(str(a) + '-' + str(b))

        file_name_b = str(a) + str(b) + '_bending.csv'
        file_name_c = str(a) + str(b) + '_lasercutting.csv'
        file_name_s = str(a) + str(b) + '_lasershaping.csv'
        file_name_r = str(a) + str(b) + '_3drobotwelter.csv'

        # 해당 일자의 각 machine에 대한 모든 파일이 존재할 때
        if (os.path.isfile(file_name_b) and os.path.isfile(file_name_c) and os.path.isfile(
                file_name_s) and os.path.isfile(file_name_r)):
            df_extracted_b = pd.read_csv(file_name_b)
            df_extracted_c = pd.read_csv(file_name_c)
            df_extracted_s = pd.read_csv(file_name_s)
            df_extracted_r = pd.read_csv(file_name_r)

            y_new_b = df_extracted_b['Energy consumption per timeslot [kWh]'].tolist()
            y_new_c = df_extracted_c['Energy consumption per timeslot [kWh]'].tolist()
            y_new_s = df_extracted_s['Energy consumption per timeslot [kWh]'].tolist()
            y_new_r = df_extracted_r['Energy consumption per timeslot [kWh]'].tolist()

            # data 개수가 맞지 않을 때 (즉, '모든' 시각의 data가 존재하지 않을 때)
            if (len(y_new_b) != 1440 or len(y_new_c) != 1440 or len(y_new_s) != 1440 or len(y_new_r) != 1440):
                # 해당 일자의 csv파일은 만들지 않는다.
                continue

            else:

                y_new_sum = []
                timestamp = df_extracted_b['Timestamp'].tolist()

                for i in range(0, 1440):
                    be = y_new_b[i]
                    c = y_new_c[i]
                    s = y_new_s[i]
                    r = y_new_r[i]
                    y_new_sum.append(be + c + s + r)

                print(y_new_sum)
                print(timestamp)

                print(len(y_new_sum))
                print(len(timestamp))

                # 2 날짜별 csv파일 만들기

                file_name = str(a) + str(b) + '_allmachinesum.csv'

                allmachinesum_data={'Timestamp':timestamp, 'Energy consumption per timeslot [kWh]': y_new_sum}
                allmachinesum_data=pd.DataFrame(allmachinesum_data)
                allmachinesum_data.to_csv(file_name,sep=',',index= False)



#월요일 ~ 그 다음 월요일까지의 사용량을 합친 파일 만들기

start_day = [518,525,61,68]


for week, day in enumerate(start_day,1):
    df = pd.read_csv(str(day) + '_allmachinesum.csv')
    for i in range(1,8):
        if day+i == 532:
            df_new = pd.read_csv(str(61) + '_allmachinesum.csv')
            df = pd.concat([df, df_new], ignore_index=True)
            continue
        if day+i == 70:
            date=610
            while date < 616:
                df_new = pd.read_csv(str(date) + '_allmachinesum.csv')
                df = pd.concat([df, df_new], ignore_index=True)
                date += 1
            break

        df_new = pd.read_csv(str(day + i) + '_allmachinesum.csv')
        df = pd.concat([df, df_new], ignore_index=True)

    df.to_csv('data_'+str(week)+'w.csv', sep=',', index=False)

#read files
w = input('원하는 주차의 숫자를 입력하세요 (1-4) : ')
df = pd.read_csv('data_'+str(w)+'w.csv')
y = df['Energy consumption per timeslot [kWh]'].to_numpy()

# next Mon(energy 8): Goal data
energy7_real = y[8640 : 10080] # NOT Low Pass Filtered
energy8_real = y[10080 : 11520] # NOT Low Pass Filtered

# 모든 data가 순서대로 들어갈 list
energylist=[] # original
in_energylist=[] # reshape


# LPF 여부 선택
lpf = input("Low Pass Filtering (EMA) 하시겠습니까? (y/n):")

while lpf != 'y' and lpf != 'n':
    print('다시 입력하십시오.')
    lpf = input("Low Pass Filtering (EMA) 하시겠습니까? (y/n):")

if lpf == 'y':
    f_cut = input('cutting frequency를 입력하세요 (권장=0.02): ')  # 차단주파수 (Hz)
    f_cut = float(f_cut)
    y = np.array(Model.LPF(f_cut, y))

elif lpf == 'n':
    print('Original Data를 사용하겠습니다.')

# Mon
energy1 = y[:1440]
in_energy1=energy1.reshape(-1,1)
energylist.append(energy1)
in_energylist.append(in_energy1)
# Tue
energy2 = y[1440:2880]
in_energy2=energy2.reshape(-1,1)
energylist.append(energy2)
in_energylist.append(in_energy2)
# Wed
energy3 = y[2880:4320]
in_energy3=energy3.reshape(-1,1)
energylist.append(energy3)
in_energylist.append(in_energy3)
# Thu
energy4 = y[4320:5760]
in_energy4=energy4.reshape(-1,1)
energylist.append(energy4)
in_energylist.append(in_energy4)
# Fri
energy5 = y[5760:7200]
in_energy5=energy5.reshape(-1,1)
energylist.append(energy5)
in_energylist.append(in_energy5)
# Sat
energy6 = y[7200 : 8640]
in_energy6=energy6.reshape(-1,1) # test_x
energylist.append(energy6)
in_energylist.append(in_energy6)
# Sun
energy7 = y[8640 : 10080] # test_y
in_energy7=energy7.reshape(-1,1) # goal_x
energylist.append(energy7)
in_energylist.append(in_energy7)

energy8 = y[10080 : 11520] #goal_y

test_x = in_energy6
test_y = energy7

goal_x = in_energy7
goal_y = energy8

tot_model=len(in_energylist)-2
print('model 개수 : ', tot_model)

# 각 알고리즘 model 만드는 함수
# model을 만들되 pickle로 따로 저장하지 않음.

# 알고리즘의 Parameter 값 바꾸기
change = input("Default 설정 값을 바꾸시겠습니까?(y/n):")
while ((change != 'y') and (change !='n')):
    print('다시 입력하십시오.')
    change = input("Default 설정 값을 바꾸시겠습니까?(y/n):")

if change == 'y':
    # decision tree sampling number 정하기
    decisionsample = input ("Sampling number of Decision Tree (권장=400):")
    decisionsample = int(decisionsample)
    while (decisionsample <= 0):
        print(" 0 이상 자연수를 입력하세요.")
        decisionsample = input("Sampling number of Decision Tree (권장=400):")

    # decision tree max depth 정하기
    maxdepth = input ("Max depth of Decision Tree (if 0, default=None):")
    maxdepth = int(maxdepth)
    while ((maxdepth < 1) and (maxdepth !=0)):
        print(" 1 이상 자연수를 입력하세요.")
        maxdepth = input("Max depth of Decision Tree (if 0, default=None):")
    if maxdepth == 0:
        maxdepth = None

    # adaboost sampling number 정하기
    adaboostsample = input("Sampling number of Adaboost (권장=400):")
    adaboostsample = int(adaboostsample)
    while (adaboostsample < 0) :
        print("0 이상 자연수를 입력하세요.")
        adaboostsample = input("Sampling number of Adaboost (권장=400):")

    # adaboost learning rate 정하기
    learningrate = input("Learning rate of Adaboost (0<learning rate<1) (권장=0.1):")
    learningrate = float (learningrate)
    while ((learningrate < 0) or (learningrate > 1)):
        print("0과 1 사이의 실수를 입력하세요.")
        learningrate = input("Learning rate of Adaboost (0<learning rate<1) (권장=0.1):")

elif change == 'n':
    decisionsample=400
    maxdepth = None
    adaboostsample=400
    learningrate=0.1

# 각 알고리즘별 모델 만들기

DT_repeatedRMSE=[]
AB_repeatedRMSE=[]
KNN_repeatedRMSE=[]
repeatnum = input("알고리즘 반복 횟수를 입력하세요:")
repeatnum = int(repeatnum)


# Decision Tree
for repeat in range(0,repeatnum):
    DTmodel_list=[]
    modelnum = 1
    for datanum in range(tot_model):
        DTmodel_list=Model.decisiontreeModel(in_energylist[datanum],energylist[datanum+1],modelnum,maxdepth,decisionsample,DTmodel_list)
        modelnum +=1

    DT_RMSEtest_list, DT_predict_list = Model.PredictandRMSE(DTmodel_list,test_x,goal_x,test_y,goal_y)
    DT_weighted = Model.normalization(DT_RMSEtest_list, DT_predict_list, tot_model)

    # final RMSE
    RMSE_final = mean_squared_error(goal_y, DT_weighted, squared=False)
    DT_repeatedRMSE.append(RMSE_final)
    print(str(repeat+1) + ' round Ended!')

DT_repeatedRMSE = np.array(DT_repeatedRMSE)
DT_RMSE_avg = np.mean(DT_repeatedRMSE)

# Adaboost
for repeat in range(0,repeatnum):
    ABmodel_list=[]
    modelnum = 1
    for datanum in range(tot_model):
        # Adaboost
        ABmodel_list=Model.adaboostModel(in_energylist[datanum],energylist[datanum+1],modelnum,adaboostsample,learningrate,ABmodel_list)
        modelnum += 1

    AB_RMSEtest_list, AB_predict_list = Model.PredictandRMSE(ABmodel_list,test_x,goal_x,test_y,goal_y)
    AB_weighted = Model.normalization(AB_RMSEtest_list, AB_predict_list, tot_model)

    # final RMSE
    RMSE_final = mean_squared_error(goal_y, AB_weighted, squared=False)
    AB_repeatedRMSE.append(RMSE_final)
    print(str(repeat+1) + ' round Ended!')


AB_repeatedRMSE = np.array(AB_repeatedRMSE)
AB_RMSE_avg = np.mean(AB_repeatedRMSE)

# KNN
# KNN은 sampling 과정이 없으므로 반복하지 않는다.
KNNmodel_list=[]
modelnum = 1
for datanum in range(tot_model):
    k = 38  #square root of the total number of samples
    KNNmodel_list=Model.KNNModel(k,in_energylist[datanum], energylist[datanum+1],modelnum,KNNmodel_list)
    modelnum += 1

KNN_RMSEtest_list, KNN_predict_list = Model.PredictandRMSE(KNNmodel_list,test_x,goal_x,test_y,goal_y)
KNN_weighted = Model.normalization(kNN_RMSEtest_list, KNN_predict_list, tot_model)
KNN_RMSE = mean_squared_error(goal_y, KNN_weighted, squared=False)

print('################################################')
print('Decision Tree 병렬 모델의 RMSE(평균) : ', round(DT_RMSE_avg, 3))
print('Decision Tree 병렬 모델의 RMSE 오차:', round(max(DT_repeatedRMSE)-min(DT_repeatedRMSE),3))
print()

print('Adaboost 병렬 모델의 RMSE (평균) : ', round(AB_RMSE_avg, 3))
print('Adaboost 병렬 모델의 RMSE 오차:', round(max(AB_repeatedRMSE)-min(AB_repeatedRMSE),3))
print()

print('KNN 병렬 모델의 RMSE : ', round(KNN_RMSE, 3))
print('################################################')

# plot_DecisionTree
plt.figure(figsize=(20, 10))
plt.plot(DT_weighted, label='prediction')
plt.plot(energy8_real, label='real')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
plt.xlabel("index")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- Decision Tree parallel', size=20)
plt.show()

# plot_Adaboost
plt.figure(figsize=(20, 10))
plt.plot(AB_weighted, label='prediction')
plt.plot(energy8_real, label='real')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
plt.xlabel("index")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- Adaboost parallel', size=20)
plt.show()

# plot_KNN
plt.figure(figsize=(20, 10))
plt.plot(KNN_weighted, label='prediction')
plt.plot(energy8_real, label='real')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
plt.xlabel("index")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- KNN parallel', size=20)
plt.show()


# Plot All
plt.figure(figsize=(20, 10))
plt.plot(DT_weighted, label = 'Decision Tree prediction')
plt.plot(AB_weighted, label = 'Adaboost prediction')
plt.plot(KNN_weighted, label='KNN prediction')
plt.plot(energy8_real, label='real')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
plt.xlabel("index")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- ALL parallel algorithms ('+str(repeatnum)+' repeated)', size=20)
plt.show()
