# 1주일치 data > 그 다음주 월요일을 예측하는 직렬 모델
# Decision Tree, Adaboost, kNN
# train data: Mon~Fri(x) , Tue~Sat(y)
# test data: Wed~Sun(x), Thu~next Mon(y)
# 예측 목표: next Mon 예측하기

# EMA 여부 정할 수 있음.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import os
from datetime import datetime

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

###read files###
w = input('원하는 주차의 숫자를 입력하세요 : ')
df = pd.read_csv('data_'+str(w)+'w.csv')

def LPF(f_cut,data):
    w_cut = 2 * np.pi * f_cut
    tau = 1 / w_cut
    ts = 1

    lpf_result = [data[0]]

    for i in range(1, len(data)):
        value = (tau * lpf_result[i - 1] + ts * data[i]) / (ts + tau)
        lpf_result.append(value)

    return(lpf_result)

### x, y define ###
y = df['Energy consumption per timeslot [kWh]'].to_numpy()

# LPF 여부 선택
lpf = input("Low Pass Filtering (EMA) 하시겠습니까? (y/n):")

while lpf != 'y' and lpf != 'n':
    print('다시 입력하십시오.')
    lpf = input("Low Pass Filtering (EMA) 하시겠습니까? (y/n):")

if lpf == 'y':
    f_cut = input('cutting frequency를 입력하세요 (권장=0.02): ')  # 차단주파수 (Hz)
    f_cut = float(f_cut)
    y = np.array(LPF(f_cut, y))

elif lpf == 'n':
    print('Original Data를 사용하겠습니다.')

y_test = y[4320 : 11520]
y_test_cut = y_test[5760:7200]

x_train = y[:7200].reshape(-1, 1)
y_train = y[1440:8640]
x_test = y[2880:10080].reshape(-1, 1)
x_test_cut = x_test[5760:7200]

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
        maxdepth = input("Sampling number of Decision Tree (권장=400):")

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

repeatnum = input("알고리즘 반복 횟수를 입력하세요:")
repeatnum = int(repeatnum)

###Adaboost###

df2 = pd.DataFrame()
adaboosttrain_list=[]
adaboostRMSE_list=[]
for repeat in range(0,repeatnum):

    adaboost = make_pipeline(StandardScaler(), AdaBoostRegressor(base_estimator=GradientBoostingRegressor(min_samples_split=2, loss='ls', n_estimators=adaboostsample, learning_rate=learningrate, random_state=None))).fit(x_train, y_train)

    if repeat<repeatnum:
        predict_adaboost = adaboost.predict(x_test)
        df2['predict' + str(repeat + 1)] = predict_adaboost
        adaboosttrain_list.append(adaboost.score(x_train, y_train))
        adaboostRMSE_list.append(mean_squared_error(y_test_cut, predict_adaboost[5760:7200], squared=False))

    print(str(repeat + 1) + ' round Ended!')

# 평균 train score
adaboosttrain_list=np.array(adaboosttrain_list)
adaboosttrain=np.mean(adaboosttrain_list)

# 평균 RMSE
adaboostRMSE_list=np.array(adaboostRMSE_list)
adaboostRMSE=np.mean(adaboostRMSE_list)

# 평균 결과
df2['mean'] = df2.iloc[:, 0:repeatnum].mean(axis=1)

print('################################################')

###Decision Tree###
df3=pd.DataFrame()

DTRMSE_list=[]
DTtrain_list=[]

for repeat in range(0,repeatnum):
    decision_tree_model = DecisionTreeRegressor(random_state=None, max_depth=maxdepth)
    decisiontree = BaggingRegressor(base_estimator=decision_tree_model, n_estimators=decisionsample, verbose=0).fit(x_train,y_train)

    if repeat<repeatnum:
        predict_DT=decisiontree.predict(x_test)
        df3['predict' + str(repeat + 1)] = predict_DT
        DTtrain_list.append(decisiontree.score(x_train, y_train))
        DTRMSE_list.append(mean_squared_error(y_test_cut, predict_DT[5760:7200], squared=False))

    print(str(repeat + 1) + ' round Ended!')

# 평균 train score
DTtrain_list=np.array(DTtrain_list)
DTtrain=np.mean(DTtrain_list)

# 평균 RMSE
DTRMSE_list=np.array(DTRMSE_list)
DTRMSE=np.mean(DTRMSE_list)

# 평균 결과
df3['mean'] = df3.iloc[:, 0:repeatnum].mean(axis=1)

print('################################################')

###knn###
k=85 #k=square root of the total number of samples
knn = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(x_train, y_train)
predict_knn = knn.predict(x_test)
RMSE_knn = mean_squared_error(y_test_cut, predict_knn[5760:7200], squared=False)

print('################################################')

# 알고리즘별 train score과 RMSE 값

print(str(repeatnum)+'repeated-train score(adaboost):',adaboosttrain)
print(str(repeatnum)+'repeated-RMSE(adaboost) : ',adaboostRMSE)
print('Adaboost 직렬 모델의 RMSE 오차:', round(max(adaboostRMSE_list)-min(adaboostRMSE_list),3))
print()

print(str(repeatnum)+'repeated-train score(Decision Tree):', DTtrain)
print(str(repeatnum)+'repeated-RMSE(adaboost) : ',DTRMSE)
print('Decision Tree 직렬 모델의 RMSE 오차:', round(max(DTRMSE_list)-min(DTRMSE_list),3))
print()

print('train score(knn):',knn.score(x_train, y_train))
print('RMSE(knn) : ',RMSE_knn)
print()

###plot###
# test data: Thu~next Mon(y)

# Adaboost
plt.figure(figsize=(20, 10))
plt.plot(y_test, label='LPF real')
plt.plot(df2['mean'], label = 'adaboost')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
plt.xlabel("DateTime")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- Adaboost series', size=20)
plt.show()

# Decision Tree
plt.figure(figsize=(20, 10))
plt.plot(y_test, label='LPF real')
plt.plot(df3['mean'], label = 'decision tree')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
plt.xlabel("DateTime")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- Decision Tree series', size=20)
plt.show()

#kNN
plt.figure(figsize=(20, 10))
plt.plot(y_test, label='LPF real')
plt.plot(predict_knn, label = 'knn')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
plt.xlabel("DateTime")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- kNN series', size=20)
plt.show()

#Plot All
plt.figure(figsize=(20, 10))
plt.plot(df3['mean'], label = 'Decision Tree prediction')
plt.plot(df2['mean'], label = 'Adaboost prediction')
plt.plot(predict_knn, label='kNN prediction')
plt.plot(y_test, label='LPF real')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
plt.xlabel("index")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- ALL series algorithms ('+str(repeatnum)+' repeated)', size=20)
plt.show()

