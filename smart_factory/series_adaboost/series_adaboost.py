# 1주일치 data > 그 다음주 월요일을 예측하는 직렬 모델
# Adaboost
# train data: Mon~Fri(x) , Tue~Sat(y)
# test data: Wed~Sun(x), Thu~next Mon(y)
# 예측 목표: next Mon 예측하기

# EMA 여부 정할 수 있음.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error


from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import SW_preprocessing as data

#make files
data.preprocessing()

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


# 알고리즘별 train score과 RMSE 값

print(str(repeatnum)+'repeated-train score(adaboost):',adaboosttrain)
print(str(repeatnum)+'repeated-RMSE(adaboost) : ',adaboostRMSE)
print('Adaboost 직렬 모델의 RMSE 오차:', round(max(adaboostRMSE_list)-min(adaboostRMSE_list),3))
print()
print(str(repeatnum)+'repeated-RMSE(adaboost) : ',DTRMSE)

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

