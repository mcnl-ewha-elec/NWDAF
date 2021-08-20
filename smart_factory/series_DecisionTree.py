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

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

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


elif change == 'n':
    decisionsample=400
    maxdepth = None

repeatnum = input("알고리즘 반복 횟수를 입력하세요:")
repeatnum = int(repeatnum)


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



# train score과 RMSE 값

print(str(repeatnum)+'repeated-train score(Decision Tree):', DTtrain)
print('Decision Tree 직렬 모델의 RMSE 오차:', round(max(DTRMSE_list)-min(DTRMSE_list),3))
print()

###plot###
# test data: Thu~next Mon(y)

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



