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


###knn###
k = round(len(x_train) ** 0.5) #k=square root of the total number of samples
knn = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(x_train, y_train)
predict_knn = knn.predict(x_test)
RMSE_knn = mean_squared_error(y_test_cut, predict_knn[5760:7200], squared=False)


print('train score(knn):',knn.score(x_train, y_train))
print('RMSE(knn) : ',RMSE_knn)
print()

###plot###
# test data: Thu~next Mon(y)

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
