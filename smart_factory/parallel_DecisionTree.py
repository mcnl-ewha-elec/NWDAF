# 1주일치 data > 그 다음주 월요일을 예측하는 병렬 모델
# Decision Tree
# 성능 test data: 토 > 일 data

# EMA 여부 정할 수 있음.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
import SW_makeModel as Model
import SW_preprocessing as data

#make data files
data.preprocessing()

#read files
w = input('원하는 주차의 숫자를 입력하세요 (1-4) : ')
df = pd.read_csv('data_'+str(w)+'w.csv')
y = df['Energy consumption per timeslot [kWh]'].to_numpy()


energy7_real = y[8640 : 10080] # test_y without LPF
energy8_real = y[10080 : 11520] # goal_y without LPF

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

elif change == 'n':
    decisionsample=400
    maxdepth = None

# 각 알고리즘별 모델 만들기

DT_repeatedRMSE=[]
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


print('Decision Tree 병렬 모델의 RMSE(평균) : ', round(DT_RMSE_avg, 3))
print('Decision Tree 병렬 모델의 RMSE 오차:', round(max(DT_repeatedRMSE)-min(DT_repeatedRMSE),3))
print()

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

