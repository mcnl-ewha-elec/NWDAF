# NWDAF
## 프로그램 명칭(국문)
다음날 스마트팩토리 에너지 사용량을 예측하는 최근접이웃 병렬 프로그램

## 알고리즘 명칭(영문)
knn parallel program to predict next day's energy consumption of Smart Factory

## 발명자 리스트
심서영, 여현영, 홍수민, 차채연, 박형곤

## 알고리즘 전반적인 목적
본 프로그램은 규칙성이 없는 에너지 사용량 데이터를 학습하고 다음날 에너지 사용량을 예측할 때, 그 성능을 높이기 위해 고안되었다. 데이터를 학습하는 방식은 parallel 방식으로 각각의 모델의 평균 제곱근 오차(RMSE)값을 기준으로 성능이 좋은 모델에 더 높은 가중치를 설정하여 하나의 모델을 만듦으로서 프로그램의 예측 성능을 높인다. 본 프로그램에서는 알고리즘이 예측한 사용량과 실제 사용량의 평균 제곱근 오차(RMSE)값으로 성능을 확인한다. EMA 저역통과필터를 사용해서 data의 noise를 줄여 성능을 높였다.

## 알고리즘 실행 방법
1. https://data.lab.fiware.org/dataset/smart_energy_data-_aachen__cologne_smart_factory 에서

- aachensf3drobotwelter.csv
- aachensfbending.csv
- aachensflasercutting.csv
- aachensflasershaping.csv
총 4가지 파일을 다운받아서 SW_preprocessing.py를 실행시켜 프로그램 실행에 필요한 데이터를 만든다.

2. SW_makemodel.py를 실행시켜 프로그램 실행에 필요한 knn 모델을 만든다.

3. 파일을 실행시키고 원하는 예측하고자 하는 날짜를 선택한다.
-1주차 (5/18-5/24):5/25 예측, 2주차(5/25-5/31):6/1 예측, 3주차(6/1-6/7):6/8 예측, 4주차(6/8-6/14):6/15 예측

4. EMA 여부를 선택한다.

*knn 모델의 개수에 따라 성능이 달라질 수 있으므로 목적에 맞추어 모델의 개수를 조정할 수 있다.

## 알고리즘 실행 결과에 대한 설명

<프로그램 예측 결과의 RMSE값을 나타낸 표>

![image](https://user-images.githubusercontent.com/88702793/130397932-fb5793a6-c118-4460-9830-b01ca7f56022.png)

<모델 개수에 따른 프로그램 예측 결과의 RMSE값을 나타낸 표>
![image](https://user-images.githubusercontent.com/88702793/130399265-8d7d14f3-7ebf-400f-a9be-e76203564ea6.png)

<직렬과 병렬 프로그램의 예측 결과를 비교한 표>
![image](https://user-images.githubusercontent.com/88702793/130401429-d64dc485-9561-4eb4-a4c9-6f1723a07136.png)





