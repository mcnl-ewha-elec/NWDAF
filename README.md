# NWDAF
## 알고리즘 명칭(국문)
다음날 스마트팩토리 에너지 사용량을 예측하는 머신러닝 알고리즘

## 알고리즘 명칭(영문)
ML algorithm to predict next day's energy consumption of Smart Factory

## 발명자 리스트
심서영, 여현영, 홍수민, 차채연, 박형곤

## 알고리즘 전반적인 목적
본 알고리즘은 규칙성이 없는 에너지 사용량 데이터를 학습하고 다음날 에너지 사용량을 예측할 때, 그 성능을 높이기 위해 고안한 알고리즘이다. 데이터를 학습하는 방식은 총 두가지로 1)parallel 방식 과 2)series 방식이 있다. 본 알고리즘에서는 알고리즘이 예측한 사용량과 실제 사용량의 평균 제곱근 오차(RMSE)값으로 성능을 확인한다. EMA 저역통과필터를 사용해서 data의 noise를 줄여 성능을 높였다. parallel 방식은 머신러닝 모델을 여러개 만들어 성능이 좋은 모델에 가중치를 주는 방식으로 성능을 높였다.

## 알고리즘 실행 방법
1. https://data.lab.fiware.org/dataset/smart_energy_data-_aachen__cologne_smart_factory 에서

- aachensf3drobotwelter.csv
- aachensfbending.csv
- aachensflasercutting.csv
- aachensflasershaping.csv
총 4가지 파일을 다운받아서 data_preprocessing.py를 실행시켜 머신러닝 알고리즘 실행에 필요한 데이터를 만든다.

2. parallel.py 혹은 series.py를 실행한다

3. 해당 파일 안에 각각 Adaboost, DecisionTree, K-NN 총 3가지 머신러닝 모델이 있으므로 원하는 모델을 제외하고 주석처리한다.

4. 파일을 실행시키고 원하는 예측하고자 하는 날짜를 선택한다.

1주차 (5/18-5/24):5/25 예측, 2주차(5/25-5/31):6/1 예측, 3주차(6/1-6/7):6/8 예측, 4주차(6/8-6/14):6/15 예측



## 알고리즘 실행 결과에 대한 설명

<알고리즘이 예측한 사용량과 실제 사용량의 RMSE값을 머신러닝 모델별, 알고리즘 별로 비교한 표>

![telegram-cloud-photo-size-5-6098029360847105161-y](https://user-images.githubusercontent.com/88702793/128999072-eb354bcb-c2f1-4256-a573-72b0edfad6d4.jpg)

![photo_2021-08-11 16 49 22](https://user-images.githubusercontent.com/87114999/129133894-7136535a-e091-4199-bb06-a4c64cc610ea.jpeg)



@Decisoin Tree는 알고리즘 돌리는 중
