# NWDAF
## 프로그램 명칭(국문)
다음날 스마트팩토리 에너지 사용량을 예측하는 에이다부스트 프로그램

## 프로그램 명칭(영문)
Adaboost program to predict next day's energy consumption of Smart Factory

## 발명자 리스트
심서영, 여현영, 홍수민, 차채연, 박형곤

## 프로그램 전반적인 목적
본 프로그램은 규칙성이 없는 에너지 사용량 데이터를 학습하고 다음날 에너지 사용량을 예측할 때, 그 성능을 높이기 위해 고안한 프로그램이다. 데이터를 학습하는 방식은 총 두가지로 1)parallel 방식 과 2)series 방식이 있다. 본 프로그램에서는 알고리즘이 예측한 사용량과 실제 사용량의 평균 제곱근 오차(RMSE)값으로 성능을 확인한다. EMA 저역통과필터를 사용해서 data의 noise를 줄여 성능을 높였다. parallel 방식은 에이다부스트 모델을 여러개 만들어 성능이 좋은 모델에 가중치를 주는 방식으로 성능을 높였다.

## 프로그램 실행 방법
1. https://data.lab.fiware.org/dataset/smart_energy_data-_aachen__cologne_smart_factory 에서

- aachensf3drobotwelter.csv
- aachensfbending.csv
- aachensflasercutting.csv
- aachensflasershaping.csv

    4가지 기계의 에너지 사용량 파일을 다운받는다.

2. parallel_Adaboost.py 혹은 series_Adaboost.py를 연다.

3. 코드를 실행시키면 데이터 전처리 과정이 실행됨과 동시에 adaboost 모델로 원하는 날짜의 에너지 사용량을 예측할 수 있다.

    - 1주차(5/18-5/24):5/25 예측, 2주차(5/25-5/31):6/1 예측, 3주차(6/1-6/7):6/8 예측, 4주차(6/8-6/14):6/15 예측
    - prameter값과 ,EMA 선택 여부에 따라 값이 달라짐



## 프로그램 실행 결과에 대한 설명

<프로그램이 예측한 사용량과 실제 사용량의 RMSE값을 나타낸 표>

![photo_2021-08-11 16 48 43](https://user-images.githubusercontent.com/87114999/129167496-d14ccb58-f277-41f2-bab7-a3adea12ca2b.jpeg)


