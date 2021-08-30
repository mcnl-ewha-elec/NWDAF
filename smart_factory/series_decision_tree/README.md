## 프로그램 명칭(국문)
스마트 팩토리에서 의사 결정 나무를 이용한 에너지 사용량 예측 시뮬레이터

## 프로그램 명칭(영문)
Energy consumption prediction simulator using decision tree in smart factory

## 발명자 리스트
심서영, 여현영, 홍수민, 차채연, 박형곤

## 프로그램 전반적인 목적
본 프로그램은 규칙성이 없는 에너지 사용량 데이터를 학습하고 다음날 에너지 사용량을 예측할 때, 그 성능을 높이기 위해 고안되었다. 데이터를 학습하는 방식은 series 방식이다. 본 프로그램에서는 알고리즘이 예측한 사용량과 실제 사용량의 평균 제곱근 오차(RMSE)값으로 성능을 확인한다. EMA 저역통과필터를 사용해서 data의 noise를 줄여 성능을 높였다.

## 프로그램 실행 방법
1. https://data.lab.fiware.org/dataset/smart_energy_data-_aachen__cologne_smart_factory 에서

- aachensf3drobotwelter.csv
- aachensfbending.csv
- aachensflasercutting.csv
- aachensflasershaping.csv
총 4가지 파일을 다운받는다(첨부되어있는 파일에 전부 포함되어있으나 없다면 위의 주소를 통해 다운받는다).

2. series_decision_tree.py 파일을 실행시키고 원하는 예측하고자 하는 날짜를 선택한다.
-1주차 (5/18-5/24):5/25 예측, 2주차(5/25-5/31):6/1 예측, 3주차(6/1-6/7):6/8 예측, 4주차(6/8-6/14):6/15 예측

3. EMA 여부를 선택한다.


## 프로그램 실행 결과에 대한 설명

<프로그램 예측 결과의 RMSE값을 나타낸 표>

<img src="https://user-images.githubusercontent.com/87114999/130406889-72f54f0e-6040-4f97-bff6-32f9bcf990b5.png" width="60%" height="60%">
