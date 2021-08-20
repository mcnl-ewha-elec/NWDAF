## 프로그램 명칭(국문)

## 프로그램 명칭(영문)

## 발명자 리스트
심서영, 여현영, 홍수민, 차채연, 박형곤

## 프로그램 전반적인 목적

## 프로그램 실행 방법
1. https://data.lab.fiware.org/dataset/smart_energy_data-_aachen__cologne_smart_factory 에서

- aachensf3drobotwelter.csv
- aachensfbending.csv
- aachensflasercutting.csv
- aachensflasershaping.csv

    4가지 기계의 에너지 사용량 파일을 다운받는다.

2. parallel.py 혹은 series.py를 연다.

3. 해당 파일 안에 각각 Adaboost, K-NN, DecisionTree 총 3가지 머신러닝 모델이 있으므로 원하는 모델을 제외하고 주석처리한다.

4. 코드를 실행시키면 데이터 전처리 과정이 실행됨과 동시에 선택한 머신러닝 모델로 원하는 날짜의 에너지 사용량을 예측할 수 있다.

    1주차(5/18-5/24):5/25 예측, 2주차(5/25-5/31):6/1 예측, 3주차(6/1-6/7):6/8 예측, 4주차(6/8-6/14):6/15 예측



## 프로그램 실행 결과에 대한 설명
