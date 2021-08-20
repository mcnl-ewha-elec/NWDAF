import pandas as pd
import os
import numpy as np

def preprocessing():
    ### Original Data -> 주차별 Data로 분리 ###
    machine_list = ['3drobotwelter', 'bending', 'lasercutting', 'lasershaping']
    for machine in machine_list:
        df = pd.read_csv('aachensf' + machine + '.csv')

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')

        for a in range(3, 8):
            for b in range(1, 32):
                df_new = df[(pd.DatetimeIndex(df.index).month == a) & (pd.DatetimeIndex(df.index).day == b)]
                if len(df_new) == 0:
                    continue
                elif len(df_new) != 1440:
                    continue
                else:
                    text = str(a) + str(b) + "_" + machine + ".csv"
                    df_new = df_new.rename(
                        columns={'Energy consumption per timeslot [MWh]': 'Energy consumption per timeslot [kWh]'})
                    y = df_new['Energy consumption per timeslot [kWh]'].to_numpy()
                    y = y * 1000
                    df_new['Energy consumption per timeslot [kWh]'] = y
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

                    # 날짜별 csv파일 만들기

                    file_name = str(a) + str(b) + '_allmachinesum.csv'

                    allmachinesum_data = {'Timestamp': timestamp, 'Energy consumption per timeslot [kWh]': y_new_sum}
                    allmachinesum_data = pd.DataFrame(allmachinesum_data)
                    allmachinesum_data.to_csv(file_name, sep=',', index=False)

    # 월요일 ~ 그 다음 월요일까지의 사용량을 합친 파일 만들기

    start_day = [518, 525, 61, 68]

    for week, day in enumerate(start_day, 1):
        df = pd.read_csv(str(day) + '_allmachinesum.csv')
        for i in range(1, 8):
            if day + i == 532:
                df_new = pd.read_csv(str(61) + '_allmachinesum.csv')
                df = pd.concat([df, df_new], ignore_index=True)
                continue
            if day + i == 70:
                date = 610
                while date < 616:
                    df_new = pd.read_csv(str(date) + '_allmachinesum.csv')
                    df = pd.concat([df, df_new], ignore_index=True)
                    date += 1
                break

            df_new = pd.read_csv(str(day + i) + '_allmachinesum.csv')
            df = pd.concat([df, df_new], ignore_index=True)

        df.to_csv('data_' + str(week) + 'w.csv', sep=',', index=False)
