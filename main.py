import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsfresh as tsf


def main():
    # df = pd.read_csv('/home/jiananyuan/Desktop/DM_homework/dataset/train.csv')
    df = pd.read_csv(r'C:\Users\Jiananyuan\Desktop\QuickAccess\DM_homework\dataset\train.csv')
    # print(df)
    #           id                                  heartbeat_signals  label
    # 0          0  0.9912297987616655,0.9435330436439665,0.764677...    0.0
    # 1          1  0.9714822034884503,0.9289687459588268,0.572932...    0.0
    # 2          2  1.0,0.9591487564065292,0.7013782792997189,0.23...    2.0
    # 3          3  0.9757952826275774,0.9340884687738161,0.659636...    0.0
    # 4          4  0.0,0.055816398940721094,0.26129357194994196,0...    2.0
    # ...      ...                                                ...    ...
    # 99995  99995  1.0,0.677705342021188,0.22239242747868546,0.25...    0.0
    # 99996  99996  0.9268571578157265,0.9063471198026871,0.636993...    2.0
    # 99997  99997  0.9258351628306013,0.5873839035878395,0.633226...    3.0
    # 99998  99998  1.0,0.9947621698382489,0.8297017704865509,0.45...    2.0
    # 99999  99999  0.9259994004527861,0.916476635326053,0.4042900...    0.0
    #
    # [100000 rows x 3 columns]

    signals = df['heartbeat_signals'].str.split(',', expand=True)
    # print(signals)
    #                       0                     1    ...  203  204
    # 0      0.9912297987616655    0.9435330436439665  ...  0.0  0.0
    # 1      0.9714822034884503    0.9289687459588268  ...  0.0  0.0
    # 2                     1.0    0.9591487564065292  ...  0.0  0.0
    # 3      0.9757952826275774    0.9340884687738161  ...  0.0  0.0
    # 4                     0.0  0.055816398940721094  ...  0.0  0.0
    # ...                   ...                   ...  ...  ...  ...
    # 99995                 1.0     0.677705342021188  ...  0.0  0.0
    # 99996  0.9268571578157265    0.9063471198026871  ...  0.0  0.0
    # 99997  0.9258351628306013    0.5873839035878395  ...  0.0  0.0
    # 99998                 1.0    0.9947621698382489  ...  0.0  0.0
    # 99999  0.9259994004527861     0.916476635326053  ...  0.0  0.0
    #
    # [100000 rows x 205 columns]

    # last_one_signal = signals.iloc[-1:].values.tolist()[0]
    # plt.plot(last_one_signal)
    # plt.show()

    # print(df['label'].value_counts())
    # 0.0    64327
    # 3.0    17912
    # 2.0    14199
    # 1.0     3562
    # Name: label, dtype: int64

    signals.insert(0, '', df['label'], allow_duplicates=False)
    col_names = ['label']
    for i in range(0, 205):
        col_names.append('signal_' + str(i))
    signals.columns = col_names
    signals = pd.DataFrame(signals, dtype=np.float64)

    def signal_enhance(_signal, sigma=0.1):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, _signal.shape[1]))
        myNoise = np.matmul(np.ones((_signal.shape[0], 1)), scalingFactor)
        return _signal * myNoise

    # align to label: 0
    # process label: 1
    idx_set = signals.query('label==1').index
    record_label_1 = signals.iloc[idx_set, :].reset_index(drop=True)
    enhanced_signal_label_1 = record_label_1
    for i in np.arange(16):
        tmp_enhanced_signal_label_1 = signal_enhance(record_label_1)
        enhanced_signal_label_1 = pd.concat((enhanced_signal_label_1, tmp_enhanced_signal_label_1), axis=0).reset_index(
            drop=True)
    enhanced_signal_label_1['label'] = 1
    # print(enhanced_signal_label_1)

    # process label: 2
    idx_set = signals.query('label==2').index
    record_label_2 = signals.iloc[idx_set, :].reset_index(drop=True)
    enhanced_signal_label_2 = record_label_2
    for i in np.arange(3):
        tmp_enhanced_signal_label_2 = signal_enhance(record_label_2)
        enhanced_signal_label_2 = pd.concat((enhanced_signal_label_2, tmp_enhanced_signal_label_2), axis=0).reset_index(
            drop=True)
    enhanced_signal_label_2['label'] = 2
    # print(enhanced_signal_label_2)

    # process label: 3
    idx_set = signals.query('label==3').index
    record_label_3 = signals.iloc[idx_set, :].reset_index(drop=True)
    enhanced_signal_label_3 = record_label_3
    for i in np.arange(3):
        tmp_enhanced_signal_label_3 = signal_enhance(record_label_3)
        enhanced_signal_label_3 = pd.concat((enhanced_signal_label_3, tmp_enhanced_signal_label_3), axis=0).reset_index(
            drop=True)
    enhanced_signal_label_3['label'] = 3
    # print(enhanced_signal_label_3)

    # process label: 0
    idx_set = signals.query('label==0').index
    record_label_0 = signals.iloc[idx_set, :].reset_index(drop=True)
    # print(record_label_0)

    data_train = pd.concat([record_label_0, enhanced_signal_label_1, enhanced_signal_label_2, enhanced_signal_label_3], ignore_index=True)
    print(data_train)

    def remove_last_zero(_series):
        nps = _series.to_numpy()
        zero_begin_idxs = np.arange(nps.shape[0])
        for i in np.arange(nps.shape[1])[::-1]:
            idxs = np.where(nps[zero_begin_idxs, i] <= 1.e-5)[0]
            if idxs.size > 0:
                nps[zero_begin_idxs[idxs], i] = np.nan
                zero_begin_idxs = zero_begin_idxs[idxs]
            else:
                break
        return pd.DataFrame(nps[:, :], index=_series.index, columns=_series.columns[:])

    data_train = remove_last_zero(data_train)
    train_data = data_train.iloc[:, 1:].stack()
    train_data = train_data.reset_index()
    train_data.rename(columns={"level_0": "id", "level_1": "time", 0: "signals"}, inplace=True)
    train_data["signals"] = train_data["signals"].astype(float)
    print(train_data)

    fc_settings = tsf.feature_extraction.ComprehensiveFCParameters()
    feature_setup = {
        "length": None,
        "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}],
        "abs_energy": None,
        "absolute_sum_of_changes": None,
        "agg_linear_trend": [{"f_agg": 'mean', 'attr': 'pvalue', 'chunk_len': 2}]
    }

    train_label = data_train.iloc[:, 0]
    train_features = tsf.extract_relevant_features(train_data, train_label, column_id='id', column_sort='time',
                                                   default_fc_parameters=fc_settings)
    # train_features = tsf.extract_features(train_data, column_id='id', column_sort='time',
    #                                       default_fc_parameters=feature_setup)
    train_features.to_csv("feature.csv")
    train_label.to_csv("label.csv")


if __name__ == '__main__':
    main()
