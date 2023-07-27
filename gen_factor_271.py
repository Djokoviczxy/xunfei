#%%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time, math
import os
import warnings
warnings.filterwarnings("ignore")
#%%
def cal_imbalance(data):
    # 3
    ret = np.zeros((data.shape[0], 3))
    col1 = [f'n_bid{i}' for i in range(1, 6)]
    col2 = [f'n_bsize{i}' for i in range(1, 6)]
    col3 = [f'n_ask{i}' for i in range(1, 6)]
    col4 = [f'n_asize{i}' for i in range(1, 6)]
    time = data['time'].values

    bid_price = data[col1].values
    bid_volume = data[col2].values
    ask_price = data[col3].values
    ask_volume = data[col4].values

    weight = [1 - (i - 1) / 5 for i in range(1, 6)]
    weight = np.tile(weight, (bid_price.shape[0], 1))

    bidsum_np = (bid_price * bid_volume * weight).sum(axis=1)
    asksum_np = (ask_price * ask_volume * weight).sum(axis=1)
    # 计算spread_tick
    imbalance = (bidsum_np - asksum_np) / (bidsum_np + asksum_np)
    ret[:, 0] = bidsum_np
    ret[:, 1] = asksum_np
    ret[:, 2] = imbalance
    return ret, time

def cal_spread(data):
    # 10
    col1 = [f'n_bid{i}' for i in range(1, 6)]
    col2 = [f'n_bsize{i}' for i in range(1, 6)]
    col3 = [f'n_ask{i}' for i in range(1, 6)]
    col4 = [f'n_asize{i}' for i in range(1, 6)]

    bid_price = data[col1].values
    bid_volume = data[col2].values
    ask_price = data[col3].values
    ask_volume = data[col4].values

    price_spread = bid_price - ask_price
    price_sum_mid = (ask_price + bid_price) / 2
    ret = np.concatenate([price_spread, price_sum_mid], axis=1)
    return ret

def cal_SOIR_PIR(data):
    # 12
    col1 = [f'n_bid{i}' for i in range(1, 6)]
    col2 = [f'n_bsize{i}' for i in range(1, 6)]
    col3 = [f'n_ask{i}' for i in range(1, 6)]
    col4 = [f'n_asize{i}' for i in range(1, 6)]

    bid_price = data[col1].values
    bid_volume = data[col2].values
    ask_price = data[col3].values
    ask_volume = data[col4].values

    weight = [1 - (i - 1) / 5 for i in range(1, 6)]
    weight = np.tile(weight, (bid_volume.shape[0], 1))

    soir = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    weight_soir = (soir * weight).sum(axis = 1) / weight.sum(axis = 1)
    weight_soir = weight_soir.reshape((-1, 1))

    pir = (bid_price - ask_price) / (bid_price + ask_price)
    weight_pir = (pir * weight).sum(axis = 1) / weight.sum(axis = 1)
    weight_pir = weight_pir.reshape((-1, 1))
    ret = np.concatenate([soir, weight_soir, pir, weight_pir], axis = 1)
    return ret

def cal_MCI(data):
    # 3
    m = data['n_midprice'].replace(0, np.nan).values
    col1 = [f'n_bid{i}' for i in range(1, 6)]
    col2 = [f'n_bsize{i}' for i in range(1, 6)]
    col3 = [f'n_ask{i}' for i in range(1, 6)]
    col4 = [f'n_asize{i}' for i in range(1, 6)]

    bid_price = data[col1].replace(0, np.nan).values
    bid_volume = data[col2].replace(0, np.nan).values
    ask_price = data[col3].replace(0, np.nan).values
    ask_volume = data[col4].replace(0, np.nan).values

    dolvalB = (bid_price * bid_volume).sum(axis = 1)
    dolvalA  = (ask_price * ask_volume).sum(axis = 1)
    volAsum = ask_volume.sum(axis = 1)
    volBsum = bid_volume.sum(axis = 1)
    vwapa = dolvalA / volAsum
    vwapa = (vwapa - m) / m
    mcia = (vwapa / dolvalA)[:, np.newaxis]

    vwapb = dolvalB / volBsum
    vwapb = (vwapb -m) / m
    mcib = (vwapb / dolvalB)[:, np.newaxis]

    mciimb = ((mcib - mcia) / (mcib + mcia))
    ret = np.concatenate([mcia, mcib, mciimb], axis=1)
    return ret

def cal_slope(data, i):
    '''
    :param data: snap数
    :param i: 对应的挡位
    '''

    log_price_bid = np.log(data[f'n_bid{i}'].replace(0, np.nan))
    log_price_ask = np.log(data[f'n_ask{i}'].replace(0, np.nan))
    log_vol_bid = np.log(data[f'n_bsize{i}'].replace(0, np.nan))
    log_vol_ask = np.log(data[f'n_asize{i}'].replace(0, np.nan))

    select = ((log_vol_bid + log_vol_ask) != 0)
    log_price_bid = log_price_bid[select]
    log_price_ask = log_price_ask[select]
    log_vol_bid = log_vol_bid[select]
    log_vol_ask = log_vol_ask[select]

    slope = (log_price_ask - log_price_bid) / (log_vol_ask + log_vol_bid)
    return slope


def cal_ofi(data, i):
    '''
    :param data: snap数
    :param i: 对应的挡位
    '''
    temp_buy = np.zeros(data.shape[0])
    x_hat = data[f'n_bid{i}']
    x = data[f'n_bid{i}'].shift(1)
    flagb1 = (x_hat < x).values
    flagb2 = (x_hat > x).values
    flagb3 = (x_hat == x).values

    volb1 = -data[f'n_bsize{i}'].shift(1)
    volb2 = data[f'n_bsize{i}']
    volb3 = data[f'n_bsize{i}'] - data[f'n_bsize{i}'].shift(1)

    temp_buy[flagb1] = volb1.values[flagb1]
    temp_buy[flagb2] = volb2.values[flagb2]
    temp_buy[flagb3] = volb3.values[flagb3]

    y_hat = data[f'n_ask{i}']
    y = data[f'n_ask{i}'].shift(1)
    flaga1 = (y_hat < y).values
    flaga2 = (y_hat > y).values
    flaga3 = (y_hat == y).values

    vola1 = data[f'n_asize{i}']
    vola2 = -data[f'n_asize{i}'].shift(1)
    vola3 = data[f'n_asize{i}'] - data[f'n_asize{i}'].shift(1)

    temp_sell = np.zeros(data.shape[0])
    temp_sell[flaga1] = vola1.values[flaga1]
    temp_sell[flaga2] = vola2.values[flaga2]
    temp_sell[flaga3] = vola3.values[flaga3]

    ofi = (temp_buy - temp_sell)
    return ofi

def cal_factor_ofi(df):
    # 7
    #计算OFI1
    ofi1 = cal_ofi(df, 1)[:, np.newaxis]

    # 计算OFI2
    ofi2 = cal_ofi(df, 2)[:, np.newaxis]

    # 计算OFI3
    ofi3 = cal_ofi(df, 3)[:, np.newaxis]

    # 计算OFI4
    ofi4 = cal_ofi(df, 4)[:, np.newaxis]

    # 计算OFI5
    ofi5 = cal_ofi(df, 5)[:, np.newaxis]

    # 计算mofi（等权求和）
    mofi = (ofi1 + ofi2 + ofi3 + ofi4 + ofi5) / 5

    # 计算weight_mofi
    weight_mofi = ofi1 * 1/5 + ofi2 * 2/5 + ofi3 * 3/5 + ofi4 * 4/5 + ofi5 * 5/5
    ret = np.concatenate([ofi1, ofi2, ofi3, ofi4, ofi5, mofi, weight_mofi], axis=1)
    return ret
def cal_factor_slope(df):
    # 7
    #计算slope1
    slope1 = cal_slope(df, 1)[:, np.newaxis]
    # 计算slope2
    slope2 = cal_slope(df, 2)[:, np.newaxis]

    # 计算slope3
    slope3 = cal_slope(df, 3)[:, np.newaxis]

    # 计算slope4
    slope4 = cal_slope(df, 4)[:, np.newaxis]

    # 计算slope5
    slope5 = cal_slope(df, 5)[:, np.newaxis]

    # 计算mofi（等权求和）
    mslope = (slope1 + slope2 + slope3 + slope4 + slope5) / 5

    # 计算weight_mslope
    weight_mslope = slope1 * 1/5 + slope2 * 2/5 + slope3 * 3/5 + slope4 * 4/5 + slope5 * 5/5
    ret = np.concatenate([slope1, slope2, slope3, slope4, slope5, mslope, weight_mslope], axis=1)
    return ret


# #%%
# data = pd.read_csv('./data/train/snapshot_sym0_date0_am.csv', index_col=0)
# aa = cal_factor_slope(data)

#%%
def gen_factor(date, code):
    time_am = np.array([])
    factor_am = np.empty(shape=[0, 42])
    time_pm = np.array([])
    factor_pm = np.empty(shape=[0, 42])

    am_file = f'./data/train/snapshot_sym{code}_date{date}_am.csv'
    isExists = os.path.exists(am_file)
    if isExists:
        data_am = pd.read_csv(am_file, index_col = 0)
        factor_am0, time_am  = cal_imbalance(data_am)
        factor_am1 = cal_spread(data_am)
        factor_am2 = cal_SOIR_PIR(data_am)
        factor_am3 = cal_MCI(data_am)
        factor_am4 = cal_factor_ofi(data_am)
        factor_am5 = cal_factor_slope(data_am)
        factor_am = np.concatenate([factor_am0, factor_am1, factor_am2, factor_am3, factor_am4, factor_am5], axis = 1)

    pm_file = f'./data/train/snapshot_sym{code}_date{date}_pm.csv'
    isExists = os.path.exists(pm_file)
    if isExists:
        data_pm = pd.read_csv(pm_file, index_col=0)
        factor_pm0 , time_pm = cal_imbalance(data_pm)
        factor_pm1 = cal_spread(data_pm)
        factor_pm2 = cal_SOIR_PIR(data_pm)
        factor_pm3 = cal_MCI(data_pm)
        factor_pm4 = cal_factor_ofi(data_pm)
        factor_pm5 = cal_factor_slope(data_pm)
        factor_pm = np.concatenate([factor_pm0, factor_pm1, factor_pm2, factor_pm3, factor_pm4, factor_pm5], axis = 1)
    time = np.concatenate((time_am, time_pm))
    factor = np.concatenate((factor_am, factor_pm))

    col =  ['WBidSum', 'WAskSum', 'Imbalance']
    col.extend([f'Spread_{i}' for i in range(1, 6)])
    col.extend([f'SumMid_{i}' for i in range(1, 6)])
    col.extend([f'Soir_{i}' for i in range(1, 6)])
    col.extend(['WSoir'])
    col.extend([f'Pir_{i}' for i in range(1, 6)])
    col.extend(['WPir'])
    col.extend(['Mcia', 'Mcib', 'MciImbalance'])
    col.extend(['ofi1', 'ofi2', 'ofi3', 'ofi4', 'ofi5', 'mofi', 'weight_mofi'])
    col.extend(['slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'mslope', 'weight_mslope'])
    df = pd.DataFrame(factor, columns = col)
    df['sym'] = code
    df['date'] = date
    df['time'] = time
    return df
# #%%
# gen_factor(0, 0)
#%%
num = 60
j_num = 2
start = time.time()
# n_jobs is the number of parallel jobs
results = Parallel(n_jobs = 6)(delayed(gen_factor)(i, j) for i in range(num) for j in range(j_num))
end = time.time()
print('{:.4f} s'.format(end-start))
#%%
ret = dict()
for res in results:
    print(res)
