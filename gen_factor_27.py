#%%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time, math
import os
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

# #%%
# data = pd.read_csv('./data/train/snapshot_sym0_date0_am.csv', index_col=0)
# aa = cal_MCI(data)

# #%%
# def gen_factor(date, code):
#     time_am = np.array([])
#     factor_am = np.empty(shape=[0, 28])
#     time_pm = np.array([])
#     factor_pm = np.empty(shape=[0, 28])
#
#     am_file = f'./data/train/snapshot_sym{code}_date{date}_am.csv'
#     isExists = os.path.exists(am_file)
#     if isExists:
#         data_am = pd.read_csv(am_file, index_col = 0)
#         factor_am0, time_am  = cal_imbalance(data_am)
#         factor_am1 = cal_spread(data_am)
#         factor_am2 = cal_SOIR_PIR(data_am)
#         factor_am3 = cal_MCI(data_am)
#         print(factor_am0.shape, factor_am1.shape, factor_am2.shape, factor_am3.shape)
#         factor_am = np.concatenate([factor_am0, factor_am1, factor_am2, factor_am3], axis = 1)
#
#     pm_file = f'./data/train/snapshot_sym{code}_date{date}_pm.csv'
#     isExists = os.path.exists(pm_file)
#     if isExists:
#         data_pm = pd.read_csv(pm_file, index_col=0)
#         factor_pm0 , time_pm = cal_imbalance(data_pm)
#         factor_pm1 = cal_spread(data_pm)
#         factor_pm2 = cal_SOIR_PIR(data_pm)
#         factor_pm3 = cal_MCI(data_pm)
#         factor_pm = np.concatenate([factor_pm0, factor_pm1, factor_pm2, factor_pm3], axis = 1)
#     time = np.concatenate((time_am, time_pm))
#     factor = np.concatenate((factor_am, factor_pm))
#
#     col =  ['WBidSum', 'WAskSum', 'Imbalance']
#     col.extend([f'Spread_{i}' for i in range(1, 6)])
#     col.extend([f'SumMid_{i}' for i in range(1, 6)])
#     col.extend([f'Soir_{i}' for i in range(1, 6)])
#     col.extend(['WSoir'])
#     col.extend([f'Pir_{i}' for i in range(1, 6)])
#     col.extend(['WPir'])
#     col.extend(['Mcia', 'Mcib', 'MciImbalance'])
#     df = pd.DataFrame(factor, columns = col)
#     df['sym'] = code
#     df['date'] = date
#     df['time'] = time
#     return df
# #%%
# gen_factor(0, 0)
# #%%
# num = 60
# j_num = 2
# start = time.time()
# # n_jobs is the number of parallel jobs
# results = Parallel(n_jobs = 6)(delayed(gen_factor)(i, j) for i in range(num) for j in range(j_num))
# end = time.time()
# print('{:.4f} s'.format(end-start))
# #%%
# ret = dict()
# for res in results:
#     print(res)
