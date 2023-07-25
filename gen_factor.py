#%%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time, math
import os
#%%
def cal_imbalance(data):
    '''
    :param data: tick数据
    :return: 单只股票一天的因子值
    '''
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
    ret = np.zeros((data.shape[0], 3))
    col1 = [f'n_bid{i}' for i in range(1, 6)]
    col2 = [f'n_bsize{i}' for i in range(1, 6)]
    col3 = [f'n_ask{i}' for i in range(1, 6)]
    col4 = [f'n_asize{i}' for i in range(1, 6)]

    bid_price = data[col1].values
    bid_volume = data[col2].values
    ask_price = data[col3].values
    ask_volume = data[col4].values

    price_spread = ask_price - bid_price
    price_sum_mid = (ask_price + bid_price) / 2
    ret = np.concatenate([price_spread, price_sum_mid], axis=1)
    return ret
# #%%
# data = pd.read_csv('./data/train/snapshot_sym0_date0_am.csv', index_col=0)
# cal_spread(data)

#%%
def gen_factor(date, code):
    time_am = np.array([])
    factor_am = np.empty(shape=[0, 13])
    time_pm = np.array([])
    factor_pm = np.empty(shape=[0, 13])

    am_file = f'./data/train/snapshot_sym{code}_date{date}_am.csv'
    isExists = os.path.exists(am_file)
    if isExists:
        data_am = pd.read_csv(am_file, index_col = 0)
        factor_am0, time_am  = cal_imbalance(data_am)
        factor_am1 = cal_spread(data_am)
        factor_am = np.concatenate([factor_am0, factor_am1], axis = 1)

    pm_file = f'./data/train/snapshot_sym{code}_date{date}_pm.csv'
    isExists = os.path.exists(pm_file)
    if isExists:
        data_pm = pd.read_csv(pm_file, index_col=0)
        factor_pm0 , time_pm = cal_imbalance(data_pm)
        factor_pm1 = cal_spread(data_pm)
        factor_pm = np.concatenate([factor_pm0, factor_pm1], axis = 1)
    time = np.concatenate((time_am, time_pm))
    factor = np.concatenate((factor_am, factor_pm))

    col =  ['WBidSum', 'WAskSum', 'Imbalance']
    col.extend([f'Spread_{i}' for i in range(1, 6)])
    col.extend([f'SumMid_{i}' for i in range(1, 6)])
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
