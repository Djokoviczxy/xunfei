#%%
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time, math
import os
import warnings
warnings.filterwarnings("ignore")
import gc
from tqdm import tqdm
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

    bid_price = data[col1].values
    bid_volume = data[col2].values
    ask_price = data[col3].values
    ask_volume = data[col4].values

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

    log_price_bid = np.log(data[f'n_bid{i}'])
    log_price_ask = np.log(data[f'n_ask{i}'])
    log_vol_bid = np.log(data[f'n_bsize{i}'])
    log_vol_ask = np.log(data[f'n_asize{i}'])

    select = ((log_vol_bid + log_vol_ask) != 0)
    log_price_bid = log_price_bid[select]
    log_price_ask = log_price_ask[select]
    log_vol_bid = log_vol_bid[select]
    log_vol_ask = log_vol_ask[select]

    slope = (log_price_ask - log_price_bid) / (log_vol_ask + log_vol_bid)
    return slope.values


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
def cal_extra_fea(df_original_fea, window, fac_to_remain=None, stadardize=False):
    '''
    df_original_fea: pd.Series, index为date_id和security_id, name为因子名称
    window: list
    fac_to_remain: list, 指保留的因子名称
     '''
    f = df_original_fea.name
    fea_wide = df_original_fea.unstack()
    feature = pd.DataFrame(index=df_original_fea.index)
    # 是否进行量纲处理
    if stadardize:
        df_stadardize = df_original_fea
    else:
        df_stadardize = 1
    for w in window:
        # mom
        fname = f'{f}_mom{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            #             fea_tmp = fea_wide/fea_wide.shift(w)-1
            #             default_value = np.nan
            #             fea_tmp = np.where(fea_wide.shift(w) != 0, fea_tmp, default_value)
            #             feature[fname] = fea_tmp.stack(dropna=False)
            shifted_fea_wide = fea_wide.shift(w)
            divisor = np.where(shifted_fea_wide != 0, shifted_fea_wide, np.nan)
            fea_tmp = (fea_wide - shifted_fea_wide) / divisor
            feature[fname] = fea_tmp.stack(dropna=False)

        # mean
        fname = f'{f}_mean{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).mean()
            feature[fname] = fea_tmp.stack(dropna=False)
        # std
        fname = f'{f}_std{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).std()
            feature[fname] = fea_tmp.stack(dropna=False)
        # max
        fname = f'{f}_max{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).max()
            feature[fname] = fea_tmp.stack(dropna=False) / df_stadardize
        # min
        fname = f'{f}_min{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).min()
            feature[fname] = fea_tmp.stack(dropna=False) / df_stadardize
        # skew
        fname = f'{f}_skew{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).skew()
            feature[fname] = fea_tmp.stack(dropna=False) / df_stadardize
        # kurt
        fname = f'{f}_kurt{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).kurt()
            feature[fname] = fea_tmp.stack(dropna=False) / df_stadardize
        # qtlu
        fname = f'{f}_qtlu{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).quantile(0.8)
            feature[fname] = fea_tmp.stack(dropna=False) / df_stadardize
        # qtld
        fname = f'{f}_qtld{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).quantile(0.2)
            feature[fname] = fea_tmp.stack(dropna=False) / df_stadardize
        # rank
        fname = f'{f}_rank{w}'
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).rank(pct=True)
            feature[fname] = fea_tmp.stack(dropna=False)
        # IMAX
        fname = f'{f}_imax{w}'
        findMaxIdx = lambda series: series.shape[0] - series.reset_index(drop=True).idxmax()
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).apply(findMaxIdx) / w
            feature[fname] = fea_tmp.stack(dropna=False)
        # IMIN
        fname = f'{f}_imin{w}'
        findMinIdx = lambda series: series.shape[0] - series.reset_index(drop=True).idxmin()
        if fac_to_remain is None or fname in fac_to_remain:
            fea_tmp = fea_wide.rolling(w, min_periods=w).apply(findMinIdx) / w
            feature[fname] = fea_tmp.stack(dropna=False)

    return feature

# #%%
# def gen_factor_train(date, code):
#     print(f"====== generate factor for date{date} ======")
#     # 原始特征
#     col_org = ['date', 'time', 'sym']
#     # 生成的特征
#     col_gen =  ['WBidSum', 'WAskSum', 'Imbalance']
#     col_gen.extend([f'Spread_{i}' for i in range(1, 6)])
#     col_gen.extend([f'SumMid_{i}' for i in range(1, 6)])
#     col_gen.extend([f'Soir_{i}' for i in range(1, 6)])
#     col_gen.extend(['WSoir'])
#     col_gen.extend([f'Pir_{i}' for i in range(1, 6)])
#     col_gen.extend(['WPir'])
#     col_gen.extend(['Mcia', 'Mcib', 'MciImbalance'])
#     col_gen.extend(['ofi1', 'ofi2', 'ofi3', 'ofi4', 'ofi5', 'mofi', 'weight_mofi'])
#     col_gen.extend(['slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'mslope', 'weight_mslope'])
#     # 算子
#     operators = ['mom', 'mean', 'std', 'max' , 'min', 'skew', 'kurt', 'qtlu', 'qtld', 'rank', 'imax', 'imin']
#     # 'mean', 'std', 'max' , 'min', 'skew', 'kurt', 'qtlu', 'qtld', 'rank', 'imax', 'imin'
#     # 扩充特征——生成的特征
#     fea_gen_expanded = [f'{f}_{op}{w}' for f in col_gen for op in operators for w in [5, 10, 20, 40]]
#     # 所有被扩充的特征
#     fac_to_remain = fea_gen_expanded
#
#     window = [5, 10, 20, 40]
#
#     # 处理csv不存在的情况
#     df_am = pd.DataFrame()
#     df_pm = pd.DataFrame()
#
#     am_file = f'./data/train/snapshot_sym{code}_date{date}_am.csv'
#     isExists = os.path.exists(am_file)
#     if isExists:
#         data_am = pd.read_csv(am_file, index_col = 0)
#         factor_am0, time_am  = cal_imbalance(data_am)
#         factor_am1 = cal_spread(data_am)
#         factor_am2 = cal_SOIR_PIR(data_am)
#         factor_am3 = cal_MCI(data_am)
#         factor_am4 = cal_factor_ofi(data_am)
#         factor_am5 = cal_factor_slope(data_am)
#
#         am_index = data_am[col_org]
#         factor_am = np.concatenate(
#             [am_index.values, factor_am0, factor_am1, factor_am2, factor_am3, factor_am4, factor_am5], axis=1)
#         df_am = pd.DataFrame(factor_am, columns=col_org+col_gen)
#
#     pm_file = f'./data/train/snapshot_sym{code}_date{date}_pm.csv'
#     isExists = os.path.exists(pm_file)
#     if isExists:
#         data_pm = pd.read_csv(pm_file, index_col=0)
#         factor_pm0 , time_pm = cal_imbalance(data_pm)
#         factor_pm1 = cal_spread(data_pm)
#         factor_pm2 = cal_SOIR_PIR(data_pm)
#         factor_pm3 = cal_MCI(data_pm)
#         factor_pm4 = cal_factor_ofi(data_pm)
#         factor_pm5 = cal_factor_slope(data_pm)
#
#         pm_index = data_pm[col_org]
#         factor_pm = np.concatenate([pm_index.values, factor_pm0, factor_pm1, factor_pm2, factor_pm3, factor_pm4, factor_pm5], axis = 1)
#         df_pm = pd.DataFrame(factor_pm, columns=col_org + col_gen)
#
#     df = pd.concat([df_am, df_pm], axis = 0)
#     if len(df) == 0:
#         return df
#     df = df.set_index(['time', 'sym'])
#     df_expanded = pd.DataFrame(index=df.index)
#     for fea in df.columns:
#         tmp = cal_extra_fea(df[fea], window=window, fac_to_remain=fac_to_remain)
#         df_expanded = pd.concat([df_expanded, tmp], axis=1)
#     factor = pd.concat([df, df_expanded], axis=1)
#     factor = factor.reset_index()
#     factor = factor.replace([np.inf, -np.inf], np.nan)
#     factor[factor.select_dtypes(np.float64).columns] = factor.select_dtypes(np.float64).astype(np.float32)
#
#     return factor
# #%%
# num = 64
# start = time.time()
# results_train_1 = Parallel(n_jobs = 8)(delayed(gen_factor_train)(i, j) for i in tqdm(range(num), position=0) for j in range(4, 5))
# end = time.time()
# print('{:.4f} s'.format(end-start))
#
# df_train_1 = pd.concat(results_train_1, axis=0)
# df_train_1 = df_train_1.sort_values(['sym','date','time'])
# # 记得修改
# df_train_1.to_parquet('./feature_data/df_train_4.parquet')

#%%
def gen_factor_test(date, code):
    print(f"====== generate factor for date{date} ======")
    # 原始特征
    col_org = ['date', 'time', 'sym']
    # 生成的特征
    col_gen =  ['WBidSum', 'WAskSum', 'Imbalance']
    col_gen.extend([f'Spread_{i}' for i in range(1, 6)])
    col_gen.extend([f'SumMid_{i}' for i in range(1, 6)])
    col_gen.extend([f'Soir_{i}' for i in range(1, 6)])
    col_gen.extend(['WSoir'])
    col_gen.extend([f'Pir_{i}' for i in range(1, 6)])
    col_gen.extend(['WPir'])
    col_gen.extend(['Mcia', 'Mcib', 'MciImbalance'])
    col_gen.extend(['ofi1', 'ofi2', 'ofi3', 'ofi4', 'ofi5', 'mofi', 'weight_mofi'])
    col_gen.extend(['slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'mslope', 'weight_mslope'])
    # 算子
    operators = ['mom', 'mean', 'std', 'max' , 'min', 'skew', 'kurt', 'qtlu', 'qtld', 'rank', 'imax', 'imin']
    # 'mean', 'std', 'max' , 'min', 'skew', 'kurt', 'qtlu', 'qtld', 'rank', 'imax', 'imin'
    # 扩充特征——生成的特征
    fea_gen_expanded = [f'{f}_{op}{w}' for f in col_gen for op in operators for w in [5, 10, 20, 40]]
    # 所有被扩充的特征
    fac_to_remain = fea_gen_expanded

    window = [5, 10, 20, 40]

    # 处理csv不存在的情况
    df_am = pd.DataFrame()
    df_pm = pd.DataFrame()

    am_file = f'./data/test/snapshot_sym{code}_date{date}_am.csv'
    isExists = os.path.exists(am_file)
    if isExists:
        data_am = pd.read_csv(am_file, index_col = 0)
        factor_am0, time_am  = cal_imbalance(data_am)
        factor_am1 = cal_spread(data_am)
        factor_am2 = cal_SOIR_PIR(data_am)
        factor_am3 = cal_MCI(data_am)
        factor_am4 = cal_factor_ofi(data_am)
        factor_am5 = cal_factor_slope(data_am)

        am_index = data_am[col_org]
        factor_am = np.concatenate(
            [am_index.values, factor_am0, factor_am1, factor_am2, factor_am3, factor_am4, factor_am5], axis=1)
        df_am = pd.DataFrame(factor_am, columns=col_org+col_gen)

    pm_file = f'./data/test/snapshot_sym{code}_date{date}_pm.csv'
    isExists = os.path.exists(pm_file)
    if isExists:
        data_pm = pd.read_csv(pm_file, index_col=0)
        factor_pm0 , time_pm = cal_imbalance(data_pm)
        factor_pm1 = cal_spread(data_pm)
        factor_pm2 = cal_SOIR_PIR(data_pm)
        factor_pm3 = cal_MCI(data_pm)
        factor_pm4 = cal_factor_ofi(data_pm)
        factor_pm5 = cal_factor_slope(data_pm)

        pm_index = data_pm[col_org]
        factor_pm = np.concatenate([pm_index.values, factor_pm0, factor_pm1, factor_pm2, factor_pm3, factor_pm4, factor_pm5], axis = 1)
        df_pm = pd.DataFrame(factor_pm, columns=col_org + col_gen)

    df = pd.concat([df_am, df_pm], axis = 0)
    if len(df) == 0:
        return df
    df = df.set_index(['time', 'sym'])
    df_expanded = pd.DataFrame(index=df.index)
    for fea in df.columns:
        tmp = cal_extra_fea(df[fea], window=window, fac_to_remain=fac_to_remain)
        df_expanded = pd.concat([df_expanded, tmp], axis=1)
    factor = pd.concat([df, df_expanded], axis=1)
    factor = factor.reset_index()
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor[factor.select_dtypes(np.float64).columns] = factor.select_dtypes(np.float64).astype(np.float32)

    return factor
#%%
num = range(64, 79)
j_num = range(9, 10)
start = time.time()
results_test_1 = Parallel(n_jobs=8)(delayed(gen_factor_test)(i, j) for i in tqdm(num, position=0) for j in j_num)
end = time.time()
print('{:.4f} s'.format(end-start))

df_test_1 = pd.concat(results_test_1, axis=0)
df_test_1 = df_test_1.sort_values(['sym','date','time'])
df_test_1.to_parquet('./feature_data/df_test_9.parquet')

#%%
train_result = []
path = './feature_data/'
file_list = os.listdir(path)
file_list.sort()
for file_dir in file_list:
    if file_dir.startswith('df_train'):
        if (file_dir[9] <= '9') & (file_dir[9] > '6'):
            df = pd.read_parquet(path + file_dir)
            train_result.append(df)
data = pd.concat(train_result, axis=0)  #自行调整行列
data = data.reset_index(drop = True)
data.to_parquet('./feature_data/df_train_7_9.parquet')
#%%
# label = pd.read_parquet('./test_label.parquet')
# data = data.drop(columns = ['time', 'sym', 'date'])
data = data.reset_index(drop = True)
# label = label.reset_index(drop = True)
# data = pd.concat([label, data], axis = 1)
data.to_parquet('./feature_data/df_train_0_4')