# %% [markdown]

# %%
import torch
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import math

# %%
# Constants
DATE_FORMAT = '%Y-%m-%d'
TEST_YEAR = 2023
DIM_PE = 10  # Dimension for positional encoding
INDI_DIM = 10
FEATURES_DIM = 6  # Assuming 6 features: Open, High, Low, Close, Volume, Adj Close
WEEK_NUM = 3
PRED_LEN = 4
TIME_STEP = 5

class DataManager:
    def __init__(self, assets_path, indi_path, train_start_dt, train_end_dt, test_end_dt):
        self.assets_path = assets_path
        self.indi_path = indi_path
        self.train_start_dt = train_start_dt
        self.train_end_dt = train_end_dt
        self.test_end_dt = test_end_dt
        
    def read_assets_data(self):
        with open(self.assets_path, 'rb') as f:
            assets_df = pickle.load(f)
        
        assets_df = self.filter_by_date_range(assets_df, self.train_start_dt, self.test_end_dt)
        assets_df_sort = assets_df.groupby(level=0).apply(lambda x: x.sort_values('Material')).droplevel(0)

        assets_train = assets_df_sort[assets_df_sort.index.get_level_values('Date') <= self.train_end_dt]
        assets_test = assets_df_sort[(assets_df_sort.index.get_level_values('Date') > self.train_end_dt) & (assets_df_sort.index.get_level_values('Date') <= test_end_dt)]
        return assets_train, assets_test

    def read_indi_data(self):
        with open(self.indi_path, 'rb') as f:
            indi_df = pickle.load(f)
    
        indi_df = self.filter_by_date_range(indi_df, self.train_start_dt, self.test_end_dt)
        indi_train = indi_df.loc[indi_df.index.get_level_values('Date') <= self.train_end_dt]
        indi_test = indi_df.loc[(indi_df.index.get_level_values('Date') > self.train_end_dt)
                                 & (indi_df.index.get_level_values('Date') <= self.test_end_dt)]

        return indi_train, indi_test

    @staticmethod
    def filter_by_date_range(dfs, start_date, end_date):
        mask = (dfs.index.get_level_values('Date') >= start_date) & (dfs.index.get_level_values('Date') <= end_date)

        return dfs.loc[mask]

    @staticmethod
    def positional_encoding(df):
        features = df.drop(['month', 'year'], axis=1)
        updated_df = pd.DataFrame(index=df.index, columns=features.columns)
        for _, group in df.groupby([df.index.year, df.index.month]):
            length = len(group)
            position = np.arange(length).reshape(-1, 1)
            div_term = np.exp(np.arange(0, DIM_PE, 2) * -(math.log(10000.0) / DIM_PE))
            pe = np.zeros((length, DIM_PE))
            pe[:, 0::2] = np.sin(position * div_term / length)
            pe[:, 1::2] = np.cos(position * div_term / length)
            updated_group = features.loc[group.index].values + pe
            updated_df.loc[group.index] = updated_group

        return updated_df.astype(float)


# %%
class mydata(Dataset):
    def __init__(self, price_data, indi_dict_data, data_week_num, target_week_num):
        # file_path, price train/test data, indicator train/test data, week number
        self.price_dict_data, self.price_ma_data = self.price_to_dict(price_data)
        self.data_week_num = data_week_num
        self.target_week_num = target_week_num
        self.window_size = data_week_num * TIME_STEP + self.target_week_num * TIME_STEP # 4 weeks * 5 days + 5 days *(target)
        self.namelist = list(self.price_dict_data.keys())
        self.indi_dict_data = indi_dict_data
        self.price_df_length = len(next(iter(self.price_dict_data.values())))
        self.window_data_list, self.window_indi_list = [], []
        self.window_reg_list, self.window_cls_list = [], []

        for idx in range(0, self.price_df_length - self.window_size + 1):
            weeks_data_list, reg_target_list, cls_target_list, indi_weeks_list = [], [], [], []
            next_start = idx + self.data_week_num * TIME_STEP

            # Get price data
            for name, data in self.price_ma_data.items():
                weeks_data = data.iloc[idx:next_start].values
                reg_asset_list, cls_asset_list = [], []
                reg_previous = data.iloc[idx:next_start - TIME_STEP]['Close'].values.mean()

                for i in range(self.target_week_num):
                    reg_weeks = data.iloc[next_start + i*TIME_STEP:next_start + (i+1)*TIME_STEP]['Close'].values
                    reg_weeks_mean = torch.tensor(reg_weeks, dtype=torch.float32).mean().item()
                    cls_next_week = 1 if reg_weeks_mean > reg_previous else 0
                    
                    reg_asset_list.append(reg_weeks_mean)
                    cls_asset_list.append(cls_next_week)

                    reg_previous = reg_weeks_mean
                    # print(weeks_data.shape, reg_next_week.shape, cls_next_week)

                weeks_data_list.append(torch.tensor(weeks_data, dtype=torch.float32))
                reg_target_list.append(torch.tensor(reg_asset_list, dtype=torch.float32))
                cls_target_list.append(torch.tensor(cls_asset_list, dtype=torch.float32))
                
            # Get indicator data
            for _, indi_df in self.indi_dict_data.items():
                indi_weeks_data = indi_df.iloc[idx:next_start].values
                indi_weeks_list.append(torch.tensor(indi_weeks_data, dtype=torch.float32))

            window_data = torch.stack(weeks_data_list)
            window_indi = torch.stack(indi_weeks_list)
            window_reg = torch.stack(reg_target_list) # [assets_num, target_week_num]
            window_cls = torch.stack(cls_target_list) # [assets_num, target_week_num]
            window_data = window_data.reshape(-1, self.data_week_num, TIME_STEP, FEATURES_DIM+PRED_LEN) # 6: OHLVAV
            #window_indi = window_indi.reshape(-1, self.data_week_num, TIME_STEP, indicator_num+dim_pe) # V2: 10+10: positional encoding dim + indicator value (10)
            window_indi = window_indi.reshape(-1, self.data_week_num, TIME_STEP, INDI_DIM) # V3: 10: indicator value
            print(window_data.shape, window_indi.shape, window_reg.shape, window_cls.shape)

            self.window_data_list.append(window_data)
            self.window_indi_list.append(window_indi)
            self.window_reg_list.append(window_reg)
            self.window_cls_list.append(window_cls)

    def __len__(self):
        price_df_length = len(next(iter(self.price_dict_data.values())))
        return price_df_length - self.window_size + 1

    def __getitem__(self, idx):
        return self.window_data_list[idx], self.window_indi_list[idx], self.window_reg_list[idx], self.window_cls_list[idx]

    def get_namelist(self):
        return self.namelist
    
    def add_ma(self, dict, col_name='Close'):
        # add moving average
        for name, df in dict.items():
            for i in range(1, PRED_LEN+1):
                window_size = i * TIME_STEP
                df[f'MA_{window_size}'] = df[col_name].rolling(window=window_size).mean().fillna(0)

        return dict
    
    def price_to_dict(self, data):
        '''output: {dict} key: material name, value: material df '''

        dfs = data.reset_index(level='Material')
        date = dfs.index.drop_duplicates()
        price_dict_fetch = {name: group.drop(columns='Material') for name, group in dfs.groupby('Material')}
        # print('price_dict_fetch:', price_dict_fetch)
        price_dict_ma = self.add_ma(price_dict_fetch)
        price_df_length = len(next(iter(price_dict_fetch.values())))

        return price_dict_fetch, price_dict_ma

# %%
if __name__ == "__main__":   

    test_end_date = datetime(TEST_YEAR, 12, 31).strftime('%Y-%m-%d')
    train_end_date = datetime(TEST_YEAR - 1, 12, 31).strftime('%Y-%m-%d')
    train_start_date = datetime(TEST_YEAR - 5, 1, 1).strftime('%Y-%m-%d')
    train_start_dt = datetime.strptime(train_start_date, DATE_FORMAT)
    train_end_dt = datetime.strptime(train_end_date, DATE_FORMAT)
    test_end_dt = datetime.strptime(test_end_date, DATE_FORMAT)
    print(train_start_dt, train_end_dt, test_end_dt)

    assets_path = './data/material_data.pkl'
    indi_path = './data/indicator_data.pkl'

    data_manager = DataManager(assets_path, indi_path, train_start_dt, train_end_dt, test_end_dt)
    assets_train, assets_test = data_manager.read_assets_data()
    indi_train, indi_test = data_manager.read_indi_data()
    
    indi_ps_train = DataManager.positional_encoding(indi_train)
    indi_ps_test = DataManager.positional_encoding(indi_test)

    train_data = mydata(assets_train, indi_ps_train, WEEK_NUM, PRED_LEN) #Run time error
    test_data = mydata(assets_test, indi_ps_test, WEEK_NUM, PRED_LEN)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=False, drop_last=False)

    os.makedirs(f'./data/{test_end_dt.year}', exist_ok=True)

    with open(f'./data/{test_end_dt.year}/train_{WEEK_NUM*5}_ma.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    with open(f'./data/{test_end_dt.year}/test_{WEEK_NUM*5}_ma.pkl', 'wb') as f:
        pickle.dump(test_data, f)


