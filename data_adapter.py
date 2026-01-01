# data_adapter.py
import yfinance as yf
import pandas as pd
import torch
import numpy as np
import pickle
from datetime import datetime, timedelta

# 設定對應表：您的 material 名稱對應 Yahoo Finance Ticker
# 請根據您實際訓練的資產順序填寫，這非常重要，順序必須與訓練時一致！
TICKER_MAP = {
    'aluminum': 'ALI=F',   # 鋁期貨
    'copper': 'HG=F',      # 銅期貨
    'gold': 'GC=F',        # 黃金期貨
    'lead': 'LEAD=F',      # 鉛 (註: Yahoo 可能無此數據，若抓不到可嘗試用倫敦金屬交易所代碼)
    'nickel': 'TICKER_NEEDED', # 鎳 (Yahoo Finance 較難抓到鎳期貨，若無可填 'LNrn.L' 試試，或暫時忽略)
    'palladium': 'PA=F',   # 鈀金期貨
    'platinum': 'PL=F',    # 白金期貨
    'silver': 'SI=F',      # 白銀期貨
    'tin': 'TIN=F',        # 錫
    'zinc': 'ZNC=F',       # 鋅
    # 如果您有其他資產，請依照此格式補上，記得 Key 要用小寫
}

class RealtimeDataManager:
    def __init__(self, materials_list):
        self.materials_list = materials_list
        self.time_step = 5 # 一週5天
        self.week_num = 4  # 過去4週
        self.feature_dim = 6 # Open, High, Low, Close, Volume, Adj Close (假設)

    def get_live_data(self):
        """
        從 Yahoo Finance 抓取最新數據並整理成 Tensor
        """
        # 為了確保有足夠數據計算 MA 和填滿 4 週，我們抓取過去 60 天
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        data_list = []
        
        # 1. 抓取所有資產的價格
        print("正在下載最新金屬價格...")
        for name in self.materials_list:
            ticker = TICKER_MAP.get(name)
            
            if ticker is None:
                print(f"❌ 警告: 找不到資產 '{name}' 的對應代碼！請檢查 TICKER_MAP。")
            else:
                print(f"📥 正在下載: {name} ({ticker})")


            if not ticker:
                # 如果找不到對應，暫時用假數據填充 (避免報錯)，實際請務必補全 TICKER_MAP
                df = pd.DataFrame(np.zeros((30, 6)), columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
            else:
                df = yf.download(ticker, start=start_date, progress=False)
                # 確保只有 6 個 feature，順序要對
                if df.empty:
                    # 建立一個全 0 的 DataFrame 作為備用
                    df = pd.DataFrame(np.zeros((30, 6)), columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
                else:
                    # 1. 處理 'Adj Close' 缺失的問題 (期貨常發生)
                    if 'Adj Close' not in df.columns:
                        if 'Close' in df.columns:
                            df['Adj Close'] = df['Close']
                        else:
                            df['Adj Close'] = 0

                    # 2. 處理 'Volume' 缺失的問題 (有些指數沒成交量)
                    if 'Volume' not in df.columns:
                        df['Volume'] = 0

                    # 3. 確保欄位存在後，再進行選取與排序
                    # 這裡加入 try-except 以防萬一還有其他欄位名稱變更
                    try:
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].ffill()
                    except KeyError as e:
                        print(f"⚠️ 數據格式警告: {e}, 將使用 0 填充缺失欄位")
                        # 萬用備案：缺少的欄位都補 0
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                        for col in required_cols:
                            if col not in df.columns:
                                df[col] = 0
                        df = df[required_cols].ffill()

            # 取最近的 20 天 (4週 * 5天)
            # 注意：這裡簡化處理，實際應用需處理週末/休市對齊
            if len(df) < 20:
                 # 數據不足補 0
                 padding = pd.DataFrame(np.zeros((20-len(df), 6)), columns=df.columns)
                 df = pd.concat([padding, df], axis=0)
            
            recent_data = df.tail(20).values # shape (20, 6)
            data_list.append(recent_data)
            
        # 2. 轉換形狀
        # 目標: [1, input_num, week_num, time_step, features]
        # data_list shape: (21, 20, 6)
        batch_data = np.array(data_list) 
        
        # Reshape to (21, 4, 5, 6)
        batch_data = batch_data.reshape(len(self.materials_list), self.week_num, self.time_step, self.feature_dim)
        
        # Add batch dimension: (1, 21, 4, 5, 6)
        input_tensor = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(0)
        padding = torch.zeros(1, len(self.materials_list), self.week_num, self.time_step, 4)
        
        # 拼接在一起變 [1, 21, 4, 5, 10]
        input_tensor = torch.cat((input_tensor, padding), dim=-1)
        return input_tensor

    def get_indicator_data(self, indi_path='./data/indicator_data.pkl'):
        """
        MVP 權宜之計：讀取歷史指標數據，取最後一週的數據重複使用。
        """
        with open(indi_path, 'rb') as f:
            indi_data = pickle.load(f)
        
        # 這裡需要根據您 dataset_v2 的邏輯還原指標數據的處理
        # 假設我們只是為了讓模型跑起來，我們建立一個符合維度的 Dummy Tensor
        # 模型需要: [batch, indi_num, week, days, features] 
        # 根據您的 train.py: [128, 1, 4, 5, 17]
        
        # 這裡生成全 0 或隨機數據，或讀取最後一筆真實數據
        # 建議：未來這裡要接 FRED API
        dummy_indi = torch.zeros(1, 1, 4, 5, 10) 
        return dummy_indi