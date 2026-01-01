# predictor.py
import torch
import numpy as np
import pickle
from model import TransAttention # 引用您的模型
import os

class MetalPredictor:
    def __init__(self, model_path='./models/best_model.pth', device='cpu'):
        self.device = device
        self.year = 2023
        self.week_num = 4
        self.pred_len = 4
        self.time_step = 5
        self.hidden_dim = 64
        
        # 載入資產列表以確定 input_num
        with open('./data/material_data.pkl', 'rb') as f:
            assets_df = pickle.load(f)
            # 取得所有唯一的資產名稱並排序 (必須與訓練時一致)
            self.namelist = sorted(assets_df.index.get_level_values('Material').unique().tolist())
            self.input_num = len(self.namelist)

        # 載入圖結構
        self.stock_edge = np.load("./data/stock_graph_data.pkl", allow_pickle=True)
        self.all_edge = np.load("./data/total_graph_data.pkl", allow_pickle=True)
        
        # 模型參數 (需與訓練時一致)
        self.input_dim = 10
        self.indi_dim = 10
        self.indi_num = 1
        
        # 初始化模型
        self.model = TransAttention(
            self.input_dim, self.indi_dim, self.indi_num, self.time_step, 
            self.hidden_dim, self.stock_edge, self.all_edge, 
            self.week_num, self.pred_len, self.input_num, self.device
        )
        
        # 載入權重
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("模型載入成功！")
        else:
            print(f"找不到模型檔案: {model_path}")

    def predict(self, input_tensor, indi_tensor):
        with torch.no_grad():
            outputs = self.model(input_tensor.to(self.device), indi_tensor.to(self.device))
            reg_output = outputs[0] # [batch, stocks, pred_weeks]
            cls_output = outputs[1] # [batch, stocks, pred_weeks]
            
            # 轉換為 numpy 並移除 batch 維度
            # reg_output: [stocks, pred_weeks]
            return reg_output.cpu().numpy()[0], cls_output.cpu().numpy()[0]
            
    def get_asset_names(self):
        return self.namelist