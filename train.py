# %%
import pickle
import wandb
import sys
import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from dataset_v2 import mydata
from model import GraphAttention, TransAttention
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
from model import *

# ================= 加入這段程式碼 =================
import __main__
# 將 dataset_v2 中的 mydata 類別指派給 __main__ 的屬性
setattr(__main__, "mydata", mydata)
# =================================================

# %%
# year = sys.argv[1]
if __name__ == "__main__":
    year = 2023
    week_num = 4
    pred_len = 4
    device = 'cuda:0'
    weight = 1
    epochs = 75
    learning_rate = 0.01
    hidden_dim = 64
    batch_size = 128

    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    args = parser.parse_args()

    year = args.year

    #D:\School_Project\Project\Project\data\2023

    with open(f'./data/{year}/train_{week_num*5}_ma.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open(f'./data/{year}/test_{week_num*5}_ma.pkl', 'rb') as f:
        test_data = pickle.load(f)


    stock_edge = np.load("./data/stock_graph_data.pkl", allow_pickle=True) # (2, 30)
    all_edge = np.load("./data/total_graph_data.pkl", allow_pickle=True) # (2, 30)


    train_loader = DataLoader(
        dataset=train_data,     
        batch_size=batch_size,  
        shuffle=True,
        pin_memory=True,
        num_workers=6,
    )

    test_loader = DataLoader(
        dataset=test_data,     
        batch_size=batch_size,      
        shuffle=False,
        pin_memory=True,
        num_workers=6,
    )

    train_features, indi_dict, labels_reg, labels_cls = next(iter(train_loader))

    # %%
    # torch.Size([128, 21, 4, 5, 6])
    input_num = train_features.shape[1] # 21 stocks
    week_num = train_features.shape[2] # 4 weeks
    time_step = train_features.shape[3] # 5 days
    input_dim = train_features.shape[-1] # 6 features
    # print(input_num, week_num, time_step, input_dim)

    # indicator_data: [128, 10, 4, 5, 17] 
    # (batch, indicator_num, week_num, days_a_week, PE_dim(16) + indicator value (1))
    indi_num = indi_dict.shape[1]
    indi_dim = indi_dict.shape[-1]
    # print(indi_num, indi_dim)

    # %%
    # train_model = GraphAttention(input_dim,indi_dim,indi_num,time_step,hidden_dim,stock_edge,all_edge,week_num,pred_len,input_num,device)
    train_model = TransAttention(input_dim,indi_dim,indi_num,time_step,hidden_dim,stock_edge,all_edge,week_num,pred_len,input_num,device)
    # train_model = torch.nn.DataParallel(train_model)
    train_model.to(device)

    # %%
    # initialize parameters
    for p in train_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    pytorch_total_params = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
    # print("Number of parameters:%s" % pytorch_total_params)

    # %%
    # select_cate_dict, asset_idx_dict = categorize_assets(namelist, asset_categories)
    # print(select_cate_dict, asset_idx_dict)
    # print(train_features.shape, indi_dict.shape, labels_reg.shape, labels_cls.shape)

    # %%

    def train(model, reg_criterion, cls_criterion, optimizer, scheduler):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0 
            with tqdm(train_loader, unit=" batches") as train_epoch:
                for train_features, indi_dict, labels_reg, labels_cls in train_epoch:
                    train_features, indi_dict, labels_reg, labels_cls = train_features.to(device), indi_dict.to(device), labels_reg.to(device), labels_cls.to(device)
                    train_epoch.set_description(f"Epoch {epoch}") # Show epoch number

                    # forward
                    #reg_output, cls_output = model(train_features, indi_dict)
                    
                    # forward
                    outputs = model(train_features, indi_dict)
                    # 確保無論模型回傳幾個值(例如回傳了 attention map)，我們只取前兩個作為預測結果
                    reg_output = outputs[0]
                    cls_output = outputs[1] 

                    #calculate loss
                    reg_loss = reg_criterion(reg_output, labels_reg)
                    cls_loss = cls_criterion(cls_output, labels_cls)
                    reg_loss = torch.mean(reg_loss)
                    cls_loss = torch.mean(cls_loss)

                    total_loss = weight*reg_loss + (1-weight)*cls_loss

                    # feedback and optimization 
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    running_loss += total_loss.item()
                    
            with torch.no_grad():
                average_loss = running_loss / len(train_loader)


            scheduler.step(average_loss)

        save_dir = f"model_save_{year}"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_save_path = os.path.join(save_dir, f"state_dict_{year}_{timestamp}.pth")
        torch.save(model.state_dict(), model_save_path)

        return reg_output, cls_output

    # %%
    reg_criterion = nn.L1Loss()
    cls_criterion = nn.BCELoss()
    optimizer = optim.Adam(train_model.parameters(), weight_decay=0,lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=10, verbose=True)

    pred_reg, pred_cls = train(train_model, reg_criterion, cls_criterion, optimizer, scheduler)

    # %%
    from torchmetrics.classification import BinaryAccuracy
    from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError

    def test(model, reg_criterion, cls_criterion):
        model.eval()  # set the model to evaluation mode
        total_reg_loss = 0
        total_cls_loss = 0
        total_reg_mae = 0
        cls_metric = BinaryAccuracy().to(device)
        mae_metric = MeanAbsoluteError().to(device)
        mape_metric = MeanAbsolutePercentageError().to(device)
        reg_list, labels_reg_list = [], []
        total_reg_mae =  [0] * pred_len
        total_reg_mape = [0] * pred_len

        with torch.no_grad():
            with tqdm(test_loader, unit=" batches") as test_epoch:
                for test_feature, indi_dict, labels_reg, labels_cls in test_epoch:
                    test_feature, indi_dict, labels_reg, labels_cls = test_feature.to(device), indi_dict.to(device), labels_reg.to(device), labels_cls.to(device)
                    #reg_output, cls_output = model(test_feature, indi_dict)

                    outputs = model(test_feature, indi_dict)
                    reg_output = outputs[0]
                    cls_output = outputs[1]
                    
                    labels_reg_selected = labels_reg
                    reg_list += reg_output.cpu().numpy().tolist()
                    labels_reg_list += labels_reg_selected.cpu().numpy().tolist()
                    reg_loss = reg_criterion(reg_output, labels_reg_selected)
                    cls_loss = cls_criterion(cls_output, labels_cls)
                    total_reg_loss += reg_loss.item()
                    total_cls_loss += cls_loss.item()

                    cls_metric.update(cls_output, labels_cls)
                    for i in range(pred_len):
                        total_reg_mae[i] += mae_metric(labels_reg[:, :, i], reg_output[:, :, i]).item()
                        total_reg_mape[i] += mape_metric(labels_reg[:, :, i], reg_output[:, :, i]).item()
                        # mae = mae_metric(labels_reg[:, :, i], reg_output[:, :, i])
                        # total_reg_mae[i] += mae

        avg_reg_loss = round(total_reg_loss / len(test_loader), 5)
        avg_cls_loss = round(total_cls_loss / len(test_loader), 5)
        avg_reg_mae = [round(total / len(test_loader), 5) for total in total_reg_mae]
        avg_reg_mape = [round(total / len(test_loader), 5) for total in total_reg_mape]
        avg_accuracy = cls_metric.compute()
        
        print(f"Average Accuracy: {avg_accuracy}")
        print(f'Test Loss: {avg_reg_loss}, Test MAE: {avg_reg_mae}, Test MAPE: {avg_reg_mape}')

        # wandb.log({"Average Accuracy": total_train_accuracy})
        # wandb.log({"Test Loss": avg_reg_loss, "Test MAE": avg_reg_mae})

        cls_metric.reset()
        mae_metric.reset()
        mape_metric.reset()
        
        return reg_list, labels_reg_list

    # %%
    test_model = train_model
    # test_loader = DataLoader(test_dataloader, batch_size=128)
    reg_criterion = nn.L1Loss()
    cls_criterion = nn.BCELoss()

    output = test(test_model, reg_criterion, cls_criterion)
    pred = np.array(output[0])
    label = np.array(output[1])