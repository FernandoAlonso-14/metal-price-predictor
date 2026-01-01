'''
Version 13: only one GAT (one sector) predict multiple outputs
Version 12: only one GAT (one sector) predict one target
Driver Version: 581.57         CUDA Version: 13.0     CUDA 13.0 Update 2
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool , global_max_pool
from torch_geometric.data import Data, Batch

def build_edge_index(batch_size, num_nodes):
    row = torch.arange(num_nodes).view(-1, 1).repeat(1, num_nodes).view(-1)
    col = torch.arange(num_nodes).view(-1, 1).repeat(num_nodes, 1).view(-1)
    edge_index = torch.stack([row, col], dim=0)
    edge_index_list = [edge_index for _ in range(batch_size)]
    return edge_index_list

def build_graph_batch(x_list, edge_index_list):
    data_list = []
    for graph_nodes, ei in zip(x_list, edge_index_list):
        data_list.append(Data(x=graph_nodes, edge_index=ei))
    graph = Batch.from_data_list(data_list)
    return graph

class AttentionBlock(nn.Module):
    def __init__(self,time_step,dim):
        super(AttentionBlock, self).__init__()
        self.attention_matrix = nn.Linear(time_step, time_step)
    
    def forward(self, inputs):
        '''
        inp : torch.tensor (batch,time_step,input_dim)
        '''
        inputs_t = torch.transpose(inputs,2,1) # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight,dim=-1)
        attention_probs = torch.transpose(attention_probs,2,1) 
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec,dim=1) # (batch_size, input_dim)
        return attention_vec, attention_probs

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.GRU(input_size=self.input_dim,
                              hidden_size=hidden_dim,
                              num_layers=2,
                              dropout=0.5,
                              batch_first=True)
        
        self.attention_block = AttentionBlock(time_step,hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.dim = hidden_dim

    def forward(self,seq):
        '''
        seq : torch.tensor (batch,time_step,input_dim)
        '''
        self.encoder.flatten_parameters()
        seq_vector,_ = self.encoder(seq)
        seq_vector = self.dropout(seq_vector) # (batch, time_step, hidden_dim)
        
        # weekly attention for each stock & each week
        attention_vec, _ = self.attention_block(seq_vector) # (batch, hidden_dim)
        # attention_vec = attention_vec.view(-1,1,self.dim) # prepare for concat(128, 1, 16)
        return attention_vec

    
class BaseModule(nn.Module):
    def __init__(self, input_dim,indi_dim, indi_num, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device):
        super(BaseModule, self).__init__()
        self.input_num = input_num
        self.input_dim = input_dim
        self.indi_dim = indi_dim
        self.indi_num = indi_num
        self.time_step = time_step
        self.week_num = week_num
        self.pred_weeks = pred_weeks
        self.stock_edge = stock_edge
        self.all_edge = all_edge
        self.device = device
        self.dim = hidden_dim

''' Learning From Assets '''
class WeeklyModule(BaseModule):
    def __init__(self,input_dim,time_step,hidden_dim,stock_edge,all_edge,week_num,pred_weeks,input_num,device):
        super(WeeklyModule, self).__init__(input_dim, None,None, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.weekly_attention = AttentionBlock(self.week_num, self.dim)
        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim, self.time_step, self.dim) for _ in range(self.input_num)])
        self.inner_gat = GATConv(self.dim, self.dim) # select 1 GAT for single category

    def forward(self, train_features):
        # train_features: [batch, input_num, week_num, days, features] [128, 5, 4, 5, 10]

        # Time: ai
        encoded_weeks_list = []
        for i in range(self.week_num):
            encoded_data_list = []
            for j in range(self.input_num):
                week_data = train_features[:, j, i, :, :] # [batch, days, features]
                # weekly encoder (attentive GRU -> a_i)
                encoded_week = self.encoder_list[j](week_data) # a_i: [batch, hidden_dim]
                encoded_data_list.append(encoded_week) #stock/ week 's embedding

            encoded_weeks_list.append(torch.stack(encoded_data_list, dim=1)) # a_i,...a_i+n [batch, stock_num, hidden_dim]
        
        encoded_data_tensor = torch.stack(encoded_weeks_list, dim=1)  # all stocks/weeks a_i: [batch, week, stock_num, hidden_dim]

        # Graph: all stocks/ each week
        graph_embeddings = []
        edge_index_list = build_edge_index(encoded_data_tensor.shape[0], self.input_num)

        for week in range(self.week_num):
            x_list = torch.split(encoded_data_tensor[:, week, :, :], 1, dim=0)
            x_list = [x.squeeze(0) for x in x_list]
            graph = build_graph_batch(x_list, edge_index_list).to(self.device)
            graph_encoded = self.inner_gat(graph.x, graph.edge_index).to(self.device)
            graph_encoded = graph_encoded.view(-1, self.input_num, self.dim) # [batch, stock_num, hidden_dim]
            graph_embeddings.append(graph_encoded)

        encoded_graph_tensor = torch.stack(graph_embeddings, dim=1) # [batch, week, stock_num, hidden_dim]

        # Attention
        week_atten_encoded, graph_atten_encoded = [], []

        # merge weeks -> monthly: T_ai [batch, hidden_dim]
        for stock_idx in range(self.input_num):
            encoded_stock = encoded_data_tensor[:, :, stock_idx, :]
            encoded_graph = encoded_graph_tensor[:, :, stock_idx, :]

            weekly_att_vector, _ = self.weekly_attention(encoded_stock)
            week_atten_encoded.append(weekly_att_vector)

            # merge graph -> monthly: T_gi [batch, hidden_dim]
            graph_att_vector, _ = self.weekly_attention(encoded_graph)
            graph_atten_encoded.append(graph_att_vector)

        # merge nodes -> T_ai [batch, stock_num, hidden_dim]
        weeks_atten_tensor = torch.stack(week_atten_encoded, dim=1).to(self.device)
        graph_atten_tensor = torch.stack(graph_atten_encoded, dim=1).to(self.device)
        # print("weeks_atten_tensor: ", weeks_atten_tensor.shape)

        # return tensor (ai): merge weeks
        encoded_data_month = encoded_data_tensor.view(-1, self.week_num*self.input_num, self.dim) # [batch, stock_num*week_num, hidden_dim]
        
        return weeks_atten_tensor, graph_atten_tensor, encoded_data_month

''' Dynamic Correlation Encoder: Transformer Encoder version (used in the final model) '''
class TransModule(BaseModule):
    def __init__(self,input_dim,time_step,hidden_dim,stock_edge,all_edge,week_num,pred_weeks,input_num,device):
        super(TransModule, self).__init__(input_dim, None,None, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.weekly_attention = AttentionBlock(self.week_num, self.dim)
        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim, self.time_step, self.dim) for _ in range(self.input_num)])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=8,
                                                        dim_feedforward=self.dim*2,
                                                        dropout=0.2,
                                                        batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=8, dropout=0.2, batch_first=True)

    def forward(self, train_features):
        # train_features: [batch, input_num, week_num, days, features] [128, 5, 4, 5, 10]

        # Time: ai
        encoded_weeks_list = []
        for i in range(self.week_num):
            encoded_data_list = []
            for j in range(self.input_num):
                week_data = train_features[:, j, i, :, :] # [batch, days, features]
                # weekly encoder (attentive GRU -> a_i)
                encoded_week = self.encoder_list[j](week_data) # a_i: [batch, hidden_dim]
                encoded_data_list.append(encoded_week) #stock/ week 's embedding

            encoded_weeks_list.append(torch.stack(encoded_data_list, dim=1)) # a_i,...a_i+n [batch, stock_num, hidden_dim]
        
        encoded_data_tensor = torch.stack(encoded_weeks_list, dim=1)  # all stocks/weeks a_i: [batch, week, stock_num, hidden_dim]

        # Transformer encoding for each week
        trans_embedding = []
        attention_weights_list = []
        for week in range(self.week_num):
            encoded_week_data = encoded_data_tensor[:, week, :, :] # [batch, stock_num, hidden_dim]
            trans_encoded, attn_weights = self.multihead_attn(encoded_week_data, encoded_week_data, encoded_week_data)
            trans_embedding.append(trans_encoded)
            attention_weights_list.append(attn_weights)

        encoded_trans_tensor = torch.stack(trans_embedding, dim=1) # [batch, week, stock_num, hidden_dim]      
        attention_weights_tensor = torch.stack(attention_weights_list, dim=1) # [batch, week, stock_num, stock_num]

        # Attention
        week_atten_encoded, trans_atten_encoded = [], []

        # merge weeks -> monthly: T_ai [batch, hidden_dim]
        for stock_idx in range(self.input_num):
            encoded_stock = encoded_data_tensor[:, :, stock_idx, :]
            encoded_trans = encoded_trans_tensor[:, :, stock_idx, :]

            weekly_att_vector, _ = self.weekly_attention(encoded_stock)
            week_atten_encoded.append(weekly_att_vector)

            # merge trans -> monthly: T_ti [batch, hidden_dim]
            trans_att_vector, _ = self.weekly_attention(encoded_trans)
            trans_atten_encoded.append(trans_att_vector)

        # merge nodes -> T_ai [batch, stock_num, hidden_dim]
        weeks_atten_tensor = torch.stack(week_atten_encoded, dim=1).to(self.device)
        trans_atten_tensor = torch.stack(trans_atten_encoded, dim=1).to(self.device)
        # print("trans_atten_tensor: ", trans_atten_tensor.shape)
        
        return weeks_atten_tensor, trans_atten_tensor, attention_weights_tensor

''' Learning From Macro Indicators '''
class MonthlyModule(BaseModule):
    def __init__(self,indi_dim,indi_num,time_step,hidden_dim,stock_edge,all_edge,week_num,pred_weeks,input_num,device):
        super(MonthlyModule, self).__init__(None, indi_dim, indi_num, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.indi_num = indi_num # 1 (all indicator combined into 1)
        self.indi_dim = indi_dim
        self.weekly_attention = AttentionBlock(self.week_num, self.dim)
        self.encoder_list = SequenceEncoder(self.indi_dim, self.time_step, self.dim)

    def forward(self, indi_data):
        # indi_dict: [batch, indi_num, week, time_step, indi_input_dim] [256, 1, 4, 5, 20]
        # Attention
        indi_weeks_list = []
        for week in range(self.week_num):

            week_indi = indi_data[:, 0, week, :, :] # [batch, time_step, indi_dim]
            encoded_week = self.encoder_list(week_indi) # a_i: [batch, hidden_dim]
            indi_weeks_list.append(encoded_week)

        encoded_indi_tensor = torch.stack(indi_weeks_list, dim=1) # [batch, week, hidden_dim]

        # merge weeks -> monthly: T_ai [batch, hidden_dim]
        weekly_att_vector, _ = self.weekly_attention(encoded_indi_tensor) # [batch, hidden_dim]
        weekly_att_tensor = weekly_att_vector.unsqueeze(1) # [batch, 1, hidden_dim]

        return weekly_att_tensor.to(self.device)

'''Dynamic Correlation Encoder: GAT version (didn't use in the final model)'''
class GraphAttention(BaseModule):
    def __init__(self,input_dim,indi_dim,indi_num,time_step,hidden_dim,stock_edge,all_edge,week_num,pred_weeks,input_num,device):
        super(GraphAttention, self).__init__(input_dim,indi_dim, indi_num, time_step, hidden_dim, stock_edge,all_edge, week_num, pred_weeks, input_num, device)

        self.WeeklyModule = WeeklyModule(input_dim, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.MonthlyModule = MonthlyModule(indi_dim, indi_num, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.fusion = nn.Linear(hidden_dim*3, hidden_dim)
        self.reg_layer = nn.Linear(hidden_dim, week_num)
        self.cls_layer = nn.Linear(hidden_dim, week_num)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, train_features, indi_dict):
        # indi_dict: [batch, indi_num, week, days, features]

        weeks_atten, graph_atten, encoded_month = self.WeeklyModule(train_features) # [batch, stock_num, hidden_dim]
        indi_atten = self.MonthlyModule(indi_dict)  # [batch, hidden_dim]
        # indi_atten = indi_atten.unsqueeze(1) # [batch, 1, hidden_dim]

        # fusion
        stock_fusion = torch.cat((weeks_atten, graph_atten),dim=-1) # [batch, stock_num, 2*hidden_dim]
        indi_atten_repeated = indi_atten.repeat(1, self.input_num, 1) # [batch, stock_num, hidden_dim]
        fusion_vec = torch.cat((stock_fusion, indi_atten_repeated), dim=-1) # [batch, stock_num, 3*hidden_dim]
        fusion_vec = torch.relu(self.fusion(fusion_vec)) # [batch, stock_num, hidden_dim]
        # fusion_vec = self.dropout(fusion_vec)

        # residual connection
        # result = fusion_vec
        result = fusion_vec + weeks_atten

        # output: [batch, stock, week]
        reg_output = self.reg_layer(result)
        cls_output = torch.sigmoid(self.cls_layer(result))

        return reg_output, cls_output

class TimeAttention(BaseModule):
    'without Asset Correlation'
    def __init__(self,input_dim,indi_dim,indi_num,time_step,hidden_dim,stock_edge,all_edge,week_num, pred_weeks,input_num,device):
        super(TimeAttention, self).__init__(input_dim,indi_dim, indi_num, time_step, hidden_dim, stock_edge,all_edge, pred_weeks, week_num, input_num, device)

        self.WeeklyModule = WeeklyModule(input_dim, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.MonthlyModule = MonthlyModule(indi_dim, indi_num, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)
        self.reg_layer = nn.Linear(hidden_dim, week_num)
        self.cls_layer = nn.Linear(hidden_dim, week_num)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, train_features, indi_dict):
        # indi_dict: [batch, indi_num, week, days, features]

        weeks_atten, graph_atten, encoded_month = self.WeeklyModule(train_features) # [batch, stock_num, hidden_dim]
        indi_atten = self.MonthlyModule(indi_dict)  # [batch, hidden_dim]
        indi_atten_repeated = indi_atten.repeat(1, self.input_num, 1) # [batch, stock_num, hidden_dim]

        fusion_vec = torch.cat((weeks_atten, indi_atten_repeated), dim=-1) # [batch, stock_num, 3*hidden_dim]
        fusion_vec = torch.relu(self.fusion(fusion_vec)) # [batch, stock_num, hidden_dim]
        fusion_vec = self.dropout(fusion_vec)


        # fusion
        fusion_vec = torch.cat((weeks_atten, graph_atten),dim=-1) # [batch, stock_num, 2*hidden_dim]
        # indi_atten_repeated = indi_atten.repeat(1, self.input_num, 1) # [batch, stock_num, hidden_dim]
        # fusion_vec = torch.cat((weeks_atten, indi_atten_repeated), dim=-1) # [batch, stock_num, 3*hidden_dim]
        fusion_vec = torch.relu(self.fusion(fusion_vec)) # [batch, stock_num, hidden_dim]
        # residual connection
        result = fusion_vec + weeks_atten

        # output: [batch, stock, week]
        reg_output = self.reg_layer(result)
        cls_output = torch.sigmoid(self.cls_layer(result))

        return reg_output, cls_output

'''Main Model'''
class TransAttention(BaseModule):
    def __init__(self,input_dim,indi_dim,indi_num,time_step,hidden_dim,stock_edge,all_edge,week_num, pred_weeks,input_num,device):
        super(TransAttention, self).__init__(input_dim,indi_dim, indi_num, time_step, hidden_dim, stock_edge,all_edge, week_num, pred_weeks, input_num, device)

        self.TransModule = TransModule(input_dim, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.MonthlyModule = MonthlyModule(indi_dim, indi_num, time_step, hidden_dim, stock_edge, all_edge, week_num, pred_weeks, input_num, device)
        self.fusion = nn.Linear(hidden_dim*3, hidden_dim)
        self.reg_layer = nn.Linear(hidden_dim, pred_weeks)
        self.cls_layer = nn.Linear(hidden_dim, pred_weeks)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, train_features, indi_dict):
        # indi_dict: [batch, indi_num, week, days, features]

        weeks_atten, trans_atten, atten_weight = self.TransModule(train_features) # [batch, stock_num, hidden_dim]
        indi_atten = self.MonthlyModule(indi_dict)  # [batch, 1, hidden_dim]

        # fusion
        stock_fusion = torch.cat((weeks_atten, trans_atten),dim=-1) # [batch, stock_num, 2*hidden_dim]
        indi_atten_repeated = indi_atten.repeat(1, self.input_num, 1) # [batch, stock_num, hidden_dim]
        fusion_vec = torch.cat((stock_fusion, indi_atten_repeated), dim=-1) # [batch, stock_num, 3*hidden_dim]
        fusion_vec = torch.relu(self.fusion(fusion_vec)) # [batch, stock_num, hidden_dim]
        fusion_vec = self.dropout(fusion_vec)

        # residual connection
        # result = fusion_vec
        result = fusion_vec + weeks_atten

        # output: [batch, stock, week]
        reg_output = self.reg_layer(result)
        cls_output = torch.sigmoid(self.cls_layer(result))

        return reg_output, cls_output, atten_weight