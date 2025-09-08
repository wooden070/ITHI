import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.append("..")
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from src import utils
from tqdm import tqdm
class re_rank_model(nn.Module):
    def __init__(self, in_features, out_features):
        super(re_rank_model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(self.in_features, self.out_features)
        self.fc2 = nn.Linear(self.out_features, 1)

        
    def forward(self, input_tensor, gold_tensor):
        x = torch.cat((input_tensor, gold_tensor), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output importance score between 0 and 1
        return x.squeeze(-1)

def evaluate_model(model, gold_tensor, batch_tensor, get_score):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(len(gold_tensor)):
            predicted_scores = model(batch_tensor[i], gold_tensor[i])
            predicted_scores = predicted_scores *3
            loss = F.mse_loss(predicted_scores, get_score[i])
            total_loss += loss.item()
    return total_loss / len(gold_tensor)

def read_pickle(batch_tensor_pth):    
    with open(batch_tensor_pth, 'rb') as file:
        batch_tensor_dict = pickle.load(file) 
    return batch_tensor_dict

def get_batch_tensor(top_N_indices_all, all_history_back_cont_embed, train_sample_num, find_cross_day):
    # 重新加载batch_tensor  不保存  只保存索引
    history_back_cont_tmp = all_history_back_cont_embed[all_history_back_cont_embed['timid'] < train_sample_num]
    history_all_embed = history_back_cont_tmp[train_sample_num-find_cross_day < history_back_cont_tmp['timid']]
    history_Cont_emb = np.vstack(history_all_embed['Cont_embed'].values)
    history_Head_entity_emb = np.vstack(history_all_embed['Head_entity_embed'].values)
    history_Relation_emb = np.vstack(history_all_embed['Relation_embed'].values)
    
    # 将背景嵌入和关系嵌入拼接  [总 x 1024*3]
    history_concat_embeds = np.hstack((history_Cont_emb, history_Head_entity_emb, history_Relation_emb))
    
    
    top_N_indices_all = np.array(top_N_indices_all)
    # 计算 batch_size 和每个 batch 的大小
    batch_size = len(top_N_indices_all)
    num_indices = len(top_N_indices_all[0])

    # 创建一个空的 numpy 数组，用于存储结果
    batch_array = np.empty((batch_size, num_indices, history_concat_embeds.shape[1]), dtype=history_concat_embeds.dtype)

    # 利用 numpy 的高级索引技术进行批量提取
    for i, indices in enumerate(top_N_indices_all):
        batch_array[i] = history_concat_embeds[indices]
    
    # 将 numpy 数组转换为 torch.Tensor
    batch_tensor = torch.tensor(batch_array)
    return batch_tensor

def do_train(args):
        # 加载re-rank模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = re_rank_model(in_features=768*6, out_features = 768).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    data_pth = "../data/" + args.dataset + "/train_re_rank_data"
    back_cont_embed_path = '../data/' + args.dataset
    all_history_back_cont_embed = pd.read_pickle(back_cont_embed_path + '/all_history_back_cont_embed.pkl')
    
        
    save_data_pth = data_pth
    gold_tensor_dict = {}
    get_score_dict = {}
    top_N_indices_all = {}

    
    for i in tqdm(range(args.split_flag)):
        # (all_history_back_cont_embed, train_sample_num, num_rels, find_cross_day, top_n):
        print("---------------------", i, "---------------------")
        with open(save_data_pth + '/gold_tensor_' + str(i+1) + '.pkl', 'rb') as f:
            gold_tensor = pickle.load(f)  # 以天为单位的图
            gold_tensor_dict = {**gold_tensor_dict, **gold_tensor}
            

        with open(save_data_pth + '/get_score_' + str(i+1) + '.pkl', 'rb') as f:
            get_score = pickle.load(f)  # 以天为单位的图
            get_score_dict = {**get_score_dict, **get_score}

        with open(save_data_pth + '/top_N_indices_' + str(i+1) + '.pkl', 'rb') as f:
            top_N_indices = pickle.load(f)  # 以天为单位的图
            top_N_indices_all = {**top_N_indices_all, **top_N_indices}
            

        
        
        
        
    # 天数
    data = utils.load_data(args.dataset)  # 加载train、valid、test 三种数据（四元组
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    print(num_nodes, num_rels)  # 2594 225

    train_times = np.array(sorted(set(data.train[:, 3])))
        # 加载 pkl 包含背景嵌入
    all_time = len(train_times)
    # 原
    train_time = all_time - 100
    # train_time = all_time - 10
    valid_time = all_time
    best_loss = float('inf')
    print("----------------------Start Training---------------------------")
    for epoch in range(args.epoch):
        print(f'Epoch [{epoch+1}/{args.epoch}]')
        model.train()
        
        for i in tqdm(range(20, train_time)):
            total_loss = 0.0
            with torch.no_grad():
                gold_tensor = gold_tensor_dict[i].float().to(device)
                # 重构加载batch_tensor 方式
                # batch_tensor = read_pickle(data_pth + '/batch_tensor/batch_tensor_'+str(i)+'.pkl').to(device)
                get_batch = get_batch_tensor(top_N_indices_all[i], all_history_back_cont_embed, i, 30)
                # get_batch = tensor.type
                batch_tensor = get_batch.to(device)
                
                get_score = get_score_dict[i].float().to(device)
                gold_tensor_expanded = gold_tensor.expand(-1, batch_tensor.shape[1], -1)  # Dimension becomes (190, 100, 2048)

            
            for item in range(gold_tensor.shape[0]):
                # Predict scores
                predicted_scores = model(batch_tensor[item], gold_tensor_expanded[item])
                predicted_scores = predicted_scores *3
                loss = F.mse_loss(predicted_scores, get_score[item])
                # total_loss += loss.item()
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if i %200 == 0:
                total_loss = 0.0
                for val in tqdm(range(train_time, valid_time)):
                    val_gold_tensor = gold_tensor_dict[val].float().to(device)
                    
                    val_get_batch = get_batch_tensor(top_N_indices_all[val], all_history_back_cont_embed, val, 30)
                    val_batch_tensor = val_get_batch.to(device)
                    
                    val_get_score = get_score_dict[val].float().to(device)
                    
                    val_gold_tensor_expanded = val_gold_tensor.expand(-1, val_batch_tensor.shape[1], -1)  # Dimension becomes (190, 100, 2048)
                    
                    test_loss = evaluate_model(model, val_gold_tensor_expanded, val_batch_tensor, val_get_score)
                    total_loss += test_loss
                if best_loss == 0 or total_loss < best_loss:
                    best_loss = test_loss
                    
                    torch.save(model.state_dict(), data_pth+'/re-rank_model')
            # print(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='re_rank_model_train')
    parser.add_argument("-d", "--dataset", type=str, default='EG',
                        help="which country's dataset to use: EG/ IR/ IS")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--split_flag", type=int, default=7)    
    args = parser.parse_args()
    print(args)

    do_train(args)
    sys.exit()
    

# python re_rank_model.py --dataset IS
