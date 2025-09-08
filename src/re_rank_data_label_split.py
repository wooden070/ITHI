import argparse
import os
import sys
import pickle

import faiss
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
sys.path.append("..")
from src import utils
from multiprocessing import Pool, cpu_count
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from re_rank_model import re_rank_model

import warnings
warnings.filterwarnings("ignore")
debug_pth = "cross_day_graphs_debug"


def find_top_n_events(all_history_back_cont_embed, train_sample_num, num_rels, find_cross_day, top_n):

    history_back_cont_tmp = all_history_back_cont_embed[all_history_back_cont_embed['timid'] < train_sample_num]
    history_all_embed = history_back_cont_tmp[train_sample_num-find_cross_day < history_back_cont_tmp['timid']]
    
    current_all_embed = all_history_back_cont_embed[all_history_back_cont_embed['timid'] == train_sample_num]
                            # tqdm(data_df.iterrows(), total=len(data_df))

    # 获取当天代检索向量  
    Current_Cont_emb = np.vstack(current_all_embed['Cont_embed'].values)
    Current_Head_entity_emb = np.vstack(current_all_embed['Head_entity_embed'].values)
    Current_Relation_emb = np.vstack(current_all_embed['Relation_embed'].values)

    current_concat_embed = np.hstack((Current_Cont_emb, Current_Head_entity_emb, Current_Relation_emb))


    # 历史表示
    history_Cont_emb = np.vstack(history_all_embed['Cont_embed'].values)
    history_Head_entity_emb = np.vstack(history_all_embed['Head_entity_embed'].values)
    history_Relation_emb = np.vstack(history_all_embed['Relation_embed'].values)
    
    # 将背景嵌入和关系嵌入拼接  [总 x 1024*3]
    history_concat_embeds = np.hstack((history_Cont_emb, history_Head_entity_emb, history_Relation_emb))


    # 9.15 ++ 提高效率
    # 使用Faiss构建索引
    d = history_concat_embeds.shape[1]  # 向量维度
    index = faiss.IndexFlatL2(d)  # 使用L2距离构建索引
    index.add(history_concat_embeds.astype('float32'))  # 将历史嵌入添加到索引中
    # 批量进行相似度检索
    current_concat_embed = current_concat_embed.astype('float32')
    distances, top_N_indices_all_fast = index.search(current_concat_embed, top_n)  # top_n为检索个数
  
    # 收集相应的事件信息
    top_N_events_all = []
    for top_N_indices in top_N_indices_all_fast:
        top_N_events = []
        for idx in top_N_indices:
            actor1 = history_all_embed.iloc[idx]['Actor1Name']
            relation = history_all_embed.iloc[idx]['EventCode']
            actor2 = history_all_embed.iloc[idx]['Actor2Name']
            timid = history_all_embed.iloc[idx]['timid']
            top_N_events.append((actor1, relation, actor2, timid))
        top_N_events_all.append(top_N_events)
    
    # 返回当前嵌入和top N索引
    gold_tensor = torch.tensor(current_concat_embed)



    return top_N_events_all, gold_tensor, list(top_N_indices_all_fast)
    """
        重新加载batch_tensor  不保存  只保存索引
        batch_list = []
        for i, top_N_indices in enumerate(top_N_indices_all):
            find_triplets_tensor = history_concat_embeds[top_N_indices]
            batch_list.append(find_triplets_tensor)
        batch_tensor = torch.tensor(batch_list)   
        
    """
def find_repeat_n_events(all_history_back_cont_embed, train_sample_num, find_cross_day, top_n):

    history_back_cont_tmp = all_history_back_cont_embed[all_history_back_cont_embed['timid'] < train_sample_num]
    history_all_embed = history_back_cont_tmp[train_sample_num-find_cross_day < history_back_cont_tmp['timid']]
    
    current_all_embed = all_history_back_cont_embed[all_history_back_cont_embed['timid'] == train_sample_num]
                            # tqdm(data_df.iterrows(), total=len(data_df))

    # 获取当天代检索向量  
    Current_head = np.vstack(current_all_embed['Actor1Name'].values)
    Current_relation = np.vstack(current_all_embed['EventCode'].values)
    Current_tail = np.vstack(current_all_embed['Actor2Name'].values)
    Current_time = np.vstack(current_all_embed['timid'].values)

    current_triplets = np.hstack((Current_head, Current_relation, Current_tail, Current_time))


    # 历史表示
    history_head = np.vstack(history_all_embed['Actor1Name'].values)
    history_relation = np.vstack(history_all_embed['EventCode'].values)
    history_tail = np.vstack(history_all_embed['Actor2Name'].values)
    history_time = np.vstack(history_all_embed['timid'].values)
    
    # 将背景嵌入和关系嵌入拼接  [总 x 1024*3]
    history_triplets = np.hstack((history_head, history_relation, history_tail, history_time))


    # 存储结果的列表
    matched_rows_list = []

    # 对 current_triplets 的每一行进行处理
    for current in current_triplets:
        # 提取 current 的前两列
        first_col, second_col = current[0], current[1]
        
        # 优先匹配前两列都匹配的行
        primary_mask = (history_triplets[:, 0] == first_col) & (history_triplets[:, 1] == second_col)
        primary_matches = history_triplets[primary_mask]
        
        if primary_matches.size > 0:
            # 如果有匹配，则添加这些行到结果中
            if primary_matches.shape[0]>top_n:
                primary_matches = primary_matches[-top_n:]
            matched_rows_list.append(primary_matches)
        else:
            # 如果没有前两列匹配的行，则匹配第一列
            secondary_mask = (history_triplets[:, 0] == first_col)
            secondary_matches = history_triplets[secondary_mask]
            if secondary_matches.size > 0:
                if secondary_matches.shape[0]>top_n:
                    secondary_matches = secondary_matches[-top_n:]
                matched_rows_list.append(secondary_matches)
            else:
                matched_rows_list.append([])

    return matched_rows_list



def get_bi_current(current_tri, num_rel):
    inverse_triples = current_tri[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rel

    return np.concatenate((current_tri, inverse_triples))

def bool_direct_answer(expect, related):
    if related[0]==expect or related[2]==expect:
        return True
    else:
        return False

def is_in_values(value, dict_values):
    return any(value in s for s in dict_values)

def bool_1_hop_answer(expect, related, c_all_ans):
    hop_1_dict = c_all_ans[related[0]].values()
    hop_1_dict_reverse = c_all_ans[related[2]].values()
    
    if is_in_values(expect, hop_1_dict) or is_in_values(expect, hop_1_dict_reverse):
        return True
    else:
        return False

def bool_2_hop_answer(expect, related, c_all_ans, head_ents):
    hop_1_dict = c_all_ans[related[0]].values()
    hop_1_dict_reverse = c_all_ans[related[2]].values()
        
    unique_nodes = []
    for value_set in hop_1_dict:
        # 遍历集合中的每个元素
        for element in value_set:
            # 在这里可以处理每个元素
            # 例如，这里我们只是打印它
            if element not in unique_nodes:
                unique_nodes.append(element)
    
    for value_set in hop_1_dict_reverse:
        # 遍历集合中的每个元素
        for element in value_set:
            # 在这里可以处理每个元素
            # 例如，这里我们只是打印它
            if element not in unique_nodes:
                unique_nodes.append(element)
                
    cleaned_list = [x for x in unique_nodes if x not in head_ents]
    

    for node in cleaned_list:
        if is_in_values(expect, c_all_ans[node].values()):
            return True
    return False

def get_current_score(related_event, gold, all_ans_list_train, head_ents):
    score_list = []
    expect = gold[2]
    for i, related in enumerate(related_event):
        c_all_ans = all_ans_list_train[related[3]]
        
        if bool_direct_answer(expect, related):
            score_list.append(3)
        elif bool_1_hop_answer(expect, related, c_all_ans):
            score_list.append(2)
        elif bool_2_hop_answer(expect, related, c_all_ans, head_ents):
            score_list.append(1)  # 2-hop
        else:
            score_list.append(0)  # Unrelated`
    return score_list



def calcute_relate_score(related_events, gold_events, all_ans_list_train, head_ents):
    final_score = []
    for i, gold in enumerate(gold_events):
        score = get_current_score(related_events[i], gold, all_ans_list_train, head_ents)
        final_score.append(score)
    return final_score

def save_pkl(save_data_pth, i, batch_tensor):
    with open(save_data_pth + '/batch_tensor_'+str(i)+'.pkl', 'wb') as file:
        pickle.dump(batch_tensor, file)


def deal_data(args):
    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)  # 加载train、valid、test 三种数据（四元组
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    print(num_nodes, num_rels)  # 2594 225

    print("loading popularity bias data")
    head_ents = json.load(open('../data/'+args.dataset+'/head_ents.json', 'r'))


    # data for time-aware filtering  3.26 +++++ 按天、按sro及or's 转为dict
    all_ans_list_train = utils.load_all_answers_for_time_filter(data.train, num_rels, num_nodes)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes)
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes)
    all_ans_list = all_ans_list_train + all_ans_list_valid + all_ans_list_test
    # all_ans_list_train[0]
    # all_times = np.array(sorted(set(data.train[:, 3])))
    all_times = np.array(sorted(set(data.test[:, 3])))
        # 加载 pkl 包含背景嵌入
    back_cont_embed_path = '../data/' + args.dataset
    all_history_back_cont_embed = pd.read_pickle(back_cont_embed_path + '/all_history_back_cont_embed.pkl')
    all_train_time = all_times[-1] + 1
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    
    all_list = train_list + valid_list + test_list
    train_history_cont = all_history_back_cont_embed


    save_data_pth = "../data/" + args.dataset + "/train_re_rank_data"
    if not os.path.exists(save_data_pth):
        os.makedirs(save_data_pth)
    # 从第三十天开始算起  构造数据
    related_top100_dict = {}
    gold_tensor_dict = {}
    get_score_dict = {}
    top_N_indices_all_dict = {}
    repeat_history_dict = {}
    

    
    for i in tqdm(range(args.strat_N_time, args.end_N_time)):
        # (all_history_back_cont_embed, train_sample_num, num_rels, find_cross_day, top_n):
        print("---------------------", i, "---------------------")
        """"""
        related_top100_events, gold_tensor, batch_index_tensor = find_top_n_events(train_history_cont, i, num_rels, 30, 100)
        current_gold_event = get_bi_current(all_list[i], num_rels)
        get_score = calcute_relate_score(related_top100_events, current_gold_event, all_ans_list, head_ents)
        get_score = torch.tensor(get_score)
        gold_tensor = gold_tensor.unsqueeze(1)
        
        repeat_history = find_repeat_n_events(train_history_cont, i, 30, 100)
        repeat_history_dict[i] = repeat_history
        
        related_top100_dict[i] = related_top100_events
        gold_tensor_dict[i] = gold_tensor
        top_N_indices_all_dict[i] = batch_index_tensor
        get_score_dict[i] = get_score
        

        
    with open(save_data_pth + '/related_top100_' + str(args.split_flag) + '.pkl', 'wb') as file:
        pickle.dump(related_top100_dict, file)
    with open(save_data_pth + '/gold_tensor_' + str(args.split_flag) + '.pkl', 'wb') as file:
        pickle.dump(gold_tensor_dict, file) 
    with open(save_data_pth + '/get_score_' + str(args.split_flag) + '.pkl', 'wb') as file:
        pickle.dump(get_score_dict, file)   
    with open(save_data_pth + '/top_N_indices_' + str(args.split_flag) + '.pkl', 'wb') as file:
        pickle.dump(top_N_indices_all_dict, file)        
    with open(save_data_pth + '/repeat_history_' + str(args.split_flag) + '.pkl', 'wb') as file:
        pickle.dump(repeat_history_dict, file)
        
    # with open(save_data_pth + '/repeat_history_all.pkl', 'wb') as file:
    #     pickle.dump(repeat_history_dict, file)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='re_rank_data_label')
    parser.add_argument("-d", "--dataset", type=str, default='EG',
                        help="which country's dataset to use: EG/ IR/ IS")
    parser.add_argument("--strat_N_time", type=int, required=True, 
                        help="which country's dataset to use: EG/ IR/ IS")
    parser.add_argument("--end_N_time", type=int, required=True,
                        help="which country's dataset to use: EG/ IR/ IS")
    parser.add_argument("--split_flag", type=int, required=True,
                        help="which country's dataset to use: EG/ IR/ IS")
    
    args = parser.parse_args()
    print(args)

    deal_data(args)
    sys.exit()
    

