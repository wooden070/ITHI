import numpy as np
import torch
import os
import pickle
import src.knowledge_graph as knwlgrh
import pandas as pd
import logging
from tqdm import tqdm
from collections import defaultdict
import dgl
from torch.nn.utils.rnn import pad_sequence

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))  # 例如 indices[6]=56，说明第6个三元组预测的尾实体是56个，so预测错误，indices[6]=0 才算正确  返回一个 [n, 2(分别表第几个三元组、预测尾实体索引)]
    indices = indices[:, 1].view(-1)  # 取出列索引，也就是预测后尾节点倒序后排在多少位
    return indices
# t天所有三元组(n,3), e1+r 预测后的所有尾实体特征(n, 2594), t天转换后的标签(sro,or's)
def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #  这玩意是不是有点子问题？
    return score

# 传入 [t天所有三元组(n,3), e1+r 预测后的所有尾实体特征(n, 2594), t天转换后的标签(sro,or's), eval_bz]
def get_total_rank(test_triples, score, all_ans, eval_bz):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]  # 按照batch 统一处理为batch

        target = test_triples[batch_start:batch_end, 2]

        filter_score_batch = filter_score(triples_batch, score_batch, all_ans)  # 不理解，感觉返回的还是score_batch
        filter_rank.append(sort_and_rank(filter_score_batch, target))  # 预测后尾节点倒序后排在多少位

    filter_rank = torch.cat(filter_rank)  # [n] 预测后尾节点倒序后排在多少位
    filter_rank += 1 # change to 1-indexed
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    # MRR度量了模型在排名任务中的性能  衡量了模型在给定查询的情况下，对于正确答案的平均排名的倒数
    return filter_mrr.item(), filter_rank


def get_filtered_score(test_triples, score, all_ans, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    filtered_score = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        filtered_score.append(filter_score(triples_batch, score_batch, all_ans))
    filtered_score = torch.cat(filtered_score,dim =0)

    return filtered_score


def popularity_map(tuple_tensor, head_ents):
    tag = 'head' if tuple_tensor[2].item() in head_ents else 'other'
    return tag

# 传入 [预测后尾节点倒序后排在多少位, 统计有多少个三元组中包含常见头结点, mode]
def cal_ranks(rank_list, tags_all, mode):
    total_tag_all = []
    hits = [1, 3, 10]
    rank_list = torch.cat(rank_list)
    for tag_all in tags_all:
        total_tag_all += tag_all  # 汇总所有天
    # 去除常见节点，得到去偏后的mrr指标
    all_df = pd.DataFrame({'rank_ent': rank_list.cpu(), 'ent_tag': total_tag_all})
    debiased_df = all_df[all_df['ent_tag'] != 'head']
    debiased_rank_ent = torch.tensor(list(debiased_df['rank_ent']))
    mrr_debiased = torch.mean(1.0 / debiased_rank_ent.float())

    if mode == 'test':
        logging.info("====== object prediction ======")
        logging.info("MRR: {:.6f}".format(mrr_debiased.item()))
        for hit in hits:
            avg_count_ent_debiased = torch.mean((debiased_rank_ent <= hit).float())
            logging.info("Hits@ {}: {:.6f}".format(hit, avg_count_ent_debiased.item()))

    return mrr_debiased

# 关键函数 *** 构造 A --> B 和 B <-- A (其中<-- 为 --> 的镜像)
def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    def add_subject(e1, e2, r, d, num_rel):
        if not e2 in d:
            d[e2] = {}
        if not r + num_rel in d[e2]:
            d[e2][r + num_rel] = set()
        d[e2][r + num_rel].add(e1)

    def add_object(e1, e2, r, d, num_rel):
        if not e1 in d:
            d[e1] = {}
        if not r in d[e1]:
            d[e1][r] = set()
        d[e1][r].add(e2)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans

# 3.26 看看这个
def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)  # 过滤 valid 和 test 每天四元组
        all_ans_list.append(all_ans_t)

    return all_ans_list # [ { e1: {e2: (r) / r: (e2) } } ], len = uniq_t in given dataset  添加翻转后的三元组
#  按照天对数据进行保存
def split_by_time(data):
    snapshot_list = []  # 按照天对数据进行保存
    snapshot = [] # ((s, r, o))
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i] # [s, r, o, t]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        # edges: old order = [list of s, then list of o]; the new index of each old element in new order = [uniq_v]
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list

# 关键函数 *** 仅传入 1：contextid及时间  两个维度，以及2：K  给每个文档的K个主题，添加ont-shot后返回初始化特征
def split_context_by_time_onehot(data, k_contexts):
    onehot_matrix = []
    for context in range(k_contexts):
        onehot = [0] * k_contexts
        onehot[context] = 1
        onehot_matrix.append(onehot.copy())   # 基于K，为每个contextid构造one-hot编码  [1,0,0,0,0]

    snapshot_list = []
    snapshot = [] # (contextid, ...)
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][1]  # 天
        train = data[i] # [contextid, t]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(onehot_matrix[train[0]])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1
    return snapshot_list  # 2068天共有377430条数据，均将主题编码嵌入返回，仍返回2068行，每行为每天的四元组主题编码

def split_context_by_time_avg(data, k_contexts):
    avg_vector = np.ones(k_contexts) / k_contexts

    snapshot_list = []
    snapshot = [] # (contextid, ...)
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][1]
        train = data[i] # [contextid, t]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(avg_vector.copy())
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1
    return snapshot_list


def load_data(dataset):
    if dataset in ['EG', 'IS', 'IR']:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        return knwlgrh.load_from_local("../data_disentangled", dataset, load_context=True)


def ccorr(a, b):
    """
    Compute circular correlation of two tensors.
    Parameters
    ----------
    a: Tensor, 1D or 2D
    b: Tensor, 1D or 2D
    Notes
    -----
    Input a and b should have the same dimensions. And this operation supports broadcasting.
    Returns
    -------
    Tensor, having the same dimension as the input a.
    """
    return torch.fft.irfftn(torch.conj(torch.fft.rfftn(a, (-1))) * torch.fft.rfftn(b, (-1)), (-1))



def get_repeat_history(all_history_back_cont_embed, train_sample_num, find_cross_day, top_n):

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

    gold_list = []

    # 找相似度
    similarities_all = []
    for current_embed in current_concat_embed:
        similarities = cosine_similarity([current_embed], history_concat_embeds)[0]
        similarities_all.append(similarities)
        
    top_N_indices_all = []
    for similarities in similarities_all:
        top_N_indices = np.argsort(similarities)[-top_n:][::-1]  # 从高到低排序
        top_N_indices_all.append(top_N_indices)
        
    top_N_events_all = []
    for i, top_N_indices in enumerate(top_N_indices_all):
        gold_list.append(current_concat_embed[i])
        
        top_N_events = []
        for idx in top_N_indices:
            actor1 = history_all_embed.iloc[idx]['Actor1Name']
            relation = history_all_embed.iloc[idx]['EventCode']
            actor2 = history_all_embed.iloc[idx]['Actor2Name']
            timid = history_all_embed.iloc[idx]['timid']
            top_N_events.append((actor1, relation, actor2, timid))
        top_N_events_all.append(top_N_events)
        
    gold_tensor = torch.tensor(gold_list)

    return top_N_events_all, gold_tensor, top_N_indices_all


