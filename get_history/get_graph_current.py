import numpy as np
import os
import pickle
import dgl
import torch
from tqdm import tqdm
import argparse
from collections import defaultdict

def load_quadruples(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])

            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)  # 返回五元组及时间list

# 数据为五元组  + contextID   按照天进行输入
def get_data_with_t(data, tim):
    x = data[np.where(data[:,3] == tim)].copy()
    x = np.delete(x, 3, 1)  # drops time column
    return x

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm
#  # 例：t=0 && contextID ==0 作为 curr_triplets 待处理
def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e  
    r_to_e = defaultdict(set)  # 找到关系rel，链接的头尾实体，并做inverse操作
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx  # 当前天出现的关系，当前天关系切片汇总，当天出现的实体(4.8 +++ 这玩意有啥用来着？)

# 传入 [同一天的四元组数据（drop天）, 结点总数， 关系总数，K]
def get_big_graph(triples, num_nodes, num_rels):
    g_list = [] # len = K

    curr_triplets = triples  # 找到所有包含统一contextID的四元组
    if len(curr_triplets) == 0:  # 如果找不到
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        norm = comp_deg_norm(g).view(-1, 1)
        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
        g.ndata['id'] = node_id
        g.ndata['norm'] = norm
        g.uniq_r = np.array([])
        g.r_to_e = []
        g.r_len = []
        g_list.append(g)

    # inverse处理  curr_triplets ndarray src numpy.ndarray
    src, rel, dst = curr_triplets.transpose()
    src_double, dst_double = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_double = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src_double, dst_double)

    norm = comp_deg_norm(g).view(-1, 1)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)

    g.ndata['id'] = node_id
    g.ndata['norm'] = norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel_double)
    # 这里忽略了，回头看看                                # 例：t=0 && contextID ==0 作为 curr_triplets 待处理
    uniq_r, r_len, r_to_e = r2e(curr_triplets, num_rels)  # 当前天出现的关系，当前天关系切片汇总，当天出现的实体
    g.uniq_r = uniq_r  # t=0 && contextID ==0 时，出现的关系汇总
    g.r_to_e = r_to_e  # t=0 && contextID ==0 时，参与的实体array（不去重
    g.r_len = r_len  # t=0 && contextID ==0 时，基于关系汇总涉及结点数量做切片汇总（对应r_to_e

    g_list.append(g)

    return tuple(g_list)

# 构建实体到上下文对应的id  类似于实体-文档id 字典  键是实体ID,值是一个集合,包含该实体出现过的所有上下文ID 在处理上下文相关的知识图谱任务时非常有用
def get_entid2contextid(train_data):
    entid2contextid = dict()
    train_data = train_data.tolist()
    for idx, (head, rel, tail, time, contextid) in tqdm(enumerate(train_data), total=len(train_data)):
        if head not in entid2contextid:
            entid2contextid[head] = set()
        if tail not in entid2contextid:
            entid2contextid[tail] = set()
        entid2contextid[head].add(contextid)
        entid2contextid[tail].add(contextid)
    return entid2contextid

# 构建关系出现的上下文对应id，字典的键是关系ID,值是该关系出现过的所有上下文ID的集合， 一个问题，不同文档Id的K如何确定？？当前的K存在当前文档
def get_relid2contextid(train_data, num_r):
    relid2contextid = dict()
    train_data = train_data.tolist()
    for idx, (head, rel, tail, time, contextid) in tqdm(enumerate(train_data), total=len(train_data)):
        rel_rev = rel + num_r
        if rel not in relid2contextid:
            relid2contextid[rel] = set()
        if rel_rev not in relid2contextid:
            relid2contextid[rel_rev] = set()
        relid2contextid[rel].add(contextid)
        relid2contextid[rel_rev].add(contextid)
    return relid2contextid  # 225*2？ 包含关系、被包含关系？



def main(args):
    graph_dict = {}
    data_path = args.datapath

    train_data, train_times = load_quadruples(data_path, 'train.txt')
    val_data, val_times = load_quadruples(data_path, 'valid.txt')
    test_data, test_times = load_quadruples(data_path, 'test.txt')
    with open(os.path.join(data_path, 'stat.txt'), 'r') as f:
        line = f.readline()
        num_nodes, num_r = line.strip().strip('\n').split("\t")
        num_nodes = int(num_nodes)
        num_r = int(num_r)
    print(num_nodes, num_r)


    print('---generate knowledge graph')
    with tqdm(total=len(train_times), desc="Generating graphs for training") as pbar:
        for tim in train_times:
            data = get_data_with_t(train_data, tim)  # drop天后，取出同一天四元组数据
            graph_dict[tim] = get_big_graph(data, num_nodes, num_r)
            pbar.update(1)

    with tqdm(total=len(val_times), desc="Generating graphs for validating") as pbar:
        for tim in val_times:
            data = get_data_with_t(val_data, tim)
            graph_dict[tim] = get_big_graph(data, num_nodes, num_r)
            pbar.update(1)
        
    with tqdm(total=len(test_times), desc="Generating graphs for testing") as pbar:
        for tim in test_times:
            data = get_data_with_t(test_data, tim)
            graph_dict[tim] = get_big_graph(data, num_nodes, num_r)
            pbar.update(1)

    with open(os.path.join(data_path, 'graph_dict_each_context.pkl'), 'wb') as fp:
        pickle.dump(graph_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate disentangled graphs')
    parser.add_argument("--datapath", type=str, default="../data/EG",
                        help="disentangled dataset to generate disentangled graphs")
    args = parser.parse_args()

    main(args)
