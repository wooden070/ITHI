import json
import argparse
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import gensim
from gensim import corpora
from gensim.test.utils import datapath
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sentence_transformers.util import cos_sim
import pickle
Lda = gensim.models.ldamodel.LdaModel

SEED = 2023
# step:1  4.7 ++ /data_disentangled/EG_cross_day_graphs  存放五元组（其中文本用md5代替  保存5元组
# 源自generate_context_lda.py
def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
#             d[int(line[1])] = line[0]
            d[line[0]] = int(line[1])
    return d


def restore_sentences(word_lists):
    sentences = []
    for word_list in word_lists:
        sentence = ''
        for i, word in enumerate(word_list):
            if i == 0:
                sentence += word.capitalize()
            else:
                sentence += ' ' + word
            if i == len(word_list) - 1:
                sentence += '.'
        sentences.append(sentence)
    return sentences



def main(c):
    """"""
    print('------generate context embeding for country {}'.format(c))
    md5_list = json.load(open('../data/{}/md5_list.json'.format(c), 'r'))  # 96081
    docs_title_paragraph = json.load(open('../data/{}/docs_title_paragraph.json'.format(c), 'r'))

    md52docid = {}
    for idx, md5 in enumerate(md5_list):
        md52docid[md5] = idx

    data_df = pd.read_csv('../data/{}/{}.csv'.format(c, c), sep='\t')
    model = SentenceTransformer('/home/azmat/mr/gte-base')
    # tokenizer = AutoTokenizer.from_pretrained('/data/marong/py_code/pretrain_model/gte-large')
    

        # 4.16 +++   docs_cleaned_tokens   这个要保存起来，与数据集相关  *************
        #     格式如下：dict
        # key: [md5],  value: [title, title_embedding, context, context_embedding] (文本嵌入向量模型返回结果) 
        
    ent2id = _read_dictionary('../data/{}/entity2id.txt'.format(c))  # 实体有2594个 EG dataset
    rel2id = _read_dictionary('../data/{}/relation2id.txt'.format(c))  # 224 Eg 有点迷糊了 到底是关系还是事件?
    
    with open('../data/{}/dict_id2ont.json'.format(c), 'r') as f:
        rel2name = json.load(f)
        

    # 直接加载编码后的表示

    docs_cleaned_tokens = {}  # [ [token1, token2, ...], ...]

    for idx, md5 in tqdm(enumerate(md5_list), total=len(md5_list)):
        title_cont, context_cont = [], []
        doc_title_paragraph = docs_title_paragraph[idx]
        title_embed = model.encode(doc_title_paragraph[0])
        
        title_cont.append(doc_title_paragraph[0])
        title_cont.append(title_embed)  # title
        
        
        context_embed = model.encode(' '.join(doc_title_paragraph[1]))
        
        context_cont.append(' '.join(doc_title_paragraph[1]))
        context_cont.append(context_embed)
        
        docs_cleaned_tokens[md5] = [title_cont, context_cont]
    


    entity_embed_dict = {}
    relation_embed_dict = {}
    for key, value in rel2name.items():
        keys = int(value['id'])
        rel_name = value['name']
        reverse_rel_name = "Reverse " + value['name']
        relation_embed_dict[keys] = [model.encode(rel_name), model.encode(reverse_rel_name)]
        # print(key, value)
    
    for key, value in ent2id.items():
        entity_embed_dict[key] = model.encode(key)

        # csv 格式
        # Index(['Actor1Name', 'Actor2Name', 'EventCode', 'Actor1ADM1Code',
        #     'Actor2ADM1Code', 'EventADM1Code', 'date', 'timid', 'Md5_list'])

        # 'Title_embed'
        # 'Cont_embed'
        # 'Head_entity_embed'
        # 'Tail_entity_embed'
        # 'Relation_embed'
        # 'Reverse_relation_embed'


    new_rows = []
    # 从五元组的 md_list 的 key 获取
    for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
        md5 = row['Md5_list']
        
        md5s = md5.split(', ')
        if len(md5s) == 0:
            title_embed = docs_cleaned_tokens[md5s[0]][0][1]
            cont_embed = docs_cleaned_tokens[md5s[0]][1][1]
        else:
            title_embed = []
            cont_embed = []
            title = []
            for md in md5s:
                title_embed.append(docs_cleaned_tokens[md][0][1].copy())
                title.append(docs_cleaned_tokens[md][0][0])
                cont_embed.append(docs_cleaned_tokens[md][1][1].copy())
                
            title_embed = np.array(title_embed)
            title_embed = np.sum(title_embed, axis=0) / len(md5s)
            
            cont_embed = np.array(cont_embed)
            cont_embed = np.sum(cont_embed, axis=0) / len(md5s)
            
        row['Title'] = title
        row['Title_embed'] = title_embed
        row['Cont_embed'] = cont_embed
        row['Head_entity_embed'] = entity_embed_dict[row['Actor1Name']]
        row['Tail_entity_embed'] = entity_embed_dict[row['Actor2Name']]
        row['Relation_embed'] = relation_embed_dict[row['EventCode']][0]
        row['Reverse_relation_embed'] = relation_embed_dict[row['EventCode']][1]
        row['Actor1Name'] = ent2id[row['Actor1Name']]
        row['Actor2Name'] = ent2id[row['Actor2Name']]
        
        row['EventCode'] = rel2id[str(row['EventCode'])]
        new_rows.append(row.to_dict())

    history_back_cont_embed = pd.DataFrame(new_rows)

    

    

    # deal_data_df.to_pickle(save_path + '/all_history_back_cont_embed.pkl')

    
    # 9.1 修改
    # save_path = '../data/{}/'.format(c)
    # history_back_cont_embed = pd.read_pickle(save_path + '/all_history_back_cont_embed.pkl')

    # rel2id = _read_dictionary('../data/{}/relation2id.txt'.format(c))  # 224 Eg 有点迷糊了 到底是关系还是事件?
    
    
    history_back_embed_forward = history_back_cont_embed[['Actor1Name', 'EventCode', 'Actor2Name', 'timid', 'Title', 'Cont_embed', 'Head_entity_embed', 'Relation_embed']]
    history_back_embed_back = history_back_cont_embed[['Actor1Name', 'EventCode', 'Actor2Name', 'timid', 'Title', 'Cont_embed', 'Head_entity_embed', 'Reverse_relation_embed']]
    tmp_df_back = history_back_embed_back.copy()
    history_back_embed_back.loc[:, 'EventCode'] = tmp_df_back['EventCode'] + len(rel2id)
    history_back_embed_back.loc[:, 'Actor1Name'] = tmp_df_back['Actor2Name']
    history_back_embed_back.loc[:, 'Actor2Name'] = tmp_df_back['Actor1Name']
    history_back_embed_back.rename(columns={'Reverse_relation_embed': 'Relation_embed'}, inplace=True)

    history_all_embed = pd.concat([history_back_embed_forward, history_back_embed_back])
    
    save_path = '../data/{}/'.format(c)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    history_all_embed.to_pickle(save_path + '/all_history_back_cont_embed.pkl')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate event and structured data with context information')
    parser.add_argument("--c", type=str, default="IR",
                        help="country: EG, IR, or IS")
    args = parser.parse_args()

    main(args.c)
    