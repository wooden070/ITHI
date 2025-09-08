import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl
from src.decoder import ConvTransE
from src.aggregator import Aggregator, RGCNAggregator

import pickle
from re_rank_model import re_rank_model
# 7.2 模型 +++
class SeCo(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, graph_dict, re_rank_embed, re_rank_model_pth,
                 hyper_adj_ent, hyper_adj_rel, cross_seq_len,
                 h_dim, sequence_len, num_bases=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, layer_norm=False,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0, use_cuda=False,
                 gpu=0):
        super(SeCo, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.hyper_adj_ent = hyper_adj_ent
        self.hyper_adj_rel = hyper_adj_rel


        self.num_ents_dis = num_ents
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.converse_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.relation_evolve = False
        self.emb_rel = None
        self.gpu = gpu
        self.graph_dict = graph_dict
        self.cross_seq_len = cross_seq_len
        self.emb_rel = torch.nn.Parameter(torch.Tensor(2, self.num_rels * 2, self.h_dim),
                                          requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(2, self.num_ents, self.h_dim),
                                              requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.loss_e = torch.nn.CrossEntropyLoss()

        self.aggregators = torch.nn.ModuleList()  # one rgcn for each context

        self.aggregators.append(Aggregator(
            h_dim,
            num_ents,
            num_rels * 2,
            num_bases,
            num_hidden_layers,
            encoder_name,
            self_loop=self_loop,
            dropout=dropout,
            use_cuda=use_cuda))
        # 4.8 +++
        self.cross_aggregators = torch.nn.ModuleList()  # one rgcn for each context
        self.cross_aggregators.append(Aggregator(
            h_dim,
            num_ents,
            num_rels * 2,
            num_bases,
            num_hidden_layers,
            encoder_name,
            self_loop=self_loop,
            dropout=dropout,
            use_cuda=use_cuda))
        self.dropout = nn.Dropout(dropout)

        self.time_gate_weights = torch.nn.ParameterList()
        self.time_gate_biases = torch.nn.ParameterList()

        self.aggregator_rgcn = RGCNAggregator(self.h_dim, self.converse_dim, 0.1,
                                              self.num_rels, self.encoder_name, 
                                              seq_len=self.cross_seq_len, gpu=self.gpu)

        self.time_gate_weights.append(nn.Parameter(torch.Tensor(h_dim, h_dim)))
        nn.init.xavier_uniform_(self.time_gate_weights[0], gain=nn.init.calculate_gain('relu'))
        self.time_gate_biases.append(nn.Parameter(torch.Tensor(h_dim)))
        nn.init.zeros_(self.time_gate_biases[0])

        # GRU cell for relation evolving
        self.relation_gru_cells = torch.nn.ModuleList()

        self.relation_gru_cells.append(nn.GRUCell(self.h_dim * 2, self.h_dim))
        # The number of expected features in the input x; The number of features in the hidden state h

        # cross day GRU
        self.cross_relation_gru_cells = torch.nn.ModuleList()

        self.cross_relation_gru_cells.append(nn.GRUCell(self.h_dim * 2, self.h_dim))
        self.sub_encoder = nn.GRU(self.h_dim, self.h_dim, batch_first=True) 
        # decoder
        if decoder_name == "convtranse":
            self.decoders = torch.nn.ModuleList()
            self.decoders.append(ConvTransE(num_ents, h_dim, self.num_rels, input_dropout, hidden_dropout, feat_dropout))
        else:
            raise NotImplementedError
        self.re_rank_model = re_rank_model(in_features=re_rank_embed*6, out_features = re_rank_embed)
        self.re_rank_model.load_state_dict(torch.load(re_rank_model_pth))
        
    # 得到结点和边的编码应该是  3.26 +++  到这里了，回头看看三个子图代表什么
    def get_embs(self, g_list, cross_day_glist, use_cuda):
        def move_dgl_to_cuda(g):
            g.to("cuda:"+str(torch.cuda.current_device()))
        # dynamic_emb entity embedding matrix H is global, but is normalized before every forward  [K, ent, hidden]
        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]  # 对全局矩阵归一化处理

        ent_emb_each, rel_emb_each = [], []
        # 

        ent_emb_context = self.h[0, :, :]
        rel_emb_context = self.emb_rel[0, :, :]
        
        
        for timid, g_each in enumerate(g_list):
            g = g_each[0].to(self.gpu)
            if len(g.r_len) == 0:
                continue

            temp_e = ent_emb_context[g.r_to_e]  # 实体表示
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(
                self.num_rels * 2, self.h_dim).float()  # 关系矩阵  何用？
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]  # all entities related to a relation
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean

            x_input = torch.cat((rel_emb_context, x_input), dim=1)
            rel_emb_context = self.relation_gru_cells[0](x_input, rel_emb_context)  # new input hidden = h'
            rel_emb_context = F.normalize(rel_emb_context) if self.layer_norm else rel_emb_context  # 关系汇聚

            curr_ent_emb_context = self.aggregators[0].forward(g, ent_emb_context,
                                                                       rel_emb_context)  # aggregated node embedding
            curr_ent_emb_context = F.normalize(curr_ent_emb_context) if self.layer_norm else curr_ent_emb_context

            time_weight = torch.sigmoid(
                torch.mm(ent_emb_context, self.time_gate_weights[0]) + self.time_gate_biases[0])
            ent_emb_context = time_weight * curr_ent_emb_context + (1 - time_weight) * ent_emb_context  # 历史连续三元组表示

            ent_emb_each.append(ent_emb_context )
            rel_emb_each.append(rel_emb_context)
            
        ent_emb_each = torch.stack(ent_emb_each)  # k, num_ent, h_dim
        rel_emb_each = torch.stack(rel_emb_each)  # k, num_rel * 2, h_dim


        return ent_emb_each, rel_emb_each



    def re_rank_filter(self, batch_tensor, gold_tensor, related_top100):
        gold_tensor_expanded = gold_tensor.expand(-1, batch_tensor.shape[1], -1).cuda()
        batch_tensor = batch_tensor.cuda()
        scores = self.re_rank_model.forward(batch_tensor, gold_tensor_expanded)
        topk_scores, topk_indices = torch.topk(scores, 50, dim=1)
        related_top100 = torch.tensor(related_top100).cuda()
        top_50_quadruples = torch.gather(related_top100, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 4))
        return top_50_quadruples.tolist()

    # # 传入 [前三天的历史数据，t+1天三元组，is_cuda，t+1天contextid(ont-hot表示)]
    def predict(self, glist, cross_day_glist, top_N_events_all, repeat_history, test_triplets, use_cuda):
        """"""
        # 4.29 +++ 先看单向效果
        # 交换主客体位置  inverse操作    
        # inverse_test_triplets = test_triplets[:, [2, 1, 0]]  # 交换主客体位置  inverse操作
        # inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels
        # all_triples = torch.cat((test_triplets, inverse_test_triplets))
        all_triples = test_triplets

        # 得到节点和边的特征表示（按主题进行处理） [k, num_ent, h_dim]  [k, num_rel * 2, h_dim]
        e_emb, r_emb = self.get_embs(glist, cross_day_glist, use_cuda)
        pre_emb = F.normalize(e_emb, dim=-1) if self.layer_norm else e_emb  # [k, n_ents, h_dim]

        all_score_ob = []  # object score using each context, len = k_contexts
        for context in range(1):
            # use sub-embeddings under the context
            pre_emb_context = pre_emb[context, :, :]  # [n_ents, h_dim]
            r_emb_context = r_emb[context, :, :]
            
            
            all_score_ob.append(self.decoders[context].forward(
                pre_emb_context, r_emb_context, top_N_events_all, repeat_history, all_triples).view(-1, self.num_ents))  # [n_triplets, n_ents]

        return all_triples, all_score_ob[0]  # 返回inverse后的所有三元组 及 E1 + R，各种操作后的[N, 2594]维特征
    
    
    def read_top_N_pickle(self, top_n_pth):
        with open(top_n_pth, 'rb') as file:
            batch_tensor_dict = pickle.load(file) 
        return batch_tensor_dict    

        # 7.25 ++++ 确定 all_score_ob 维度
    def forward(self, glist, cross_day_glist,  batch_tensor, gold_tensor, related_top100, repeat_history, test_triples, use_cuda):

        top_N_events_all = self.re_rank_filter(batch_tensor, gold_tensor, related_top100)
        
        all_triples, final_score_ob = self.predict(glist, cross_day_glist, top_N_events_all, repeat_history, test_triples, use_cuda)
        loss_ent = self.loss_e(final_score_ob, all_triples[:, 2])

        return loss_ent
