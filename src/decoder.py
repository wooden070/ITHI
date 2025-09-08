from torch.nn import functional as F
import dgl.function as fn
import dgl
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import os
from layers import RGCNBlockLayer
# from aggregator import Aggregator
path_dir = os.getcwd()
  # decode
class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, num_rels, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(6, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(6)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim*3)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim*3)
        # self.bn3 = torch.nn.BatchNorm1d(embedding_dim*3)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)
        self.num_rels = num_rels
        self.num_nodes = num_entities
        self.relate_emb_rel = torch.nn.Parameter(torch.Tensor(1, self.num_rels * 2, embedding_dim),
                        requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.relate_emb_rel)
        self.relate_dynamic_emb = torch.nn.Parameter(torch.Tensor(1, num_entities, embedding_dim),
                        requires_grad=True).float()
        torch.nn.init.normal_(self.relate_dynamic_emb)
        
        
        self.repeat_emb_rel = torch.nn.Parameter(torch.Tensor(1, self.num_rels * 2, embedding_dim),
                        requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.repeat_emb_rel)
        self.repeat_dynamic_emb = torch.nn.Parameter(torch.Tensor(1, num_entities, embedding_dim),
                        requires_grad=True).float()
        torch.nn.init.normal_(self.repeat_dynamic_emb)
    
    
        self.rgcn1 = UnionRGCNLayer(embedding_dim, embedding_dim, 2*self.num_rels,
                        activation=F.rrelu, dropout=feature_map_dropout)
        self.rgcn2 = UnionRGCNLayer(embedding_dim, embedding_dim, 2*self.num_rels,
                           activation=F.rrelu, dropout=feature_map_dropout)
        
        self.rgcn3 = UnionRGCNLayer(embedding_dim, embedding_dim, 2*self.num_rels,
                        activation=F.rrelu, dropout=feature_map_dropout)
        self.rgcn4 = UnionRGCNLayer(embedding_dim, embedding_dim, 2*self.num_rels,
                           activation=F.rrelu, dropout=feature_map_dropout)
        
    def comp_deg_norm(self, g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()  # 计算每个结点的入度
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1  # 处理入度为0的情况
        norm = 1.0 / in_deg
        return norm
    def move_dgl_to_cuda(self, g):
        g.to("cuda:"+str(torch.cuda.current_device()))
        
        
    def get_relate_embed(self, relate_history_graph, triplets):
        relate_ent_emb, relate_rel_emb = [], []
        # 处理K个主题（节点和边 两方面考虑

        ent_emb_context = self.relate_dynamic_emb[0, :, :]
        rel_emb_context = self.relate_emb_rel[0, :, :]
        # ent_emb_context = self.move_dgl_to_cuda(ent_emb_context)
        # rel_emb_context = self.move_dgl_to_cuda(rel_emb_context)
        # relate_g_list = []
        relate_head_list = []
        relate_rel_list = []
        for _i, relate_history in enumerate(relate_history_graph):
            
            triples = []
            for relate in relate_history:
                triples.append(list(relate[:3]))
            src, rel, dst = zip(*triples)
            
            # 将三元组转换为张量
            src = torch.tensor(src)
            rel = torch.tensor(rel)
            dst = torch.tensor(dst)
            
            g = dgl.DGLGraph()
            g.add_nodes(self.num_nodes)  # 2594个结点
            g.add_edges(src, dst)  # 还是双向边？是否合理？
            norm = self.comp_deg_norm(g)  # 计算结点的度的倒数，0 写了规则屏蔽
            node_id = torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1)
            g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})  # 更新图节点及规范化因子
            g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})  # 计算边的规范化因子
            g.edata['type'] = torch.LongTensor(rel)
            g = g.to("cuda:"+str(torch.cuda.current_device()))
            
            g.ndata['h'] = ent_emb_context
            # g.edata['h'] = rel_emb_context
            self.rgcn1(g, ent_emb_context, rel_emb_context)
            self.rgcn2(g, ent_emb_context, rel_emb_context)
            embeds = g.ndata.pop('h')[triplets[_i][0]]
            relate_head_list.append(embeds)
            relate_rel_list.append(rel_emb_context[triplets[_i][1]])
        relate_head_list = torch.tanh(torch.stack(relate_head_list, dim=0))
        relate_rel_list = torch.tanh(torch.stack(relate_rel_list, dim=0))
        return relate_head_list, relate_rel_list, ent_emb_context
    
    def get_repeat_history_embed(self, repeat_historys, triplets):
        # 处理K个主题（节点和边 两方面考虑

        ent_emb_context = self.repeat_dynamic_emb[0, :, :]
        rel_emb_context = self.repeat_emb_rel[0, :, :]
        # repeat_g_list = []
        repeat_head_list = []
        repeat_rel_list = []
        for _i, repeat_history in enumerate(repeat_historys):
            if len(repeat_history)==0:
                repeat_head_list.append(ent_emb_context[triplets[_i][0]])
                repeat_rel_list.append(rel_emb_context[triplets[_i][1]])
            else:
                triples = []
                for relate in repeat_history:
                    triples.append(list(relate[:3]))
                src, rel, dst = zip(*triples)
                
                # 将三元组转换为张量
                src = torch.tensor(src)
                rel = torch.tensor(rel)
                dst = torch.tensor(dst)
                
                g = dgl.DGLGraph()
                g.add_nodes(self.num_nodes)  # 2594个结点
                g.add_edges(src, dst)  # 还是双向边？是否合理？
                norm = self.comp_deg_norm(g)  # 计算结点的度的倒数，0 写了规则屏蔽
                node_id = torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1)
                g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})  # 更新图节点及规范化因子
                g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})  # 计算边的规范化因子
                g.edata['type'] = torch.LongTensor(rel)
                g = g.to("cuda:"+str(torch.cuda.current_device())) 
                g.ndata['h'] = ent_emb_context

                
                # g.edata['h'] = rel_emb_context
                self.rgcn3(g, ent_emb_context, rel_emb_context)
                self.rgcn4(g, ent_emb_context, rel_emb_context)
                
                embeds = g.ndata.pop('h')[triplets[_i][0]]
                repeat_head_list.append(embeds)
                repeat_rel_list.append(rel_emb_context[triplets[_i][1]])
        repeat_head_list = torch.tanh(torch.stack(repeat_head_list, dim=0))
        repeat_rel_list = torch.tanh(torch.stack(repeat_rel_list, dim=0))
        return repeat_head_list, repeat_rel_list, ent_emb_context
    
    # 传入 当前K [结点表示，关系表示，所有label三元组（inverse后的）]   # 这里是否可以加个东西？
    def forward(self, embedding, emb_rel, relate_history_graph, repeat_history, triplets):

        relate_triplet_list, relate_rel_list, relate_embed_all = self.get_relate_embed(relate_history_graph, triplets)
        repeat_head_list, repeat_rel_list, repeat_embed_all = self.get_repeat_history_embed(repeat_history, triplets)
        
        # relate_triplet_list = torch.stack(relate_triplet_list, dim=0)
        # relate_rel_list = torch.stack(relate_rel_list, dim=0)
        relate_head = relate_triplet_list.unsqueeze(1)
        relate_rel = relate_rel_list.unsqueeze(1)
        stacked_relate = torch.cat([relate_head, relate_rel], 1)


        repeat_head = repeat_head_list.unsqueeze(1)
        repeat_rel = repeat_rel_list.unsqueeze(1)
        stacked_repeat = torch.cat([repeat_head, repeat_rel], 1)
        
        # 原始  头 cat 关系表示
        embedded_all = torch.tanh(embedding) # [num_entity, h_dim]
        relate_embed_all = torch.tanh(relate_embed_all)
        repeat_embed_all = torch.tanh(repeat_embed_all)
        
        
        batch_size = len(triplets)  # 总个数固定？
        e1_embedded = embedded_all[triplets[:, 0]].unsqueeze(1) # [num_triplets, 1, h_dim]
        e2_embedded = embedded_all[triplets[:, 2]] # [num_triplets, h_dim]
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1) # [num_triplets, 1, h_dim]
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1) # [num_triplets, 2, h_dim]
        
        stacked_inputs_cat = torch.cat([stacked_inputs, stacked_repeat, stacked_relate], 1)
        
        stack_embedded_all = torch.cat([embedded_all, repeat_embed_all, relate_embed_all], 1)  # [2594, 600]
        
        
        stacked_inputs_cat = self.bn0(stacked_inputs_cat)  # N, 4, 200
        x = self.inp_drop(stacked_inputs_cat)  # N, 4, 200
        x = self.conv1(x) # [num_triplets, channels, h_dim]  
        x = self.bn1(x)  # N, 50, 200
        x = F.relu(x)  # N, 50, 200
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1) # [num_triplets, channels*h_dim]
        x = self.fc(x) # [num_triplets, h_dim]
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)  # [num_triplets, h_dim]
        query = F.relu(x) # [num_triplets, h_dim]
        x = torch.mm(query, stack_embedded_all.transpose(1, 0))  # N, 2594
                    # N, 600   600, 2594

        return x


"""

(self.h_dim, self.t_dim, self.renet_dropout_rate,
self.num_entities, self.num_relations, self.encoder_name, 
phase=self.phase, seq_len=self.seq_len, gpu=self.gpu)

"""


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, dropout=0.0, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.num_rels = num_rels
        self.rel_emb = None
        self.ob = None
        self.sub = None


        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))


        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    # GNN 参数更新  7.25 +++  头+关系  更新尾节点吗
    def propagate(self, g):
      g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)  

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # 结点表示self-loop

        loop_message = torch.mm(g.ndata['h'], self.loop_weight)
        # if len(prev_h) != 0:
        #     skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']  # 融合batch_graph  结点+边的表示  一部分

        # print(len(prev_h))
        node_repr = node_repr + loop_message  # 传播一次后的邻居节点信息+self-loop后的结点表示

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr  # 得到全图中的节点表示：连续历史图全图+邻居节点一次推理及self-loop后的表示

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)

        msg = node + relation

        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}