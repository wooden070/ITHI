import argparse
import os
import sys
import pickle
import logging
import pandas as pd
import math
from torch.utils.tensorboard import SummaryWriter

import torch
import json
import numpy as np
from tqdm import tqdm
import random
from re_rank_model import get_batch_tensor
sys.path.append("..")
from src import utils
from src.ITHI import SeCo
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import gc

import warnings
warnings.filterwarnings("ignore")

debug_pth = "cross_day_graphs_debug"
def test(args, model, model_name,
         history_times, query_times, graph_dict, cross_day_graph_dict, top_N_indices_all, all_history_back_cont_embed, gold_tensor_all, related_top100_all,
         repeat_history_dict, test_list,
         all_ans_list, head_ents,
         use_cuda, mode='eval'):
    
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        logging.info("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint[
            'epoch']))  # use best stat checkpoint
        logging.info("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])

    rank_filter_list, mrr_filter_list = [], []
    tags, tags_all = [], []

    model.eval()

    with torch.no_grad():
        # 测试 258 天
        for time_idx, test_snap in enumerate(tqdm(test_list)):
            query_time = query_times[time_idx]
            query_idx = np.where(history_times == query_time)[0].item()
            input_time_list = history_times[query_idx - args.train_history_len: query_idx]
            history_glist = [graph_dict[tim] for tim in input_time_list]


            test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
            inverse_output = test_triples_input[:, [2, 1, 0]]  # 交换主客体位置  inverse操作
            inverse_output[:, 1] = inverse_output[:, 1] + model.num_rels
            test_triples_input = torch.cat((test_triples_input, inverse_output))
            test_triples_input = test_triples_input.to(args.gpu)
            
            
            cross_day_glist = None
            batch_tensor = get_batch_tensor(top_N_indices_all[query_idx], all_history_back_cont_embed, query_idx, 30)
            
            gold_tensor = gold_tensor_all[query_idx].float()
            related_top100 = related_top100_all[query_idx]
            repeat_history = repeat_history_dict[query_idx]

            top_N_events_all = model.re_rank_filter(batch_tensor, gold_tensor, related_top100)
            
            test_triples, final_score = model.predict(history_glist, cross_day_glist, top_N_events_all, repeat_history, test_triples_input, use_cuda)

            mrr_filter, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=3000)
            # ??这是个什么东西？  评估模型对于不同流行程度的实体的预测能力 合理
            popularity_tag = list(map(lambda x: utils.popularity_map(x, head_ents), test_triples))
            tags_all.append(popularity_tag)  # 统计有多少个三元组中包含常见头结点

            rank_filter_list.append(rank_filter)  #   # [n] 预测后尾节点倒序后排在多少位
            mrr_filter_list.append(mrr_filter)  # 
    mrr_filter_all = utils.cal_ranks(rank_filter_list, tags_all, mode)  # 得到去偏后的mrr

    return mrr_filter_all



def run_experiment(args):
    
    # load graph data
    print("loading graph data")
    data_pth = "../data/" + args.dataset
    data = utils.load_data(args.dataset)  # 加载train、valid、test 三种数据（四元组
    train_list = utils.split_by_time(
        data.train)  # [((s, r, o), ...), ...] len = num_date, do not contain inverse triplets  按照天为单位统计
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    train_times = np.array(sorted(set(data.train[:, 3])))
    val_times = np.array(sorted(set(data.valid[:, 3])))
    test_times = np.array(sorted(set(data.test[:, 3])))
    history_times = np.concatenate((train_times, val_times, test_times), axis=None)  # EG 共2584天


    num_nodes = data.num_nodes
    num_rels = data.num_rels
    print(num_nodes, num_rels)  # 2594 225

    # data for time-aware filtering  3.26 +++++ 按天、按sro及or's 转为dict
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes)
    # [ { e1: {r: (e2)} } ], len = uniq_t in given dataset


    # load popularity bias data  
    print("loading popularity bias data")
    head_ents = json.load(open(data_pth +'/head_ents.json', 'r'))


        
    # logging   7.2 +++ 日志
    score_naming = '_' + args.score_aggregation
    encoder_naming = '{}_n{}_h{}'.format(args.encoder, args.n_layers, args.n_hidden)
    train_naming = '_t{}_lr{}_wd{}'.format(args.train_history_len, args.lr, args.wd)
    model_name = encoder_naming + score_naming + train_naming  # rgcn_n2_h200_hgent1_hgrel1_hard_t3_lr0.001_wd1e-06
    log_path = '../results/{}'.format(args.dataset)
    filename = '../results/{}/{}{}.log'.format(
        args.dataset, model_name, args.alias)
    # ../results/EG/EG_LDA_K3/rgcn_n2_h200_hgent1_hgrel1_hard_t3_lr0.001_wd1e-06.log
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.INFO, filename=filename)
    run_path = '../runs_search' if args.param_search else '../runs'
    run_path += data_pth + "/" + model_name + args.alias
    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    run = SummaryWriter(run_path)
    
    
    
    # models
    model_path = '../models/{}'.format(args.dataset)
    model_state_file = model_path + '/' + model_name + args.alias
    if not os.path.isdir(model_path):
        os.makedirs(model_path)


    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    each_graph_dict = None
    print("================================")
    with open(os.path.join(data_pth, 'graph_dict_each_context.pkl'), 'rb') as fp:
        each_graph_dict = pickle.load(fp)



    # 加载 pkl 包含背景嵌入
    # back_cont_embed_path = '../data_disentangled/' + args.dataset+ '_' + debug_pth
    all_history_back_cont_embed = pd.read_pickle(data_pth + '/all_history_back_cont_embed.pkl')


    # 加载保存的内容
    save_data_pth = data_pth + "/train_re_rank_data"
    

    gold_tensor_all = {}
    get_score_dict = {}
    related_top100_all = {}
    top_N_indices_all = {}
    repeat_history_dict = {}
    
    for i in tqdm(range(args.split_flag)):
        # (all_history_back_cont_embed, train_sample_num, num_rels, find_cross_day, top_n):
        print("---------------------", i, "---------------------")
        with open(save_data_pth + '/related_top100_' + str(i+1) + '.pkl', 'rb') as f:
            related_top100 = pickle.load(f)  # 以天为单位的图
            related_top100_all = {**related_top100_all, **related_top100}
            del related_top100  # 清理内存
            gc.collect()  # 垃圾回收
        
        with open(save_data_pth + '/gold_tensor_' + str(i+1) + '.pkl', 'rb') as f:
            gold_tensor = pickle.load(f)  # 以天为单位的图
            gold_tensor_all = {**gold_tensor_all, **gold_tensor}
            del gold_tensor  # 清理内存
            gc.collect()  # 垃圾回收

        with open(save_data_pth + '/get_score_' + str(i+1) + '.pkl', 'rb') as f:
            get_score = pickle.load(f)  # 以天为单位的图
            get_score_dict = {**get_score_dict, **get_score}
            del get_score  # 清理内存
            gc.collect()  # 垃圾回收

        with open(save_data_pth + '/top_N_indices_' + str(i+1) + '.pkl', 'rb') as f:
            top_N_indices = pickle.load(f)  # 以天为单位的图
            top_N_indices_all = {**top_N_indices_all, **top_N_indices}
            del top_N_indices  # 清理内存
            gc.collect()  # 垃圾回收
            
            
        with open(save_data_pth + '/repeat_history_' + str(i+1) + '.pkl', 'rb') as f:
            repeat_history = pickle.load(f)  # 以天为单位的图
            repeat_history_dict = {**repeat_history_dict, **repeat_history}
            del repeat_history  # 清理内存
            gc.collect()  # 垃圾回收


    re_rank_model_pth = save_data_pth+'/re-rank_model'
    # create stat
    model = SeCo(args.decoder,
                 args.encoder,
                 num_nodes,
                 num_rels,
                 each_graph_dict,
                 args.re_rank_embed,
                 re_rank_model_pth,
                 None,
                 None,
                 args.cross_history_len,
                 args.n_hidden,
                 sequence_len=args.train_history_len,
                 num_bases=args.n_bases,
                 num_hidden_layers=args.n_layers,
                 dropout=args.dropout,
                 self_loop=args.self_loop,
                 layer_norm=args.layer_norm,
                 input_dropout=args.input_dropout,
                 hidden_dropout=args.hidden_dropout,
                 feat_dropout=args.feat_dropout,
                 use_cuda=use_cuda,
                 gpu=args.gpu)

    print(model)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    if args.continue_train:
        print("==========True==========")
        if use_cuda:
            checkpoint = torch.load(model_state_file, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
        logging.info("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint[
            'epoch']))  # use best stat checkpoint
        logging.info("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])



    # graph_dict_each_context.pkl
    if args.test and os.path.exists(model_state_file):
        print("----------------------------------------start testing----------------------------------------\n")
        test(args,
             model=model,
             model_name=model_state_file,
             history_times=history_times,
             query_times=test_times,
             graph_dict=each_graph_dict,
             cross_day_graph_dict=cross_day_graph_dict,
             all_history_back_cont_embed= all_history_back_cont_embed,
             num_rels = num_rels,
             test_list=test_list,
             all_ans_list=all_ans_list_test,
             head_ents=head_ents,
             use_cuda=use_cuda,
             mode="test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(
            model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_val_mrr, best_test_mrr = 0, 0
        best_epoch = 0
        accumulated = 0
        

        packed_node_input_list = None
        packed_node_input_dev_list = None
        packed_node_input_test_list = None


                 
                
        for epoch in range(args.n_epochs):
            model.train()
            losses = []

            idx = [_ for _ in range(len(train_list))]  # 总天数
            batch_cnt = len(idx)
            epoch_anchor = epoch * batch_cnt
            random.shuffle(idx)  # shuffle based on time  


            for batch_idx, train_sample_num in enumerate(tqdm(idx)):
                batch_anchor = epoch_anchor + batch_idx
                if train_sample_num < 20: continue  # make sure at least one history graph
                # train_list : [((s, r, o) on the same day)], len = uniq_t in train
                

                
                output = train_list[train_sample_num]  # all triplets in the next day to be predicted
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_times[0: train_sample_num]
                else:
                    input_list = train_times[train_sample_num - args.train_history_len: train_sample_num]   # 前h天为输入



                # generate history graph
                history_glist = [each_graph_dict[tim] for tim in input_list]  # [(g), ...], len = valid history length
                # cross_day_glist = packed_node_input_list[train_sample_num] # 4.8 +++ cross_day_graph   有問題的  跨天
                cross_day_glist = None
                output = torch.from_numpy(output).long().cuda() if use_cuda else torch.from_numpy(output).long()
                
                inverse_output = output[:, [2, 1, 0]]  # 交换主客体位置  inverse操作
                inverse_output[:, 1] = inverse_output[:, 1] + num_rels
                output_all = torch.cat((output, inverse_output))
                
                
                # batch_tensor = read_pickle(save_data_pth + '/batch_tensor/batch_tensor_'+str(train_sample_num)+'.pkl')
                batch_tensor = get_batch_tensor(top_N_indices_all[train_sample_num], all_history_back_cont_embed, train_sample_num, 30)
                
                gold_tensor = gold_tensor_all[train_sample_num].float()
                related_top100 = related_top100_all[train_sample_num]
                repeat_history = repeat_history_dict[train_sample_num]
                N =800
                if len(output_all) >N:
                    for i in range(int(len(output_all)/N)):
                        _strat = i*N
                        _end = (i+1)*N
                        loss = model(history_glist, cross_day_glist, batch_tensor[_strat:_end], gold_tensor[_strat:_end], related_top100[_strat:_end], repeat_history[_strat:_end], output_all[_strat:_end], use_cuda) 
                        # loss = model(history_glist, cross_day_glist, top_N_events_all, output, use_cuda)  

                        losses.append(loss.item())
                        run.add_scalar('loss/loss_all', loss.item(), batch_anchor)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                        optimizer.step()
                        optimizer.zero_grad()
                        # print(i)
                    if len(output_all)%N != 0:
                        _strat = (i+1)*N
                        _end = len(output_all)
                                        # 传入 [前三天的历史数据，t+1天三元组，is_cuda，t+1天contextid(ont-hot表示)]  8.16  
                        loss = model(history_glist, cross_day_glist, batch_tensor[_strat:_end], gold_tensor[_strat:_end], related_top100[_strat:_end], repeat_history[_strat:_end], output_all[_strat:_end], use_cuda) 
                        # loss = model(history_glist, cross_day_glist, top_N_events_all, output, use_cuda)  

                        losses.append(loss.item())
                        run.add_scalar('loss/loss_all', loss.item(), batch_anchor)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    # 传入 [前三天的历史数据，t+1天三元组，is_cuda，t+1天contextid(ont-hot表示)]  8.16  
                    loss = model(history_glist, cross_day_glist, batch_tensor, gold_tensor, related_top100, repeat_history, output_all, use_cuda) 
                    # loss = model(history_glist, cross_day_glist, top_N_events_all, output, use_cuda)  

                    losses.append(loss.item())
                    run.add_scalar('loss/loss_all', loss.item(), batch_anchor)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()

            print("Epoch {:04d}, AveLoss: {:.4f}, BestValMRR {:.4f}, BestTestMRR: {:.4f}, Model: {}, Dataset: {} "
                  .format(epoch, np.mean(losses), best_val_mrr, best_test_mrr, model_name, args.dataset))

            # # validation and test   all_history_back_cont_embed, num_rels,
            if (epoch + 1) and (epoch + 1) % args.evaluate_every == 0:
                val_mrr = test(args,
                               model=model,
                               model_name=model_state_file,
                               history_times=history_times,
                               query_times=val_times,
                               graph_dict=each_graph_dict,
                               cross_day_graph_dict=packed_node_input_dev_list,
                               top_N_indices_all = top_N_indices_all,
                               all_history_back_cont_embed = all_history_back_cont_embed,
                               gold_tensor_all=gold_tensor_all, 
                               related_top100_all=related_top100_all,
                               repeat_history_dict=repeat_history_dict,
                               test_list=valid_list,
                               all_ans_list=all_ans_list_valid,
                               head_ents=head_ents,
                               use_cuda=use_cuda,
                               mode="eval")
                run.add_scalar('val/mrr', val_mrr, epoch)

                test_mrr = test(args,
                                model=model,
                                model_name=model_state_file,
                                history_times=history_times,
                                query_times=test_times,
                                graph_dict=each_graph_dict,
                                cross_day_graph_dict=packed_node_input_test_list,
                                top_N_indices_all = top_N_indices_all,
                                all_history_back_cont_embed = all_history_back_cont_embed,
                                gold_tensor_all=gold_tensor_all, 
                                related_top100_all=related_top100_all,
                                repeat_history_dict=repeat_history_dict,                                
                                test_list=test_list,
                                all_ans_list=all_ans_list_test,
                                head_ents=head_ents,
                                use_cuda=use_cuda,
                                mode="eval")

                if val_mrr < best_val_mrr:
                    accumulated += 1
                    if epoch >= args.n_epochs:
                        print("Max epoch reached! Training done.")
                        break
                    if accumulated >= args.patience:
                        print("Early stop triggered! Training done at epoch{}, best epoch is {}".format(epoch,
                                                                                                        best_epoch))
                        break
                else:
                    accumulated = 0
                    best_val_mrr = val_mrr
                    best_test_mrr = test_mrr
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        print('--- test best epoch model at epoch {}'.format(best_epoch))
        test(args,
             model=model,
             model_name=model_state_file,
             history_times=history_times,
             query_times=test_times,
             graph_dict=each_graph_dict,
             cross_day_graph_dict=packed_node_input_test_list,
             top_N_indices_all = top_N_indices_all,
             all_history_back_cont_embed = all_history_back_cont_embed,
             gold_tensor_all=gold_tensor_all, 
             related_top100_all=related_top100_all,
             repeat_history_dict=repeat_history_dict,             
             test_list=test_list,
             all_ans_list=all_ans_list_test,
             head_ents=head_ents,
             use_cuda=use_cuda,
             mode="test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tri_ralate_EP')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("-d", "--dataset", type=str, default='IR',
                        help="which country's dataset to use: EG/ IR/ IS")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")

    # configuration for context
    parser.add_argument("--context", type=str, default='LDA',
                        help="context clustering method: LDA/ KMeans / GMM")
    parser.add_argument("--k_contexts", type=int, default=3,
                        help="number of contexts to disentangle the sub-embeddings")

    # configuration for cross-context hypergraph
    parser.add_argument("--score_aggregation", type=str, default='hard',
                        help="score aggregation strategy: hard/ avg")

    # configuration for context specific encoder
    parser.add_argument("--encoder", type=str, default="rgcn",
                        help="method of encoder: rgcn/ compgcn")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="number of propagation rounds")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n_hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--n_bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--self_loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer_norm", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input_dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden_dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat_dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train_history_len", type=int, default=1,
                        help="history length")
    parser.add_argument("--split_flag", type=int, default=7,
                        help="history length")

    # configuration for stat training
    parser.add_argument("--n_epochs", type=int, default=20,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--patience", type=int, default=5,
                        help="early stop patience")
    parser.add_argument("--evaluate_every", type=int, default=1,
                        help="perform evaluation every n epochs")
    parser.add_argument("--param_search", action='store_true', default=False,
                        help="perform parameter search, affects runs saving path")
    parser.add_argument("--continue_train", action='store_true', default=False)
    parser.add_argument("--alias", type=str, default='',
                        help="model naming alias, better start with _")
    parser.add_argument("--cross_history_len", type=int, default=5,
                        help="model naming alias, better start with _")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-6,
                        help="weight decay")
    parser.add_argument("--grad_norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--re_rank_embed", type=int, default=768,
                        help="norm to clip gradient to")
    # parser.add_argument("--test", type=bool, default=False,
    #                     help="whether test")

    args = parser.parse_args()
    print(args)

    run_experiment(args)
    sys.exit()
    
# nohup python main.py --dataset IS --train_history_len 3 --n_layers 3 --n_epochs 10 > ./log/t_3__n_3__IS.log 2>&1 &